/*************************************************************************
 * Growing union RS (NCCL_CCD_UNION_MASK=growing_union, fp32)
 *
 * A ring reduce-scatter with no mask exchange. Each RS hop gets A, the
 * partial sum of the slice so far as {value, slice index} pairs plus the mask
 * of the indices they cover, and B, this rank's dense slice of sendbuff:
 *   1. every pair of A gets B[idx] added (ccd_union_pairs_add, ccd.cuh);
 *   2. B's nonzeros that A doesn't cover (B & ~A) are appended as pairs
 *      (ccd_growing_union_compress);
 * and the hop sends [mask (A | B) | pairs] on as an MPAIR slice. The first hop
 * is the same with A empty. The final receive scatter-adds the pairs into the
 * owner's shard. Every index is summed in ring order, as in dense RS.
 * When SpCCL's format choice says DENSE, the slice goes on dense instead
 * (prims_simple.h), and stays dense for the rest of the ring.
 *
 * Step 2 works on one mask in shared memory (the hop's scratch):
 *   stage    A's mask is copied from the received payload (which may be in the
 *            previous GPU's memory) into shared memory, 16 bytes per load;
 *   pass 1   B's mask, 1024 elements per warp per iteration: 8 float4 loads
 *            per lane, each warp load 512 contiguous bytes, and 16 lanes OR
 *            their bits into one 64-bit word with shuffles (the test's
 *            extractBitmask); the slice's last < 1024 elements 32 per thread;
 *            new = B & ~A overwrites A's word, A | B goes into the outgoing mask;
 *   count    nonzeros of new per tile of 4096 elements;
 *   scan     tile counts -> where each tile's pairs start (after A's nA pairs);
 *   pack     SpCCL's second pass on the new bits: {B[e], e} pairs, tile by
 *            tile, in element order within a tile.
 * B's slice must start 16-byte aligned (recvcount a multiple of 4).
 * NCCL_CCD_UNION_MASK=growing_union_v2 sends the same slices with the hop's
 * steps in another order (ccd_growing_union_v2_hop, at the end of the file).
 *
 * Payload of an MPAIR slice of n elements, after the CcdSparseChunkHeader:
 *   [mask: bit e%64 of word e/64 <-> element e, padded to 16 bytes]
 *   [nnz pairs: uint2 {fp32 bits, slice index}]           at vals_offset
 ************************************************************************/

#ifndef NCCL_DEVICE_CCD_GROWING_UNION_CUH
#define NCCL_DEVICE_CCD_GROWING_UNION_CUH

#include "ccd.cuh"

#define CCD_GU_LOADS 8          // float4 loads in flight per lane in pass 1: 1024 elements (16 mask words) per warp
#define CCD_GU_GROUP 32         // elements per thread in pass 1's tail: 8 float4 -> one 32-bit mask word
#define CCD_GU_STAGE_UNROLL 4   // 16-byte loads in flight per thread when staging A's mask

// Bytes of the mask of an n-element slice, i.e. where its pairs start.
__device__ __forceinline__ size_t ccd_growing_union_vals_offset(size_t n) {
  return ccd_pad16(ceil_div(n, 64) * sizeof(uint64_t));
}

// Pairs in a received MPAIR slice (payload at `in`); traps on another format.
__device__ __forceinline__ uint32_t ccd_growing_union_nnz(const char* in, int tid) {
  const volatile CcdSparseChunkHeader* hdr =
      (const volatile CcdSparseChunkHeader*)(in - sizeof(CcdSparseChunkHeader));
  if (tid == 0 && hdr->format != (size_t)CcdCompressionProtocol::MPAIR) {
    printf("CCD UNION: rank %d ch %d: got format %lu, expected MPAIR (growing union)\n",
           ncclShmem.comm.rank, ncclShmem.channelId, (unsigned long)hdr->format);
    __trap();
  }
  return (uint32_t)hdr->nnz;
}

// Copies `chunks` 16-byte chunks of A's mask from `in` (may be another GPU's
// buffer) to `mask` (shared memory). Returns this thread's popcount of what it
// copied if `count` is set, else 0.
__device__ __forceinline__ unsigned ccd_growing_union_stage(
    uint4* mask, const uint4* in, size_t chunks, bool count, int tid, int nworkers
) {
  constexpr int U = CCD_GU_STAGE_UNROLL;
  unsigned c = 0;
  for (size_t i0 = tid; i0 < chunks; i0 += U * nworkers) {
    uint4 v[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t i = i0 + u * nworkers;
      v[u] = i < chunks ? __ldcg(in + i) : make_uint4(0, 0, 0, 0);
    }
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const size_t i = i0 + u * nworkers;
      if (i < chunks) mask[i] = v[u];
      if (count) c += __popc(v[u].x) + __popc(v[u].y) + __popc(v[u].z) + __popc(v[u].w);
    }
  }
  return c;
}

// Nonzero bits of elements [32g, 32g+32) of B's slice (bit i <-> element 32g+i,
// zero past n): 8 float4 loads, the last partial group element by element.
__device__ __forceinline__ uint32_t ccd_growing_union_bits(const float* dense, size_t n, size_t g) {
  const size_t e0 = g * CCD_GU_GROUP;
  uint32_t bits = 0;
  if (e0 + CCD_GU_GROUP <= n) {
    const float4* p = (const float4*)(dense + e0);
    float4 v[CCD_GU_GROUP / 4];
    #pragma unroll
    for (int k = 0; k < CCD_GU_GROUP / 4; k++) v[k] = p[k];
    #pragma unroll
    for (int k = 0; k < CCD_GU_GROUP / 4; k++) {
      bits |= ((uint32_t)(v[k].x != 0.0f) << (4 * k)) | ((uint32_t)(v[k].y != 0.0f) << (4 * k + 1))
            | ((uint32_t)(v[k].z != 0.0f) << (4 * k + 2)) | ((uint32_t)(v[k].w != 0.0f) << (4 * k + 3));
    }
  } else {
    for (size_t e = e0; e < n; e++) bits |= (uint32_t)(dense[e] != 0.0f) << (e - e0);
  }
  return bits;
}

// Step 2 of a hop on an n-element slice: pass 1, count, scan and pack (see
// the top of the file). `mask` (shared memory, vals_offset bytes) holds A's
// mask if haveA; `inds` (shared memory) gets the tile offsets. The pairs go
// after the nA pairs of A already in `pairs`. Returns nA + the pairs packed.
__device__ __forceinline__ unsigned ccd_growing_union_compress(
    const float* dense, size_t n, uint2* pairs, uint32_t* outMask,
    uint32_t* mask, bool haveA, unsigned* inds, unsigned nA,
    int tid, int nworkers, int bar
) {
    const unsigned lane      = tid % 32;
    const unsigned warp      = tid / 32;
    const unsigned num_warps = nworkers / 32;
    const size_t words32     = ccd_growing_union_vals_offset(n) / sizeof(uint32_t);
    const size_t words       = words32 / 2;  // 64-bit mask words, including the padding
    const size_t total_tiles = ceil_div(n, CCD_UNION_TILE);

    // pass 1: B's mask; new = B & ~A stays in shared memory, A | B is sent on.
    // A warp takes 1024 elements at a time: load u covers 128 of them (512
    // contiguous bytes), lanes 0-15 the first 64 (word 2u), lanes 16-31 the
    // next 64 (word 2u+1), and each 16 lanes OR their 4 bits into the word.
    // Lane u then writes word 2u and lane 16+u word 2u+1, so the 16 words go
    // to shared memory and the outgoing mask as 128 contiguous bytes.
    {
        constexpr int U = CCD_GU_LOADS;
        constexpr size_t Step = 32 * 4;     // elements per warp load
        constexpr size_t Chunk = Step * U;  // elements per warp iteration
        uint64_t* mask64 = (uint64_t*)mask;
        uint64_t* outMask64 = (uint64_t*)outMask;
        const unsigned sub = lane % 16;
        const size_t nChunks = n / Chunk;
        for (size_t c = warp; c < nChunks; c += num_warps) {
            const size_t base = c * Chunk;
            float4 v[U];
            #pragma unroll
            for (int u = 0; u < U; u++) v[u] = ((const float4*)(dense + base + u * Step))[lane];
            uint64_t mine = 0;
            #pragma unroll
            for (int u = 0; u < U; u++) {
                unsigned long long bits = (unsigned long long)((v[u].x != 0.0f) | ((v[u].y != 0.0f) << 1)
                                        | ((v[u].z != 0.0f) << 2) | ((v[u].w != 0.0f) << 3)) << (4 * sub);
                #pragma unroll
                for (int off = 1; off < 16; off <<= 1) bits |= __shfl_xor_sync(0xffffffffu, bits, off);
                if (sub == u) mine = bits;
            }
            if (sub < U) {
                const size_t w = base / 64 + 2 * sub + lane / 16;
                const uint64_t a = haveA ? mask64[w] : 0;
                mask64[w] = mine & ~a;
                outMask64[w] = a | mine;
            }
        }
        // The rest (fewer than Chunk elements, and the mask's padding): 32 elements per thread.
        for (size_t g = nChunks * (Chunk / 32) + tid; g < words32; g += nworkers) {
            const uint32_t b = ccd_growing_union_bits(dense, n, g);
            const uint32_t a = haveA ? mask[g] : 0;
            mask[g] = b & ~a;
            outMask[g] = a | b;
        }
    }
    barrier_sync(bar, nworkers);

    // count: lane takes words 2*lane and 2*lane+1 of the tile (64 words, 4096 elements)
    const uint64_t* bv = (const uint64_t*)mask;
    for (size_t tile_index = warp; tile_index < total_tiles; tile_index += num_warps) {
        const size_t w = tile_index * 64 + 2 * lane;
        unsigned total_popc = (w < words ? __popcll(bv[w]) : 0) + (w + 1 < words ? __popcll(bv[w + 1]) : 0);
        for (unsigned off = 16; off > 0; off >>= 1) {
            total_popc += __shfl_down_sync(0xFFFFFFFF, total_popc, off);
        }
        if (lane == 0) {
            inds[tile_index] = total_popc;
        }
    }
    barrier_sync(bar, nworkers);

    // scan (SpCCL's): the packed pairs go after A's
    if (tid == 0) {
        unsigned running = nA;
        for (unsigned tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
            unsigned count = inds[tile_idx];
            inds[tile_idx] = running;
            running += count;
        }
        inds[total_tiles] = running;
    }
    barrier_sync(bar, nworkers);
    const unsigned nnz = inds[total_tiles];

    // pack (SpCCL's second pass, on the new bits in element order)
    for (size_t tile_index = warp; tile_index < total_tiles; tile_index += num_warps) {
        const unsigned nz_before_tile = inds[tile_index];
        // popc two words, adjacent
        const size_t w = tile_index * 64 + 2 * lane;
        uint64_t bits_0 = w < words ? bv[w] : 0;
        uint64_t bits_1 = w + 1 < words ? bv[w + 1] : 0;
        const unsigned popc_0 = __popcll(bits_0);
        const unsigned popc_1 = __popcll(bits_1);
        const unsigned dword_popc = popc_0 + popc_1;
        unsigned sum = dword_popc;
        // prefix sum the words
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            const unsigned rcv = __shfl_up_sync(0xFFFFFFFF, sum, offset);
            if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dword_popc;
        const size_t nz_count_0 = nz_before_tile + sum;
        const size_t nz_count_1 = nz_count_0 + popc_0;
        for (unsigned idx = 0; idx < popc_0; ++idx) {
            const unsigned e = (unsigned)(w * 64) + (__ffsll(bits_0) - 1);
            bits_0 &= (bits_0 - 1);
            pairs[nz_count_0 + idx] = make_uint2(__float_as_uint(dense[e]), e);
        }
        for (unsigned idx = 0; idx < popc_1; ++idx) {
            const unsigned e = (unsigned)((w + 1) * 64) + (__ffsll(bits_1) - 1);
            bits_1 &= (bits_1 - 1);
            pairs[nz_count_1 + idx] = make_uint2(__float_as_uint(dense[e]), e);
        }
    }
    return nnz;
}

// One RS hop of a growing union slice of n elements: steps 1 and 2 above, from
// the received MPAIR payload `in` (null on the first hop) and dense, into the
// send payload `out`. Returns the number of pairs sent; the caller writes the
// header. With `verify`, the pair count received is checked against the
// popcount of the received mask. All nworkers threads (a multiple of 32) must
// call it; uses `scratch` (shared memory, scratchBytes long) and barrier `bar`.
__device__ __forceinline__ uint32_t ccd_growing_union_hop(
    const float* dense, const char* in, char* out, size_t n, bool verify,
    void* scratch, size_t scratchBytes, int tid, int nworkers, int bar
) {
  const size_t valsOffset = ccd_growing_union_vals_offset(n);
  const size_t nTiles = ceil_div(n, CCD_UNION_TILE);
  // scratch: [mask, valsOffset bytes][tile offsets, nTiles+1][verify count]
  uint32_t* mask = (uint32_t*)scratch;
  unsigned* inds = (unsigned*)((char*)scratch + valsOffset);
  unsigned* counted = inds + nTiles + 1;
  if (tid == 0 && valsOffset + (nTiles + 2) * sizeof(unsigned) > scratchBytes) {
    printf("CCD UNION: growing union scratch too small (%lu elements, %lu bytes)\n", n, scratchBytes);
    __trap();
  }
  if (tid == 0 && ((uintptr_t)dense & 15) != 0) {
    printf("CCD UNION: growing union needs 16-byte aligned slices (recvcount a multiple of 4)\n");
    __trap();
  }
  uint32_t nA = 0;
  if (in) {
    nA = ccd_growing_union_nnz(in, tid);
    if (verify) {
      if (tid == 0) *counted = 0;
      barrier_sync(bar, nworkers);
    }
    unsigned c = ccd_growing_union_stage((uint4*)mask, (const uint4*)in, valsOffset / 16, verify, tid, nworkers);
    if (verify) {
      for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
      if (tid % 32 == 0) atomicAdd(counted, c);
    }
    // step 1
    ccd_union_pairs_add(dense, (const uint4*)(in + valsOffset), (uint4*)(out + valsOffset), nA, tid, nworkers);
    barrier_sync(bar, nworkers);
    if (verify && tid == 0 && *counted != nA) {
      printf("CCD UNION: rank %d ch %d: got %u pairs, mask has %u bits (growing union)\n",
             ncclShmem.comm.rank, ncclShmem.channelId, nA, *counted);
      __trap();
    }
  }
  return ccd_growing_union_compress(dense, n, (uint2*)(out + valsOffset), (uint32_t*)out, mask, in != nullptr,
                                    inds, nA, tid, nworkers, bar);
}

// Final receive of a growing union chain: dense[idx] += value for every pair.
// With `verify`, checks the pair count against the popcount of the received
// mask first (one word of `scratch`, shared memory).
__device__ __forceinline__ void ccd_growing_union_scatter_recv(
    float* dense, const char* in, size_t n, bool verify, void* scratch, int tid, int nworkers, int bar
) {
  const uint32_t nnz = ccd_growing_union_nnz(in, tid);
  if (verify) {
    unsigned* counted = (unsigned*)scratch;
    if (tid == 0) *counted = 0;
    barrier_sync(bar, nworkers);
    const uint64_t* mask = (const uint64_t*)in;
    const size_t words = ccd_growing_union_vals_offset(n) / sizeof(uint64_t);
    unsigned c = 0;
    for (size_t w = tid; w < words; w += nworkers) c += __popcll(ccd_ld_cg(mask + w));
    for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    if (tid % 32 == 0) atomicAdd(counted, c);
    barrier_sync(bar, nworkers);
    if (tid == 0 && *counted != nnz) {
      printf("CCD UNION: rank %d ch %d: got %u pairs, mask has %u bits (growing union)\n",
             ncclShmem.comm.rank, ncclShmem.channelId, nnz, *counted);
      __trap();
    }
  }
  ccd_union_pairs_scatter_add(dense, (const uint4*)(in + ccd_growing_union_vals_offset(n)), nnz, tid, nworkers);
}

/*************************************************************************
 * Growing union v2 (NCCL_CCD_UNION_MASK=growing_union_v2, fp32)
 *
 * The same MPAIR slices, with the same pairs in the same order, as above. The
 * hop's steps run in another order, so that A's pairs read B only where B is
 * nonzero and A's mask is read once, from the received payload:
 *   pass 1   B's mask into shared memory (pass 1 above, without A);
 *   step 1   every pair of A gets B[idx] added where B's bit idx is set, and
 *            goes on as it came where it isn't;
 *   masks    with the count: a lane reads 16 bytes of A's mask and of B's,
 *            writes A | B into the outgoing mask and new = B & ~A over B's,
 *            and counts new per tile;
 *   scan, pack as above.
 * There are no NCCL_CCD_UNION_VERIFY checks.
 ************************************************************************/

// Pass 1 of v2: B's mask of an n-element slice into `mask` (shared memory,
// words32 32-bit words: the mask and its padding), loaded as in pass 1 above.
__device__ __forceinline__ void ccd_growing_union_v2_bits(
    const float* dense, size_t n, uint32_t* mask, size_t words32, int tid, int nworkers
) {
    const unsigned lane      = tid % 32;
    const unsigned warp      = tid / 32;
    const unsigned num_warps = nworkers / 32;
    constexpr int U = CCD_GU_LOADS;
    constexpr size_t Step = 32 * 4;     // elements per warp load
    constexpr size_t Chunk = Step * U;  // elements per warp iteration
    uint64_t* mask64 = (uint64_t*)mask;
    const unsigned sub = lane % 16;
    const size_t nChunks = n / Chunk;
    for (size_t c = warp; c < nChunks; c += num_warps) {
        const size_t base = c * Chunk;
        float4 v[U];
        #pragma unroll
        for (int u = 0; u < U; u++) v[u] = ((const float4*)(dense + base + u * Step))[lane];
        uint64_t mine = 0;
        #pragma unroll
        for (int u = 0; u < U; u++) {
            unsigned long long bits = (unsigned long long)((v[u].x != 0.0f) | ((v[u].y != 0.0f) << 1)
                                    | ((v[u].z != 0.0f) << 2) | ((v[u].w != 0.0f) << 3)) << (4 * sub);
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) bits |= __shfl_xor_sync(0xffffffffu, bits, off);
            if (sub == u) mine = bits;
        }
        if (sub < U) mask64[base / 64 + 2 * sub + lane / 16] = mine;
    }
    // The rest (fewer than Chunk elements, and the mask's padding): 32 elements per thread.
    for (size_t g = nChunks * (Chunk / 32) + tid; g < words32; g += nworkers) {
        mask[g] = ccd_growing_union_bits(dense, n, g);
    }
}

// Step 1 of v2: out = in with dense[idx] added to each pair's value where B's
// bit idx is set in `bmask` (shared memory). Where it isn't, B[idx] is +-0 and
// the pair goes on as it came: v + 0.0f is v, since a pair's value is a sum of
// nonzeros and so never -0.0.
__device__ __forceinline__ void ccd_growing_union_v2_pairs_add(
    const float* dense, const uint32_t* bmask, const uint4* in, uint4* out, uint32_t nnz, int tid, int nworkers
) {
  constexpr int U = CCD_UNION_PAIR_UNROLL;
  const uint32_t nChunks = nnz / 2;
  for (uint32_t c0 = tid; c0 < nChunks; c0 += U * nworkers) {
    uint4 a[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t c = c0 + u * nworkers;
      a[u] = c < nChunks ? __ldcg(in + c) : make_uint4(0, 0, 0, 0);
    }
    // B's bits first, all of them (a chunk past the end has index 0, so this
    // reads bmask[0]); the B loads then depend on no load of their own and
    // compile to predicated loads, with no branch per bit.
    uint32_t bit0[U], bit1[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      bit0[u] = (bmask[a[u].y / 32] >> (a[u].y % 32)) & 1;
      bit1[u] = (bmask[a[u].w / 32] >> (a[u].w % 32)) & 1;
    }
    float b0[U], b1[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t c = c0 + u * nworkers;
      b0[u] = 0.0f;
      b1[u] = 0.0f;
      if (c < nChunks && bit0[u]) {
        b0[u] = dense[a[u].y];
      }
      if (c < nChunks && bit1[u]) {
        b1[u] = dense[a[u].w];
      }
    }
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const uint32_t c = c0 + u * nworkers;
      if (c < nChunks) {
        out[c] = make_uint4(__float_as_uint(__uint_as_float(a[u].x) + b0[u]), a[u].y,
                            __float_as_uint(__uint_as_float(a[u].z) + b1[u]), a[u].w);
      }
    }
  }
  if ((nnz & 1) && tid == 0) {
    const uint2 a = __ldcg((const uint2*)in + (nnz - 1));
    float b = 0.0f;
    if ((bmask[a.y / 32] >> (a.y % 32)) & 1) {
      b = dense[a.y];
    }
    ((uint2*)out)[nnz - 1] = make_uint2(__float_as_uint(__uint_as_float(a.x) + b), a.y);
  }
}

// Masks, count, scan and pack of v2 on an n-element slice. `mask` (shared
// memory, vals_offset bytes) holds B's mask; A's is read from `inMask` (the
// received payload; null on the first hop). The outgoing mask `outMask` gets
// A | B, `mask` new = B & ~A, and `inds` (shared memory) the tile offsets; the
// pairs go after the nA pairs of A already in `pairs`. Returns nA + the pairs
// packed.
__device__ __forceinline__ unsigned ccd_growing_union_v2_compress(
    const float* dense, size_t n, uint2* pairs, uint4* outMask, const uint4* inMask,
    uint32_t* mask, unsigned* inds, unsigned nA, int tid, int nworkers, int bar
) {
    const unsigned lane      = tid % 32;
    const unsigned warp      = tid / 32;
    const unsigned num_warps = nworkers / 32;
    const size_t chunks      = ccd_growing_union_vals_offset(n) / 16;  // 16-byte pieces of the mask, padding included
    const size_t words       = chunks * 2;                             // 64-bit mask words
    const size_t total_tiles = ceil_div(n, CCD_UNION_TILE);

    // masks and count: lane takes 16 bytes, words 2*lane and 2*lane+1, of the
    // tile's 64 words (64 words per tile, so the tiles cover every piece)
    uint4* bm = (uint4*)mask;
    for (size_t tile_index = warp; tile_index < total_tiles; tile_index += num_warps) {
        const size_t q = tile_index * 32 + lane;
        unsigned total_popc = 0;
        if (q < chunks) {
            const uint4 a = inMask ? __ldcg(inMask + q) : make_uint4(0, 0, 0, 0);
            const uint4 b = bm[q];
            const uint4 nw = make_uint4(b.x & ~a.x, b.y & ~a.y, b.z & ~a.z, b.w & ~a.w);
            bm[q] = nw;
            outMask[q] = make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
            total_popc = __popc(nw.x) + __popc(nw.y) + __popc(nw.z) + __popc(nw.w);
        }
        for (unsigned off = 16; off > 0; off >>= 1) {
            total_popc += __shfl_down_sync(0xFFFFFFFF, total_popc, off);
        }
        if (lane == 0) {
            inds[tile_index] = total_popc;
        }
    }
    barrier_sync(bar, nworkers);

    // scan (SpCCL's): the packed pairs go after A's
    if (tid == 0) {
        unsigned running = nA;
        for (unsigned tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
            unsigned count = inds[tile_idx];
            inds[tile_idx] = running;
            running += count;
        }
        inds[total_tiles] = running;
    }
    barrier_sync(bar, nworkers);
    const unsigned nnz = inds[total_tiles];

    // pack (SpCCL's second pass, on the new bits in element order)
    const uint64_t* bv = (const uint64_t*)mask;
    for (size_t tile_index = warp; tile_index < total_tiles; tile_index += num_warps) {
        const unsigned nz_before_tile = inds[tile_index];
        // popc two words, adjacent
        const size_t w = tile_index * 64 + 2 * lane;
        uint64_t bits_0 = w < words ? bv[w] : 0;
        uint64_t bits_1 = w + 1 < words ? bv[w + 1] : 0;
        const unsigned popc_0 = __popcll(bits_0);
        const unsigned popc_1 = __popcll(bits_1);
        const unsigned dword_popc = popc_0 + popc_1;
        unsigned sum = dword_popc;
        // prefix sum the words
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            const unsigned rcv = __shfl_up_sync(0xFFFFFFFF, sum, offset);
            if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dword_popc;
        const size_t nz_count_0 = nz_before_tile + sum;
        const size_t nz_count_1 = nz_count_0 + popc_0;
        for (unsigned idx = 0; idx < popc_0; ++idx) {
            const unsigned e = (unsigned)(w * 64) + (__ffsll(bits_0) - 1);
            bits_0 &= (bits_0 - 1);
            pairs[nz_count_0 + idx] = make_uint2(__float_as_uint(dense[e]), e);
        }
        for (unsigned idx = 0; idx < popc_1; ++idx) {
            const unsigned e = (unsigned)((w + 1) * 64) + (__ffsll(bits_1) - 1);
            bits_1 &= (bits_1 - 1);
            pairs[nz_count_1 + idx] = make_uint2(__float_as_uint(dense[e]), e);
        }
    }
    return nnz;
}

// One RS hop of a growing union v2 slice of n elements (see above), from the
// received MPAIR payload `in` (null on the first hop) and dense, into the send
// payload `out`. Returns the number of pairs sent; the caller writes the
// header. All nworkers threads (a multiple of 32) must call it; uses `scratch`
// (shared memory, scratchBytes long) and barrier `bar`.
__device__ __forceinline__ uint32_t ccd_growing_union_v2_hop(
    const float* dense, const char* in, char* out, size_t n,
    void* scratch, size_t scratchBytes, int tid, int nworkers, int bar
) {
  const size_t valsOffset = ccd_growing_union_vals_offset(n);
  const size_t nTiles = ceil_div(n, CCD_UNION_TILE);
  // scratch: [mask, valsOffset bytes][tile offsets, nTiles+1]
  uint32_t* mask = (uint32_t*)scratch;
  unsigned* inds = (unsigned*)((char*)scratch + valsOffset);
  if (tid == 0 && valsOffset + (nTiles + 1) * sizeof(unsigned) > scratchBytes) {
    printf("CCD UNION: growing union scratch too small (%lu elements, %lu bytes)\n", n, scratchBytes);
    __trap();
  }
  if (tid == 0 && ((uintptr_t)dense & 15) != 0) {
    printf("CCD UNION: growing union needs 16-byte aligned slices (recvcount a multiple of 4)\n");
    __trap();
  }
  // pass 1
  ccd_growing_union_v2_bits(dense, n, mask, valsOffset / sizeof(uint32_t), tid, nworkers);
  barrier_sync(bar, nworkers);
  // step 1
  uint32_t nA = 0;
  if (in) {
    nA = ccd_growing_union_nnz(in, tid);
    ccd_growing_union_v2_pairs_add(dense, mask, (const uint4*)(in + valsOffset), (uint4*)(out + valsOffset), nA,
                                   tid, nworkers);
    barrier_sync(bar, nworkers);
  }
  return ccd_growing_union_v2_compress(dense, n, (uint2*)(out + valsOffset), (uint4*)out, (const uint4*)in, mask,
                                       inds, nA, tid, nworkers, bar);
}

#endif
