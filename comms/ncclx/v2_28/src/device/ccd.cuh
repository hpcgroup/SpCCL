/*************************************************************************
 * CCD Sparse Communication Support
 * Header-based protocol for variable-size sparse transfers in ring
 * collectives. Used by genericOp in prims_simple.h.
 ************************************************************************/

#ifndef NCCL_DEVICE_CCD_CUH
#define NCCL_DEVICE_CCD_CUH

// Header prepended to each sparse chunk in a FIFO slot.
// Layout per slot: [CcdSparseChunkHeader][bv][inds][values]
// Each section padded to 16 bytes for int4 alignment.
// Byte offsets stored in header so receiver can locate each section.

__device__ __forceinline__
size_t ccd_pad16(size_t x) {
    return (x + 15) & ~(size_t) 15;
}

// Align x up to a multiple of 'align' (must be power of 2)
__device__ __host__ __forceinline__
size_t ccd_align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

__host__ __device__ __forceinline__
size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}


struct __align__(16) CcdSparseChunkHeader {
  size_t payload_bytes;  // total bytes after header (bv + padding + inds + padding + values)
  size_t nnz;            // number of nonzero values
  size_t bv_offset;      // byte offset from after header to bitvector (SPOP only)
  size_t inds_offset;    // byte offset from after header to tile prefix sums (SPOP only)
  size_t vals_offset;    // byte offset from after header to packed values (SPOP: after bv+inds, COO1D/DENSE: 0)
  size_t format;         // CcdCompressionProtocol enum value
};

enum CcdCompressionProtocol {
    DENSE,
    COO1D,
    SPOP,
    ADAPTIVE,
    UPACK,  // values packed at union-bitmask positions, mask known to receiver (union-pack RS)
    UPAIR,  // {value, slice index} pairs at union-bitmask positions (union-pack RS, order 2)
    MPAIR   // [mask | {value, slice index} pairs] (growing union RS, ccd_growing_union.cuh)
};

// Asymmetric density extrapolation after one reduce step.
// d_prev = post-reduce density so far, d_base = fresh local data density (d₀).
// d_next = 1 - (1 - d_prev)(1 - d_base)
__device__ __forceinline__
float ccd_expected_density(float d_prev, float d_base) {
    return 1.0f - (1.0f - d_prev) * (1.0f - d_base);
}

template<typename ValType = float, typename IndType = unsigned>
__host__ __device__ __forceinline__
size_t ccd_coo_overhead_bytes(size_t nnz) {
    return ccd_align_up(nnz * sizeof(ValType), sizeof(IndType)) + nnz * sizeof(IndType);
}

template<typename ValType = float, typename IndType = unsigned>
__host__ __device__ __forceinline__
size_t ccd_spop_overhead_bytes(size_t nnz, size_t dense_N) {
    size_t val_bytes = nnz * sizeof(ValType);
    size_t bv_bytes = ceil_div(dense_N, 8);
    size_t ind_bytes = ceil_div(dense_N, 4096) * sizeof(IndType);
    return val_bytes + bv_bytes + ind_bytes;
}

#define CCD_MIN_SEND_BYTES 1024

template<typename ValType = float, typename IndType = unsigned>
__host__ __device__ __forceinline__
CcdCompressionProtocol select_ccd_compression_protocol(
    size_t nnz, size_t dense_N, size_t allow_mask,
    float dense_threshold = 0.3f
) {
    float density = (float) nnz / (float) dense_N;
    if (density >= (1.0f - dense_threshold) && (0b0001 & allow_mask)) {
        return CcdCompressionProtocol::DENSE;
    }
    if (
        ccd_spop_overhead_bytes(nnz, dense_N) < ccd_coo_overhead_bytes(nnz)
        && (0b0100 & allow_mask)
    ) {
        return CcdCompressionProtocol::SPOP;
    } else if (0b0010 & allow_mask) {
        return CcdCompressionProtocol::COO1D;
    }
    if (0b0100 & allow_mask) return CcdCompressionProtocol::SPOP;
    if (0b0010 & allow_mask) return CcdCompressionProtocol::COO1D;
    if (0b0001 & allow_mask) return CcdCompressionProtocol::DENSE;
    return CcdCompressionProtocol::SPOP; // absolute last resort
}

#define CcdMaskDense     0b0001
#define CcdMaskCOO1D     0b0010
#define CcdMaskSPOP      0b0100
#define CcdMaskAll       0b0111

// intended to be used within a single block
template<typename ValType = float, typename IndType = unsigned, bool CheckBounds = false>
__device__
__forceinline__
void ccd_fused_single_block_spop_compress(
    const ValType * __restrict__ dense,
    ValType * __restrict__ compressed,
    const size_t N, // rows
    const size_t M, // cols
    uint64_t * __restrict__ bv,
    IndType * __restrict__ inds,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx,
    const int bar,
    const int nworkers_count,
    CcdCompressionProtocol protocol = CcdCompressionProtocol::SPOP,
    const size_t flat_count = 0  // only used when CheckBounds=true
) {
    // thread/warp indexing
    const unsigned num_warps  = warp_end_idx + 1 - warp_start_idx;
    const unsigned warp_idx   = warp_start_idx + (threadIdx.x / warpSize);
    const unsigned lane       = threadIdx.x % warpSize;
    // these include incomplete tiles
    const size_t tiles_in_N   = ceil_div(N, 64);
    const size_t tiles_in_M   = ceil_div(M, 64);
    const size_t total_tiles  = tiles_in_M * tiles_in_N;
    const ValType comp_zero(0);

    if (warp_idx == warp_start_idx && lane == 0) {
        inds[total_tiles] = 0;
    }

    for (
        size_t tile_index = warp_idx;
        tile_index < total_tiles;
        tile_index += num_warps
    ) {
        const size_t idx_N             = tile_index / tiles_in_M;
        const size_t idx_M             = tile_index % tiles_in_M;
        // number of prior rows * elements in row
        const size_t prior_rowc        = 64 * idx_N ;
        // prior rows * elements in row
        const size_t prior_row_elec    = prior_rowc * M;
        // number of elements in this row before tile start
        const size_t prior_colc        = 64 * idx_M;
        size_t base_idx                = prior_row_elec + prior_colc + lane;
        // determine how many rows and cols to process (normally 64, 64)
        const unsigned cols_after      = M - prior_colc;
        const unsigned cols_to_process = (cols_after < 64) ? cols_after : 64;
        const unsigned rows_after      = N - prior_rowc;
        const unsigned rows_to_process = (rows_after < 64) ? rows_after : 64;
        uint64_t col_0_bv   = 0;
        unsigned nz_count_0 = 0;
        // process first col in tile for this thread
        if (lane < cols_to_process) { // if there are < 32 cols, mask some threads
            for (size_t j = 0; j < rows_to_process; ++j) {
                size_t full_idx = base_idx + (j * (size_t) M);
                ValType val;
                if constexpr (CheckBounds) {
                    val = (full_idx >= flat_count) ? (ValType) 0 : dense[full_idx];
                } else {
                    val = dense[full_idx];
                }
                uint64_t is_nz = (float)val != 0.0f;
                col_0_bv |= (is_nz << j);
                nz_count_0 += (unsigned) is_nz;
            }
        }
        // advance to next col handled by this thread
        base_idx += warpSize;
        uint64_t col_1_bv   = 0;
        unsigned nz_count_1 = 0;
        // process second col in tile for this thread
        // mask if beyond what needs to be processed
        if (lane + warpSize < cols_to_process) {
            for (size_t j = 0; j < rows_to_process; ++j) {
                size_t full_idx = base_idx + (j * (size_t) M);
                ValType val;
                if constexpr (CheckBounds) {
                    val = (full_idx >= flat_count) ? (ValType) 0 : dense[full_idx];
                } else {
                    val = dense[full_idx];
                }
                uint64_t is_nz = (float)val != 0.0f;
                col_1_bv |= (is_nz << j);
                nz_count_1 += (unsigned) is_nz;
            }
        }
        size_t bv_base_idx = tile_index * 64 + lane;
        bv[bv_base_idx] = col_0_bv;
        bv_base_idx += warpSize;
        bv[bv_base_idx] = col_1_bv;
        // find popc of whole tile
        unsigned total_popc = nz_count_0 + nz_count_1;
        for (unsigned off = 16; off > 0; off >>= 1) {
            total_popc += __shfl_down_sync(
                0xFFFFFFFF,
                total_popc,
                off
            );
        }
        // write tile popc to indices
        if (lane == 0) {
            inds[tile_index] = total_popc;
        }
    }
    barrier_sync(bar, nworkers_count);
    if (threadIdx.x == 0) {
        IndType running = 0;
        for (unsigned tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
            IndType count = inds[tile_idx];
            inds[tile_idx] = running;
            running += count;
        }
        inds[total_tiles] = running;
    }
    /*
    if (warp_idx == 0) {
        IndType tile0_count = inds[tile_idx];
        IndType tile1_count = inds[tile_idx + 1];
        IndType dtile_count  = tile0_count + tile1_count;
        IndType sum = dtile_count;
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            const unsigned rcv = __shfl_up_sync(
                0xFFFFFFFF, sum, offset
            );
            if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dtile_count;
        size_t nz_tile_0 = sum;
        size_t nz_tile_1 = sum + tile0_count;
        inds[tile_idx] = nz_tile_0;
        inds[tile_idx + 1] = nz_tile_1;
    }
    */
    barrier_sync(bar, nworkers_count);
    const IndType nnz = inds[total_tiles];
    unsigned *coo1d_keys = (unsigned*) ((char*) compressed + ccd_align_up(nnz * sizeof(ValType), sizeof(unsigned)));
    const unsigned dlane = lane * 2;
    for (
        size_t tile_index = warp_idx;
        tile_index < total_tiles;
        tile_index += num_warps
    ) {
        const size_t idx_N             = tile_index / tiles_in_M;
        const size_t idx_M             = tile_index % tiles_in_M;
        // number of prior rows * elements in row
        const size_t prior_rowc        = 64 * idx_N;
        // prior rows * elements in row
        const size_t prior_row_elec    = prior_rowc * M;
        // number of elements in this row before tile start
        const size_t prior_colc        = 64 * idx_M;
        const size_t base_idx          = prior_row_elec + prior_colc + (size_t) dlane;
        // determine how many rows and cols to process (normally 64, 64)
        const unsigned cols_after      = M - prior_colc;
        const unsigned cols_to_process = (cols_after < 64) ? cols_after : 64;
        const IndType nz_before_tile   = inds[tile_index];
        // popc two cols, adjacent
        const size_t bv_word_idx = tile_index * 64 + dlane;
        uint64_t col_0_bv = bv[bv_word_idx];
        uint64_t col_1_bv = bv[bv_word_idx + 1];
        // bv includes some ghost elements which are still zeroed
        const unsigned col_0_popc = __popcll(col_0_bv);
        const unsigned col_1_popc = __popcll(col_1_bv);
        const unsigned dcol_popc  = col_0_popc + col_1_popc;
        unsigned sum = dcol_popc;
        // prefix sum the cols
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            const unsigned rcv = __shfl_up_sync(
                0xFFFFFFFF, sum, offset
            );
            if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dcol_popc;
        size_t nz_count_0 = sum;
        size_t nz_count_1 = sum + col_0_popc;
        nz_count_0 += nz_before_tile;
        nz_count_1 += nz_before_tile;
        // scan the appropriate words/cols in bv
        if (dlane < cols_to_process) {
            for (size_t idx = 0; idx < col_0_popc; ++idx) {
                const unsigned set_lsb_idx = __ffsll(col_0_bv) - 1;
                col_0_bv &= (col_0_bv - 1);
                const size_t full_dense_index = base_idx + (set_lsb_idx * M);
                const size_t full_compressed_index = nz_count_0 + idx;
                compressed[full_compressed_index] = dense[full_dense_index];
                if (protocol == CcdCompressionProtocol::COO1D) {
                    coo1d_keys[full_compressed_index] = (unsigned) full_dense_index;
                }
            }
        }
        if (dlane + 1 < cols_to_process) {
            for (size_t idx = 0; idx < col_1_popc; ++idx) {
                const unsigned set_lsb_idx = __ffsll(col_1_bv) - 1;
                col_1_bv &= (col_1_bv - 1);
                const size_t full_dense_index = base_idx + 1 + (set_lsb_idx * M);
                const size_t full_compressed_index = nz_count_1 + idx;
                compressed[full_compressed_index] = dense[full_dense_index];
                if (protocol == CcdCompressionProtocol::COO1D) {
                    coo1d_keys[full_compressed_index] = (unsigned) full_dense_index;
                }
            }
        }
    }
}

// needs 1D grid, 1D blocks, and >= 32 threads per block, as
// well as number of threads per block evenly divisible by 32
template<typename ValType = float, typename IndType = unsigned>
__device__
__forceinline__
void ccd_decompress_or_scatter_into(
    ValType * __restrict__ dense,
    const ValType * __restrict__ compressed,
    const size_t N, // rows
    const size_t M, // cols
    const uint64_t * __restrict__ bv,
    const IndType * __restrict__ inds,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx,
    const bool scatter_into = false
) {
    // thread/warp indexing
    /*
    unsigned index      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride     = blockDim.x * gridDim.x;
    unsigned num_warps  = stride / warpSize;
    unsigned warp_idx   = index / warpSize;
    unsigned lane       = index % warpSize;
    unsigned dlane      = lane * 2;
    */
    const unsigned num_warps  = warp_end_idx + 1 - warp_start_idx;
    const unsigned warp_idx   = warp_start_idx + (threadIdx.x / warpSize);
    const unsigned lane       = threadIdx.x % warpSize;
    const unsigned dlane      = lane * 2;
    // these include incomplete tiles
    size_t tiles_in_N = ceil_div(N, 64);
    size_t tiles_in_M = ceil_div(M, 64);
    size_t total_tiles  = tiles_in_M * tiles_in_N;
    ValType comp_zero(0);

    for (
        size_t tile_index = warp_idx;
        tile_index < total_tiles;
        tile_index += num_warps 
    ) {
        size_t idx_N             = tile_index / tiles_in_M;
        size_t idx_M             = tile_index % tiles_in_M;
        // number of prior rows * elements in row
        size_t prior_rowc        = 64 * idx_N;
        // prior rows * elements in row
        size_t prior_row_elec    = prior_rowc * M;
        // number of elements in this row before tile start
        size_t prior_colc        = 64 * idx_M;
        size_t base_idx          = prior_row_elec + prior_colc + (size_t) dlane;
        // determine how many rows and cols to process (normally 64, 64)
        unsigned cols_after      = M - prior_colc;
        unsigned cols_to_process = (cols_after < 64) ? cols_after : 64;
        // unsigned rows_after      = N - prior_rowc;
        // unsigned rows_to_process = (rows_after < 64) ? rows_after : 64;
        IndType nz_before_tile = inds[tile_index];
        // popc two cols, adjacent
        size_t bv_word_idx = tile_index * 64 + dlane;
        uint64_t col_0_bv = bv[bv_word_idx];
        uint64_t col_1_bv = bv[bv_word_idx + 1];
        // bv includes some ghost elements which are still zeroed
        unsigned col_0_popc = __popcll(col_0_bv);
        unsigned col_1_popc = __popcll(col_1_bv);
        unsigned dcol_popc  = col_0_popc + col_1_popc;
        unsigned sum = dcol_popc;
        // prefix sum the cols
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            unsigned rcv = __shfl_up_sync(
                0xFFFFFFFF, sum, offset
            );
           if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dcol_popc;
        size_t nz_count_0 = sum;
        size_t nz_count_1 = sum + col_0_popc;
        nz_count_0 += nz_before_tile;
        nz_count_1 += nz_before_tile;
        // scan the appropriate words/cols in bv
        if (dlane < cols_to_process) {
            for (size_t idx = 0; idx < col_0_popc; ++idx) {
                unsigned set_lsb_idx = __ffsll(col_0_bv) - 1;
                col_0_bv &= (col_0_bv - 1);
                size_t full_dense_index = base_idx + (set_lsb_idx * M);
                size_t full_compressed_index = nz_count_0 + idx;
                if (scatter_into) {
                    dense[full_dense_index] = (ValType)((float) dense[full_dense_index] + (float) compressed[full_compressed_index]);
                } else {
                    dense[full_dense_index] = compressed[full_compressed_index];
                }
            }
        }
        if (dlane + 1 < cols_to_process) {
            for (size_t idx = 0; idx < col_1_popc; ++idx) {
                unsigned set_lsb_idx = __ffsll(col_1_bv) - 1;
                col_1_bv &= (col_1_bv - 1);
                size_t full_dense_index = base_idx + 1 + (set_lsb_idx * M);
                size_t full_compressed_index = nz_count_1 + idx;
                if (scatter_into) {
                    dense[full_dense_index] = (ValType)((float) dense[full_dense_index] + (float) compressed[full_compressed_index]);
                } else {
                    dense[full_dense_index] = compressed[full_compressed_index];
                }
            }
        }
    }
}

// COO1D scatter-add: vals[i] += into dense[keys[i]]
template<typename ValType = float, typename IndType = unsigned>
__device__ __forceinline__
void ccd_coo1d_scatter_into(
    ValType * __restrict__ dense,
    const ValType * __restrict__ vals,
    const IndType * __restrict__ keys,
    const size_t nnz,
    const size_t dense_size,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx
) {
    const unsigned num_warps = warp_end_idx + 1 - warp_start_idx;
    const unsigned warp_idx = warp_start_idx + (threadIdx.x / warpSize);
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned tid_local = (warp_idx - warp_start_idx) * warpSize + lane;
    const unsigned nthreads = num_warps * warpSize;
    for (size_t i = tid_local; i < nnz; i += nthreads) {
        //if (keys[i] >= dense_size) {
        //    printf("COO1D SCATTER OOB: blk=%d tid=%d key[%lu]=%u >= dense_size=%lu nnz=%lu\n",
        //           blockIdx.x, threadIdx.x, i, (unsigned)keys[i], dense_size, nnz);
        //    continue;
        //}
        dense[keys[i]] = (ValType) ((float) dense[keys[i]] + (float) vals[i]);
    }
}

// COO1D decompress (overwrite): dst[key] = val
template<typename ValType, typename IndType>
__device__ __forceinline__
void ccd_coo1d_decompress_into(
    ValType * __restrict__ dense,
    const ValType * __restrict__ vals,
    const IndType * __restrict__ keys,
    const size_t nnz,
    const size_t dense_size,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx
) {
    const unsigned num_warps = warp_end_idx + 1 - warp_start_idx;
    const unsigned warp_idx = warp_start_idx + (threadIdx.x / warpSize);
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned tid_local = (warp_idx - warp_start_idx) * warpSize + lane;
    const unsigned nthreads = num_warps * warpSize;
    for (size_t i = tid_local; i < nnz; i += nthreads) {
        //if (keys[i] >= dense_size) {
        //    printf("COO1D DECOMP OOB: blk=%d tid=%d key[%lu]=%u >= dense_size=%lu nnz=%lu\n",
        //           blockIdx.x, threadIdx.x, i, (unsigned)keys[i], dense_size, nnz);
        //    continue;
        //}
        dense[keys[i]] = vals[i];
    }
}

// Dense scatter-add: dst[i] += src[i]
template<typename T>
__device__ __forceinline__
void ccd_dense_scatter_into(
    T * __restrict__ dst,
    const T * __restrict__ src,
    const size_t n,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx
) {
    const unsigned num_warps = warp_end_idx + 1 - warp_start_idx;
    const unsigned warp_idx = warp_start_idx + (threadIdx.x / warpSize);
    const unsigned lane = threadIdx.x % warpSize;
    const unsigned tid_local = (warp_idx - warp_start_idx) * warpSize + lane;
    const unsigned nthreads = num_warps * warpSize;
    for (size_t i = tid_local; i < n; i += nthreads) {
        dst[i] = (T)((float) dst[i] + (float) src[i]);
    }
}

// ============================================================
// Element-type dispatch wrappers
//
// AllGather kernels run with T=int8_t (NCCL byte-level conversion),
// but sparse compress/decompress must operate on the real value type.
// These wrappers take the caller's T as a template parameter:
//   - sizeof(T) > 1  ->  T is already correct (RS/AR), compile-time direct call
//   - sizeof(T) == 1 ->  AG path, runtime dispatch on ccdEltSize
// ============================================================

#define CCD_DISPATCH_BODY(eltSz, ...) \
  do { \
    if ((eltSz) == 4)      { typedef float   _CcdVT; __VA_ARGS__ } \
    else if ((eltSz) == 2) { typedef __half   _CcdVT; __VA_ARGS__ } \
    else if ((eltSz) == 8) { typedef double   _CcdVT; __VA_ARGS__ } \
    else                    { typedef int8_t   _CcdVT; __VA_ARGS__ } \
  } while(0)

// Compress dispatch (wraps ccd_fused_single_block_spop_compress)
template<typename T, typename IndType, bool CheckBounds>
__device__ __forceinline__
void ccd_compress_dispatch(
    int eltSize, void* dense, void* vals,
    size_t N, size_t M,
    uint64_t* bv, IndType* inds,
    unsigned warp_start, unsigned warp_end,
    int bar, int nworkers,
    CcdCompressionProtocol proto = CcdCompressionProtocol::SPOP,
    size_t workSizeForBounds = 0
) {
  if constexpr (sizeof(T) > 1) {
    if constexpr (CheckBounds) {
      ccd_fused_single_block_spop_compress<T, IndType, true>(
          (T*)dense, (T*)vals, N, M, bv, inds,
          warp_start, warp_end, bar, nworkers, proto, workSizeForBounds);
    } else {
      ccd_fused_single_block_spop_compress<T, IndType, false>(
          (T*)dense, (T*)vals, N, M, bv, inds,
          warp_start, warp_end, bar, nworkers, proto);
    }
  } else {
    CCD_DISPATCH_BODY(eltSize, {
      if constexpr (CheckBounds) {
        ccd_fused_single_block_spop_compress<_CcdVT, IndType, true>(
            (_CcdVT*)dense, (_CcdVT*)vals, N, M, bv, inds,
            warp_start, warp_end, bar, nworkers, proto, workSizeForBounds);
      } else {
        ccd_fused_single_block_spop_compress<_CcdVT, IndType, false>(
            (_CcdVT*)dense, (_CcdVT*)vals, N, M, bv, inds,
            warp_start, warp_end, bar, nworkers, proto);
      }
    });
  }
}

// Decompress-or-scatter dispatch (wraps ccd_decompress_or_scatter_into)
template<typename T>
__device__ __forceinline__
void ccd_spop_dispatch(
    int eltSize, void* dense, const void* compressed,
    size_t N, size_t M,
    const uint64_t* bv, const unsigned* inds,
    unsigned warp_start, unsigned warp_end,
    bool scatter_into
) {
  if constexpr (sizeof(T) > 1) {
    ccd_decompress_or_scatter_into<T, unsigned>(
        (T*)dense, (const T*)compressed, N, M,
        bv, inds, warp_start, warp_end, scatter_into);
  } else {
    CCD_DISPATCH_BODY(eltSize, {
      ccd_decompress_or_scatter_into<_CcdVT, unsigned>(
          (_CcdVT*)dense, (const _CcdVT*)compressed, N, M,
          bv, inds, warp_start, warp_end, scatter_into);
    });
  }
}

// COO1D scatter-add dispatch
template<typename T>
__device__ __forceinline__
void ccd_coo1d_scatter_dispatch(
    int eltSize, void* dense, const void* vals_raw, const void* keys_raw,
    size_t nnz, size_t dense_size,
    unsigned warp_start, unsigned warp_end
) {
  if constexpr (sizeof(T) > 1) {
    ccd_coo1d_scatter_into<T, unsigned>(
        (T*)dense, (const T*)vals_raw, (const unsigned*)keys_raw,
        nnz, dense_size, warp_start, warp_end);
  } else {
    CCD_DISPATCH_BODY(eltSize, {
      ccd_coo1d_scatter_into<_CcdVT, unsigned>(
          (_CcdVT*)dense, (const _CcdVT*)vals_raw, (const unsigned*)keys_raw,
          nnz, dense_size, warp_start, warp_end);
    });
  }
}

// COO1D decompress (overwrite) dispatch
template<typename T>
__device__ __forceinline__
void ccd_coo1d_decompress_dispatch(
    int eltSize, void* dense, const void* vals_raw, const void* keys_raw,
    size_t nnz, size_t dense_size,
    unsigned warp_start, unsigned warp_end
) {
  if constexpr (sizeof(T) > 1) {
    ccd_coo1d_decompress_into<T, unsigned>(
        (T*)dense, (const T*)vals_raw, (const unsigned*)keys_raw,
        nnz, dense_size, warp_start, warp_end);
  } else {
    CCD_DISPATCH_BODY(eltSize, {
      ccd_coo1d_decompress_into<_CcdVT, unsigned>(
          (_CcdVT*)dense, (const _CcdVT*)vals_raw, (const unsigned*)keys_raw,
          nnz, dense_size, warp_start, warp_end);
    });
  }
}

// Format selection dispatch
template<typename T>
__device__ __forceinline__
CcdCompressionProtocol ccd_select_format_dispatch(
    int eltSize, size_t nnz, size_t dense_N, size_t allow_mask, float threshold
) {
  if constexpr (sizeof(T) > 1) {
    return select_ccd_compression_protocol<T, unsigned>(nnz, dense_N, allow_mask, threshold);
  } else {
    if (eltSize == 4)
      return select_ccd_compression_protocol<float, unsigned>(nnz, dense_N, allow_mask, threshold);
    else if (eltSize == 2)
      return select_ccd_compression_protocol<__half, unsigned>(nnz, dense_N, allow_mask, threshold);
    else if (eltSize == 8)
      return select_ccd_compression_protocol<double, unsigned>(nnz, dense_N, allow_mask, threshold);
    else
      return select_ccd_compression_protocol<int8_t, unsigned>(nnz, dense_N, allow_mask, threshold);
  }
}

// ============================================================
// Union-pack (UPACK) ReduceScatter
//
// For the first N GPUs A_0..A_{N-1} a chunk visits on the ring, each GPU
// packs its slice at the positions of the union bitmask
//   U = mask(A_0) | ... | mask(A_{N-1}),
// computed locally from bitmasks allgathered before the collective. Sender
// and receiver both know U, so hops A_0..A_{N-2} carry only packed values (no
// bitvector) and the reduction is an elementwise add of packed arrays.
//
// Two packing orders (NCCL_CCD_UNION_ORDER):
//   Element: increasing element order. Hops are coalesced streams; A_{N-1}
//            scatter-adds the packed sum into its dense buffer and continues
//            on the regular sparse path (compress, format choice, ...).
//   Tile:    SPOP order (64x64 tiles, column-major within a tile). A_{N-1}
//            adds its values in packed form and sends SPOP with U as the
//            bitvector, or DENSE if the dense threshold says so; no compress.
//
// Masks: flat bitmasks over the whole sendbuff; bit (e % 64) of word (e / 64)
// is set if element e may be nonzero (a superset of the true support is fine).
//   AG masks: one per rank, allgathered, [nRanks][maskWords] uint64 words;
//             U is the OR of the chain members' masks.
//   AR mask:  one mask (the OR of all ranks', e.g. from a BitOr allreduce),
//             used as U for every chain.
//
// Tile offsets (NCCL_CCD_UNION_PASS_COUNTS=1): a slice's per-tile offsets in
// U are the same at every hop of its chain, so the first hop counts them and
// sends them in front of the packed values; later hops load them instead of
// counting U again (NCCL_CCD_UNION_VERIFY=1 counts anyway and compares).
// ============================================================

#define CCD_UNION_TILE 4096  // elements per tile: 64 rows of 64
#define CCD_UNION_UNROLL 4   // rows in flight per warp in the pack/scatter pass

enum CcdUnionOrder {
  CcdUnionOrderElement = 0,
  CcdUnionOrderTile = 1,
  CcdUnionOrderPairs = 2  // UPAIR: hop 0 packs {value, index} pairs in tile order (fp32, N = nRanks)
};

enum CcdUnionMode {
  CcdUnionPack,        // out[p] = dense[e]
  CcdUnionPackAdd,     // out[p] = in[p] + dense[e]
  CcdUnionAddInPlace,  // out[p] += dense[e]  (received values already staged in out)
  CcdUnionScatterAdd,  // dense[e] += in[p]
  CcdUnionPackPairs    // pair p of out = {dense[e], e}  (fp32)
};

// Load that bypasses L1: received FIFO data is written by a peer, and L1 may
// still hold lines from an earlier use of the same slot.
template<typename T>
__device__ __forceinline__ T ccd_ld_cg(const T* p) {
  T v;
  if constexpr (sizeof(T) == 8) {
    unsigned long long r = __ldcg((const unsigned long long*)p); memcpy(&v, &r, 8);
  } else if constexpr (sizeof(T) == 4) {
    unsigned int r = __ldcg((const unsigned int*)p); memcpy(&v, &r, 4);
  } else if constexpr (sizeof(T) == 2) {
    unsigned short r = __ldcg((const unsigned short*)p); memcpy(&v, &r, 2);
  } else {
    unsigned char r = __ldcg((const unsigned char*)p); memcpy(&v, &r, 1);
  }
  return v;
}

// OR of the chain's mask bits for sendbuff elements [e, e+valid), returned as
// one word with bit i <-> element e+i. Requires valid <= 64.
__device__ __forceinline__ uint64_t ccd_union_row(
    const uint64_t* const* chain, int nChain, size_t maskWords, size_t e, int valid
) {
  if (valid <= 0) return 0;
  const size_t w = e >> 6;
  const unsigned s = (unsigned)(e & 63);
  const bool hasHi = s != 0 && w + 1 < maskWords;
  uint64_t acc = 0;
  for (int j = 0; j < nChain; j++) {
    const unsigned long long* m = (const unsigned long long*)chain[j];
    uint64_t word = __ldg(m + w);
    if (s != 0) word = (word >> s) | (hasHi ? (uint64_t)__ldg(m + w + 1) << (64 - s) : 0);
    acc |= word;
  }
  return valid < 64 ? acc & ((1ull << valid) - 1) : acc;
}

// One slice's view of U, set up by ccd_union_count. chain and tilePre live in
// shared memory and stay valid until the next slice.
struct CcdUnionSlice {
  const uint64_t* const* chain;  // mask base of each chain member
  uint32_t* tilePre;             // [nTiles+1]: nnz(U) before each tile, then the total
  uint32_t* ownCount;            // this rank's nonzeros (counted in pass 1 or pass 2)
  size_t maskWords;
  size_t eltOffset;              // slice start, in sendbuff elements
  uint32_t n;                    // slice length in elements
  uint32_t nTiles;
  int nChain;
};

// U's rows `lane` and `lane`+32 of tile t (bit c <-> column c of the row).
__device__ __forceinline__ void ccd_union_tile_rows(
    const CcdUnionSlice& s, uint32_t t, int lane, uint64_t& r0, uint64_t& r1
) {
  const uint32_t e0 = t * CCD_UNION_TILE + lane * 64, e1 = e0 + 32 * 64;
  r0 = ccd_union_row(s.chain, s.nChain, s.maskWords, s.eltOffset + e0, e0 < s.n ? (int)min(s.n - e0, 64u) : 0);
  r1 = ccd_union_row(s.chain, s.nChain, s.maskWords, s.eltOffset + e1, e1 < s.n ? (int)min(s.n - e1, 64u) : 0);
}

// Sets up `s` for the slice of n elements at sendbuff element eltOffset, with
// the chain seen at ring step k (member j is userRanks[(j - k) mod nRanks]),
// without counting U. With `arMask`, masks is the AR mask, used as U for
// every chain; otherwise it is the AG masks. All nworkers threads (a multiple
// of 32) of the group must call it; uses `scratch` (shared memory,
// scratchBytes long) and named barrier `bar`.
__device__ __forceinline__ void ccd_union_setup(
    CcdUnionSlice& s, uint32_t n, size_t eltOffset,
    const uint64_t* masks, size_t maskWords,
    const int* userRanks, int nRanks, int k, int nChain, bool arMask,
    void* scratch, size_t scratchBytes,
    int tid, int nworkers, int bar
) {
  const uint64_t** chain = (const uint64_t**)scratch;
  if (arMask) nChain = 1;
  s.chain = chain;
  s.nChain = nChain;
  s.maskWords = maskWords;
  s.eltOffset = eltOffset;
  s.n = n;
  s.nTiles = (uint32_t)ceil_div(n, CCD_UNION_TILE);
  s.tilePre = (uint32_t*)(chain + nChain);
  uint32_t* ownCount = s.ownCount = s.tilePre + s.nTiles + 1;
  if (tid == 0 && nChain * sizeof(void*) + (s.nTiles + 2) * sizeof(uint32_t) > scratchBytes) {
    printf("CCD UNION: scratch too small (%d chain, %u tiles, %lu bytes)\n", nChain, s.nTiles, scratchBytes);
    __trap();
  }
  for (int j = tid; j < nChain; j += nworkers) {
    chain[j] = arMask ? masks : masks + (size_t)userRanks[(j - k + nRanks) % nRanks] * maskWords;
  }
  if (tid == 0) *ownCount = 0;
  barrier_sync(bar, nworkers);
}

// ccd_union_setup, then counts U per tile into s.tilePre (exclusive offsets,
// then the total). Returns nnz(U); if ownNnz is set (AG masks only), also
// returns the popcount of this rank's own (member k's) mask over the slice.
__device__ __forceinline__ uint32_t ccd_union_count(
    CcdUnionSlice& s, uint32_t n, size_t eltOffset,
    const uint64_t* masks, size_t maskWords,
    const int* userRanks, int nRanks, int k, int nChain, bool arMask,
    void* scratch, size_t scratchBytes,
    int tid, int nworkers, int bar,
    uint32_t* ownNnz = nullptr
) {
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int nwarps = nworkers / 32;
  ccd_union_setup(s, n, eltOffset, masks, maskWords, userRanks, nRanks, k, nChain, arMask,
                  scratch, scratchBytes, tid, nworkers, bar);
  if (arMask) ownNnz = nullptr;
  const uint64_t* const* chain = s.chain;
  uint32_t* ownCount = s.ownCount;

  for (uint32_t t = warp; t < s.nTiles; t += nwarps) {
    uint64_t r0, r1;
    ccd_union_tile_rows(s, t, lane, r0, r1);
    unsigned c = __popcll(r0) + __popcll(r1);
    for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    if (lane == 0) s.tilePre[t] = c;
    if (ownNnz) {
      const uint32_t e0 = t * CCD_UNION_TILE + lane * 64, e1 = e0 + 32 * 64;
      unsigned o = __popcll(ccd_union_row(chain + k, 1, maskWords, eltOffset + e0, e0 < n ? (int)min(n - e0, 64u) : 0))
                 + __popcll(ccd_union_row(chain + k, 1, maskWords, eltOffset + e1, e1 < n ? (int)min(n - e1, 64u) : 0));
      for (int off = 16; off > 0; off >>= 1) o += __shfl_down_sync(0xffffffffu, o, off);
      if (lane == 0) atomicAdd(ownCount, o);
    }
  }
  barrier_sync(bar, nworkers);

  // Exclusive scan of the tile counts by warp 0 (each lane owns a run of tiles).
  if (warp == 0) {
    const uint32_t per = (uint32_t)ceil_div(s.nTiles, 32);
    const uint32_t lo = min(lane * per, s.nTiles), hi = min(lo + per, s.nTiles);
    uint32_t sum = 0;
    for (uint32_t t = lo; t < hi; t++) sum += s.tilePre[t];
    uint32_t incl = sum;
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t v = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += v;
    }
    uint32_t run = incl - sum;
    for (uint32_t t = lo; t < hi; t++) { const uint32_t c = s.tilePre[t]; s.tilePre[t] = run; run += c; }
    if (lane == 31) s.tilePre[s.nTiles] = incl;
  }
  barrier_sync(bar, nworkers);
  if (ownNnz) *ownNnz = *ownCount;
  return s.tilePre[s.nTiles];
}

// Instead of counting, after ccd_union_setup: copies the tile offsets that the
// previous hop sent with the values (nTiles+1 words, as ccd_union_count
// leaves them in s.tilePre) into s.tilePre. Returns nnz(U).
__device__ __forceinline__ uint32_t ccd_union_load_counts(
    CcdUnionSlice& s, const uint32_t* counts, int tid, int nworkers, int bar
) {
  for (uint32_t i = tid; i <= s.nTiles; i += nworkers) s.tilePre[i] = __ldcg(counts + i);
  barrier_sync(bar, nworkers);
  return s.tilePre[s.nTiles];
}

// Byte offset of the values in a UPACK payload of an n-element slice. With
// passCounts the payload starts with U's tile offsets (nTiles+1 words, padded
// to 16 bytes); otherwise the values come first.
__device__ __forceinline__ size_t ccd_union_vals_offset(uint32_t n, bool passCounts) {
  return passCounts ? ccd_pad16((ceil_div(n, CCD_UNION_TILE) + 1) * sizeof(uint32_t)) : 0;
}

// One element of a pass-2 batch (packed position p, slice element e), split
// into load and store so all of a batch's loads issue before its stores.
template<typename T, CcdUnionMode Mode>
struct CcdUnionElt {
  T d, i;
  __device__ __forceinline__ void load(const T* dense, const T* in, const T* out, uint32_t p, uint32_t e) {
    d = dense[e];
    if constexpr (Mode == CcdUnionPackAdd || Mode == CcdUnionScatterAdd) i = ccd_ld_cg(in + p);
    if constexpr (Mode == CcdUnionAddInPlace) i = out[p];
  }
  __device__ __forceinline__ void store(T* dense, T* out, uint32_t p, uint32_t e) const {
    if constexpr (Mode == CcdUnionPack) out[p] = d;
    else if constexpr (Mode == CcdUnionScatterAdd) dense[e] = (T)((float)d + (float)i);
    else if constexpr (Mode == CcdUnionPackPairs) ((uint2*)out)[p] = make_uint2(__float_as_uint((float)d), e);
    else out[p] = (T)((float)i + (float)d);
  }
};

// Pass 2, element order: walk each tile's non-empty rows. The 32 lanes cover a
// row's 64 elements two apiece, so dense accesses are coalesced and packed
// positions are contiguous.
template<typename T, CcdUnionMode Mode>
__device__ __forceinline__ void ccd_union_pass2_element(
    const CcdUnionSlice& s, T* __restrict__ dense, const T* __restrict__ in, T* __restrict__ out,
    int tid, int nworkers, bool countNz
) {
  const int lane = tid % 32;
  const unsigned bit = 2 * lane;
  const uint64_t below = (1ull << bit) - 1;
  unsigned nz = 0;
  for (uint32_t t = tid / 32; t < s.nTiles; t += nworkers / 32) {
    uint64_t r0, r1;
    ccd_union_tile_rows(s, t, lane, r0, r1);
    uint64_t rows = (uint64_t)__ballot_sync(0xffffffffu, r0 != 0)
                  | (uint64_t)__ballot_sync(0xffffffffu, r1 != 0) << 32;
    const uint32_t tileElt = t * CCD_UNION_TILE;
    uint32_t base = s.tilePre[t];
    while (rows) {
      CcdUnionElt<T, Mode> x0[CCD_UNION_UNROLL], x1[CCD_UNION_UNROLL];
      bool m0[CCD_UNION_UNROLL], m1[CCD_UNION_UNROLL];
      uint32_t p[CCD_UNION_UNROLL], e[CCD_UNION_UNROLL];
      #pragma unroll
      for (int u = 0; u < CCD_UNION_UNROLL; u++) {
        const int j = rows ? __ffsll((long long)rows) - 1 : -1;  // warp-uniform
        rows &= rows - 1;
        const uint64_t R = j >= 0
            ? (uint64_t)__shfl_sync(0xffffffffu, (unsigned long long)(j < 32 ? r0 : r1), j & 31) : 0;
        m0[u] = (R >> bit) & 1;
        m1[u] = (R >> (bit + 1)) & 1;
        p[u] = base + __popcll(R & below);
        e[u] = tileElt + (j >= 0 ? j : 0) * 64 + bit;
        base += __popcll(R);
        if (m0[u]) x0[u].load(dense, in, out, p[u], e[u]);
        if (m1[u]) x1[u].load(dense, in, out, p[u] + m0[u], e[u] + 1);
        if (countNz) nz += (m0[u] && (float)x0[u].d != 0.0f) + (m1[u] && (float)x1[u].d != 0.0f);
      }
      #pragma unroll
      for (int u = 0; u < CCD_UNION_UNROLL; u++) {
        if (m0[u]) x0[u].store(dense, out, p[u], e[u]);
        if (m1[u]) x1[u].store(dense, out, p[u] + m0[u], e[u] + 1);
      }
    }
  }
  if (countNz) {
    for (int off = 16; off > 0; off >>= 1) nz += __shfl_down_sync(0xffffffffu, nz, off);
    if (lane == 0) atomicAdd(s.ownCount, nz);
  }
}

// Pass 2, tile (SPOP) order. Each warp transposes its tile's 64x64 bits so
// lane l holds columns 2l and 2l+1 (written to bv in SPOP layout if bv is
// set), then walks the non-empty rows like the element order; packed
// positions advance per column.
template<typename T, CcdUnionMode Mode>
__device__ __forceinline__ void ccd_union_pass2_tile(
    const CcdUnionSlice& s, T* __restrict__ dense, const T* __restrict__ in, T* __restrict__ out,
    uint64_t* __restrict__ bv, int tid, int nworkers, bool countNz
) {
  const int lane = tid % 32;
  const unsigned bit = 2 * lane;
  unsigned nz = 0;
  for (uint32_t t = tid / 32; t < s.nTiles; t += nworkers / 32) {
    uint64_t r0, r1;
    ccd_union_tile_rows(s, t, lane, r0, r1);
    uint64_t c0 = 0, c1 = 0;
    #pragma unroll
    for (unsigned c = 0; c < 64; c++) {
      const uint64_t col = (uint64_t)__ballot_sync(0xffffffffu, (r0 >> c) & 1)
                         | (uint64_t)__ballot_sync(0xffffffffu, (r1 >> c) & 1) << 32;
      if (c == bit) c0 = col;
      if (c == bit + 1) c1 = col;
    }
    if (bv) {
      bv[(size_t)t * 64 + bit] = c0;
      bv[(size_t)t * 64 + bit + 1] = c1;
    }
    // Columns are packed in order: first position of columns 2l and 2l+1.
    const uint32_t n0 = __popcll(c0), n1 = __popcll(c1);
    uint32_t incl = n0 + n1;
    for (int off = 1; off < 32; off <<= 1) {
      const uint32_t v = __shfl_up_sync(0xffffffffu, incl, off);
      if (lane >= off) incl += v;
    }
    uint32_t p0 = s.tilePre[t] + incl - n0 - n1;
    uint32_t p1 = p0 + n0;
    uint64_t rows = (uint64_t)__ballot_sync(0xffffffffu, r0 != 0)
                  | (uint64_t)__ballot_sync(0xffffffffu, r1 != 0) << 32;
    const uint32_t tileElt = t * CCD_UNION_TILE;
    while (rows) {
      CcdUnionElt<T, Mode> x0[CCD_UNION_UNROLL], x1[CCD_UNION_UNROLL];
      bool m0[CCD_UNION_UNROLL], m1[CCD_UNION_UNROLL];
      uint32_t q0[CCD_UNION_UNROLL], q1[CCD_UNION_UNROLL], e[CCD_UNION_UNROLL];
      #pragma unroll
      for (int u = 0; u < CCD_UNION_UNROLL; u++) {
        const int j = rows ? __ffsll((long long)rows) - 1 : -1;  // warp-uniform
        rows &= rows - 1;
        m0[u] = j >= 0 && ((c0 >> j) & 1);
        m1[u] = j >= 0 && ((c1 >> j) & 1);
        q0[u] = p0; p0 += m0[u];
        q1[u] = p1; p1 += m1[u];
        e[u] = tileElt + (j >= 0 ? j : 0) * 64 + bit;
        if (m0[u]) x0[u].load(dense, in, out, q0[u], e[u]);
        if (m1[u]) x1[u].load(dense, in, out, q1[u], e[u] + 1);
        if (countNz) nz += (m0[u] && (float)x0[u].d != 0.0f) + (m1[u] && (float)x1[u].d != 0.0f);
      }
      #pragma unroll
      for (int u = 0; u < CCD_UNION_UNROLL; u++) {
        if (m0[u]) x0[u].store(dense, out, q0[u], e[u]);
        if (m1[u]) x1[u].store(dense, out, q1[u], e[u] + 1);
      }
    }
  }
  if (countNz) {
    for (int off = 16; off > 0; off >>= 1) nz += __shfl_down_sync(0xffffffffu, nz, off);
    if (lane == 0) atomicAdd(s.ownCount, nz);
  }
}

// ---- UPAIR hops (NCCL_CCD_UNION_ORDER=2) ----
// A UPAIR payload is nnz {value (fp32 bits), slice index} pairs, two per
// 16-byte chunk. Every rank uses the same U, so the indices stay fixed after
// the first hop: a later hop only adds its own value at each pair's index.

#define CCD_UNION_PAIR_UNROLL 4  // 16-byte chunks in flight per thread

// out = in with dense[idx] added to each pair's value (out may be another GPU's buffer).
__device__ __forceinline__ void ccd_union_pairs_add(
    const float* dense, const uint4* in, uint4* out, uint32_t nnz, int tid, int nworkers
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
    float b0[U], b1[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const bool ok = c0 + u * nworkers < nChunks;
      b0[u] = ok ? dense[a[u].y] : 0.0f;
      b1[u] = ok ? dense[a[u].w] : 0.0f;
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
    ((uint2*)out)[nnz - 1] = make_uint2(__float_as_uint(__uint_as_float(a.x) + dense[a.y]), a.y);
  }
}

// dense[idx] += value for every pair (indices are distinct within a slice).
__device__ __forceinline__ void ccd_union_pairs_scatter_add(
    float* dense, const uint4* in, uint32_t nnz, int tid, int nworkers
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
    float b0[U], b1[U];
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const bool ok = c0 + u * nworkers < nChunks;
      b0[u] = ok ? dense[a[u].y] : 0.0f;
      b1[u] = ok ? dense[a[u].w] : 0.0f;
    }
    #pragma unroll
    for (int u = 0; u < U; u++) {
      if (c0 + u * nworkers < nChunks) {
        dense[a[u].y] = b0[u] + __uint_as_float(a[u].x);
        dense[a[u].w] = b1[u] + __uint_as_float(a[u].z);
      }
    }
  }
  if ((nnz & 1) && tid == 0) {
    const uint2 a = __ldcg((const uint2*)in + (nnz - 1));
    dense[a.y] = dense[a.y] + __uint_as_float(a.x);
  }
}

#endif /* NCCL_DEVICE_CCD_CUH */
