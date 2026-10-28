// Correctness and timing test for ncclReduceScatterSparseUnion.
//
// Every rank fills a sendbuff of nranks*count elements with small integers at
// a sparse set of positions, so all summation orders give bit-identical
// results in f32, f16 and bf16. Each variant is checked against dense
// ncclReduceScatter:
//   dense    ncclReduceScatter
//   spccl    ncclReduceScatterSparse
//   mask     extract bitmask, then the mask exchanges (--exchange):
//              bitmask    ncclAllGather of the bitmasks: the AG masks
//              coo-ag     nonzero indices (uint32 keys): encode, ncclAllGather
//                         padded to the largest rank, decode to bitmasks
//              coo-bcast  same, but one exact-size ncclBroadcast per rank
//              orreduce   ncclAllReduce(ncclBitOr) of the bitmasks: the AR
//                         mask, one mask for every rank
//   union    ncclReduceScatterSparseUnion for each N, packing order, mask
//            kind (--union-masks ag: the AG masks, ORed per chain; ar: the
//            AR mask, used for every chain) and --pass-counts value (1: the
//            first hop sends U's tile offsets with the values and later hops
//            use them instead of counting; CSV column pass_counts). Order 2
//            sends {value, index} pairs from the first hop on (fp32, N >= P);
//            other combinations with it are skipped.
//   compressed-rs  Compressed RS, one pipeline: extract bitmask, bitmask AR
//            (ncclBitOr allreduce), count_elems (nonzeros of the AR mask per
//            shard, read back to size the RS), compress each shard at the AR
//            mask padded to the largest shard, dense ncclReduceScatter of the
//            compressed array, decompress
//   union-e2e  extract bitmask, bitmask AR, then ncclReduceScatterSparseUnion
//            with the AR mask, as one pipeline, for each N and packing order
// The two pipelines are timed with an event after each phase; each phase is
// reported for the rank with the largest total, so the phases add up to it.
// The test's own kernels (extract bitmask, count_elems, compress, decompress)
// launch --grid blocks of --block threads (default: NCCL_CCD_CHANNELS blocks
// of 512 threads, the NCCL channel count and SIMPLE thread count).
//
// Data (--data):
//   iid      element e of rank r is nonzero if a draw shared by all ranks is
//            below density*overlap, or a per-rank draw is below
//            density*(1-overlap). overlap=0 gives independent supports (most
//            fill-in), overlap=1 identical supports.
//   capture  nonzero positions from a DDP mask capture converted by
//            convert_captures.py; test rank r uses capture rank r (or r mod
//            capture ranks with --wrap). --offset/--count select a window.
//
// Usage: srun -n <P> ./rs_union_test [options]   (see usage() below)

#include <cub/device/device_select.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#define CUDACHECK(cmd) do { cudaError_t cudaRes_ = (cmd); if (cudaRes_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaRes_)); \
  MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define NCCLCHECK(cmd) do { ncclResult_t ncclRes_ = (cmd); if (ncclRes_ != ncclSuccess) { \
  fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(ncclRes_)); \
  MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static int gRank = 0;

static void die(const char* fmt, const char* arg = "") {
  if (gRank == 0) { fprintf(stderr, fmt, arg); fprintf(stderr, "\n"); }
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void usage() {
  if (gRank != 0) return;
  printf(
    "Options:\n"
    "  --data iid|capture        data source [iid]\n"
    "  --density LIST            iid: per-rank densities to sweep [0.1]\n"
    "  --overlap O               iid: support overlap [0]\n"
    "  --seed S                  iid positions and values [1]\n"
    "  --capture DIR             capture: directory with manifest.txt + rank_*.bits\n"
    "  --wrap                    capture: allow more ranks than captured (rank r uses r mod R)\n"
    "  --offset E                capture: window start in elements [0]\n"
    "  --size LIST               sendbuff MiB per rank to sweep (alternative to --count)\n"
    "  --count C                 recvcount per rank [iid: 1M, capture: rest of the bucket / P]\n"
    "  --dtype f32|f16|bf16      element type [f32]\n"
    "  --variants LIST           any of dense,spccl,mask,union,compressed-rs,union-e2e [dense,spccl,mask,union]\n"
    "  --grid G --block B        launch shape of the test's own kernels [NCCL_CCD_CHANNELS or 64, 512]\n"
    "  --exchange LIST           mask exchanges to time: bitmask,coo-ag,coo-bcast,orreduce [all]\n"
    "  --union-masks LIST        masks for the union runs: ag (AG masks), ar (AR mask) [ag,ar]\n"
    "  --count-exchange mpi|nccl how COO exchanges share the per-rank counts [mpi]\n"
    "  --n LIST                  union chain lengths [1..P]\n"
    "  --order LIST              union packing: 0=element 1=tile 2=pairs (fp32, N>=P only) [0,1]\n"
    "  --pass-counts LIST        union: 1 = send U's tile offsets with the values (NCCL_CCD_UNION_PASS_COUNTS) [0]\n"
    "  --iters I --warmup W      timed / untimed iterations per variant [20, 5]\n"
    "  --no-check                skip the comparison against dense\n"
    "  --csv                     CSV output\n");
}

struct Options {
  std::string data = "iid";
  std::vector<double> densities = {0.1};
  double overlap = 0.0;
  uint64_t seed = 1;
  std::string capture;
  bool wrap = false;
  size_t offset = 0;
  size_t count = 0;  // 0: default for the data source
  std::vector<double> sizesMiB;  // sendbuff per rank; overrides count
  std::string dtype = "f32";
  std::vector<std::string> variants = {"dense", "spccl", "mask", "union"};
  std::vector<std::string> exchanges = {"bitmask", "coo-ag", "coo-bcast", "orreduce"};
  std::vector<std::string> unionMasks = {"ag", "ar"};
  // nccl hangs on 16 ranks / 4 nodes: the second 8-byte ncclAllGather after a
  // larger one never completes (2026-10-26), so counts go over MPI by default.
  std::string countExchange = "mpi";
  std::vector<int> ns;  // empty: 1..nranks
  std::vector<int> orders = {0, 1};
  std::vector<int> passCounts = {0};
  int grid = 0;  // 0: NCCL_CCD_CHANNELS, else 64
  int block = 512;
  int iters = 20;
  int warmup = 5;
  bool check = true;
  bool csv = false;
};

static std::vector<std::string> splitList(const char* s) {
  std::vector<std::string> v;
  std::string cur;
  for (const char* p = s;; p++) {
    if (*p == ',' || *p == '\0') { if (!cur.empty()) v.push_back(cur); cur.clear(); if (!*p) break; }
    else cur += *p;
  }
  return v;
}

static std::vector<int> intList(const char* s) {
  std::vector<int> v;
  for (auto& x : splitList(s)) v.push_back(atoi(x.c_str()));
  return v;
}

static std::vector<double> doubleList(const char* s) {
  std::vector<double> v;
  for (auto& x : splitList(s)) v.push_back(atof(x.c_str()));
  return v;
}

static Options parseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) die("missing value for %s", a.c_str());
      return argv[++i];
    };
    if (a == "--data") o.data = next();
    else if (a == "--density") o.densities = doubleList(next());
    else if (a == "--overlap") o.overlap = atof(next());
    else if (a == "--seed") o.seed = strtoull(next(), nullptr, 0);
    else if (a == "--capture") o.capture = next();
    else if (a == "--wrap") o.wrap = true;
    else if (a == "--offset") o.offset = strtoull(next(), nullptr, 0);
    else if (a == "--count") o.count = strtoull(next(), nullptr, 0);
    else if (a == "--size") o.sizesMiB = doubleList(next());
    else if (a == "--dtype") o.dtype = next();
    else if (a == "--variants") o.variants = splitList(next());
    else if (a == "--exchange") o.exchanges = splitList(next());
    else if (a == "--union-masks") o.unionMasks = splitList(next());
    else if (a == "--count-exchange") o.countExchange = next();
    else if (a == "--n") o.ns = intList(next());
    else if (a == "--order") o.orders = intList(next());
    else if (a == "--pass-counts") o.passCounts = intList(next());
    else if (a == "--grid") o.grid = atoi(next());
    else if (a == "--block") o.block = atoi(next());
    else if (a == "--iters") o.iters = atoi(next());
    else if (a == "--warmup") o.warmup = atoi(next());
    else if (a == "--no-check") o.check = false;
    else if (a == "--csv") o.csv = true;
    else if (a == "--help" || a == "-h") { usage(); MPI_Finalize(); exit(0); }
    else { usage(); die("unknown option %s", a.c_str()); }
  }
  if (o.data != "iid" && o.data != "capture") die("--data must be iid or capture");
  if (o.data == "capture" && o.capture.empty()) die("--data capture needs --capture DIR");
  if (o.count && !o.sizesMiB.empty()) die("--count and --size are mutually exclusive");
  if (o.densities.empty()) die("--density needs at least one value");
  if (o.countExchange != "nccl" && o.countExchange != "mpi") die("--count-exchange must be nccl or mpi");
  if (o.grid == 0) {
    const char* c = getenv("NCCL_CCD_CHANNELS");
    o.grid = c && atoi(c) > 0 ? atoi(c) : 64;
  }
  if (o.grid <= 0) die("--grid must be positive");
  if (o.block < 32 || o.block > 1024 || o.block % 32) die("--block must be a multiple of 32 in [32, 1024]");
  return o;
}

static bool has(const std::vector<std::string>& v, const char* s) {
  return std::find(v.begin(), v.end(), s) != v.end();
}

// ---------------------------------------------------------------------------
// Capture input (written by convert_captures.py)
// ---------------------------------------------------------------------------

struct Capture {
  int nranks = 0;
  size_t numel = 0;
  long iteration = -1;
};

static Capture readManifest(const std::string& dir) {
  Capture c;
  FILE* f = fopen((dir + "/manifest.txt").c_str(), "r");
  if (!f) die("cannot open %s/manifest.txt", dir.c_str());
  char key[256], val[4096];
  while (fscanf(f, "%255s %4095s", key, val) == 2) {
    if (!strcmp(key, "nranks")) c.nranks = atoi(val);
    else if (!strcmp(key, "numel")) c.numel = strtoull(val, nullptr, 10);
    else if (!strcmp(key, "iteration")) c.iteration = atol(val);
  }
  fclose(f);
  if (c.nranks <= 0 || c.numel == 0) die("bad manifest in %s", dir.c_str());
  return c;
}

// Reads bits [first, first+nbits) of a rank's bitmask file. Returns the bytes
// covering them; bit `first` is bit (first % 8) of byte 0.
static std::vector<uint8_t> readBits(const std::string& path, size_t first, size_t nbits) {
  const size_t byte0 = first / 8, nbytes = (first % 8 + nbits + 7) / 8;
  std::vector<uint8_t> buf(nbytes);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) die("cannot open %s", path.c_str());
  if (fseeko(f, (off_t)byte0, SEEK_SET) != 0 || fread(buf.data(), 1, nbytes, f) != nbytes) {
    die("short read from %s", path.c_str());
  }
  fclose(f);
  return buf;
}

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

template<typename T> struct DType;
template<> struct DType<float> { static constexpr ncclDataType_t nccl = ncclFloat; };
template<> struct DType<__half> { static constexpr ncclDataType_t nccl = ncclHalf; };
template<> struct DType<__nv_bfloat16> { static constexpr ncclDataType_t nccl = ncclBfloat16; };

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}

__device__ __forceinline__ double unit(uint64_t h) { return (double)(h >> 11) * (1.0 / 9007199254740992.0); }

// Small positive integer, so sums of up to 64 ranks are exact in bf16.
__device__ __forceinline__ float value(uint64_t seed, int rank, size_t e) {
  return (float)(1 + (mix64(seed ^ mix64(((uint64_t)rank + 1) * 0xbf58476d1ce4e5b9ull) ^ e) >> 60) % 4);
}

template<typename T>
__global__ void fillIid(T* x, size_t n, int rank, double density, double overlap, uint64_t seed) {
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const uint64_t shared = mix64(seed * 0x9e3779b97f4a7c15ull + e);
    const uint64_t own = mix64(shared ^ mix64(((uint64_t)rank + 1) * 0x94d049bb133111ebull));
    const bool nz = unit(shared) < density * overlap || unit(own) < density * (1.0 - overlap);
    x[e] = (T)(nz ? value(seed, rank, e) : 0.0f);
  }
}

// Element e is nonzero iff bit (shift + e) of `bits` is set.
template<typename T>
__global__ void fillFromBits(T* x, size_t n, const uint8_t* bits, unsigned shift, int rank, uint64_t seed) {
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const size_t b = shift + e;
    x[e] = (T)(((bits[b >> 3] >> (b & 7)) & 1) ? value(seed, rank, e) : 0.0f);
  }
}

// Flat nonzero bitmask: bit e%64 of word e/64; mask must hold ceil(n/64)
// words and x must be 16-byte aligned. Each lane loads 16 bytes (V elements)
// per step with U steps in flight, so a few blocks still keep enough loads
// outstanding; the 64/V lanes covering one word OR their bits together.
template<typename T>
__global__ void extractBitmask(const T* x, size_t n, uint64_t* mask) {
  constexpr int V = 16 / sizeof(T);  // elements per 16-byte load
  constexpr int L = 64 / V;          // lanes per mask word
  constexpr int U = 8;               // loads in flight per lane
  constexpr size_t Step = 32 * V;    // elements per warp load
  const int lane = threadIdx.x & 31;
  const size_t warp = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32;
  const size_t nWarps = (size_t)gridDim.x * blockDim.x / 32;
  const size_t nFull = n / (Step * U);
  for (size_t s = warp; s < nFull; s += nWarps) {
    const size_t base = s * Step * U;
    uint4 v[U];
    #pragma unroll
    for (int u = 0; u < U; u++) v[u] = __ldcs((const uint4*)(x + base + u * Step) + lane);
    #pragma unroll
    for (int u = 0; u < U; u++) {
      const T* e = (const T*)&v[u];
      unsigned long long bits = 0;
      #pragma unroll
      for (int i = 0; i < V; i++) bits |= (unsigned long long)((float)e[i] != 0.0f) << i;
      bits <<= V * (lane % L);
      #pragma unroll
      for (int off = 1; off < L; off <<= 1) bits |= __shfl_xor_sync(0xffffffffu, bits, off);
      if (lane % L == 0) mask[(base + u * Step) / 64 + lane / L] = bits;
    }
  }
  // The rest (fewer than Step*U elements, starting on a word boundary).
  const size_t nWords = (n + 63) / 64;
  for (size_t w = nFull * Step * U / 64 + blockIdx.x * (size_t)blockDim.x + threadIdx.x; w < nWords;
       w += (size_t)gridDim.x * blockDim.x) {
    uint64_t bits = 0;
    for (size_t e = w * 64; e < n && e < w * 64 + 64; e++) bits |= (uint64_t)((float)x[e] != 0.0f) << (e - w * 64);
    mask[w] = bits;
  }
}

template<typename T>
__global__ void countNonzero(const T* x, size_t n, unsigned long long* out) {
  unsigned long long c = 0;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    c += (float)x[e] != 0.0f;
  }
  atomicAdd(out, c);
}

// Bitwise comparison: T is compared through an unsigned integer of its size.
template<typename T>
__global__ void countMismatch(const T* a, const T* b, size_t n, unsigned long long* out) {
  using Bits = typename std::conditional<sizeof(T) == 8, unsigned long long,
               typename std::conditional<sizeof(T) == 4, uint32_t, uint16_t>::type>::type;
  static_assert(sizeof(Bits) == sizeof(T), "unsupported element size");
  const Bits* x = reinterpret_cast<const Bits*>(a);
  const Bits* y = reinterpret_cast<const Bits*>(b);
  unsigned long long c = 0;
  for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    c += x[e] != y[e];
  }
  atomicAdd(out, c);
}


// CUB select predicate: index i is kept if x[i] is nonzero.
template<typename T>
struct NonzeroAt {
  const T* x;
  __device__ __forceinline__ bool operator()(uint32_t i) const { return (float)x[i] != 0.0f; }
};

// COO keys to bitmasks: sets bit k of mask r for each of rank r's keys, which
// start at keys[offsets[r]] (counts[r] of them). Masks must be zeroed first.
__global__ void decodeCoo(const uint32_t* keys, const unsigned long long* offsets,
                          const unsigned long long* counts, uint64_t* masks, size_t maskWords) {
  const int r = blockIdx.y;
  const uint32_t* k = keys + offsets[r];
  unsigned long long* m = (unsigned long long*)(masks + (size_t)r * maskWords);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < counts[r]; i += (size_t)gridDim.x * blockDim.x) {
    const uint32_t e = k[i];
    atomicOr(&m[e >> 6], 1ull << (e & 63));
  }
}

// Reference for the BitOr allreduce: out = OR of the nranks gathered masks.
__global__ void orMasks(const uint64_t* masks, int nranks, size_t maskWords, uint64_t* out) {
  for (size_t w = blockIdx.x * (size_t)blockDim.x + threadIdx.x; w < maskWords; w += (size_t)gridDim.x * blockDim.x) {
    uint64_t acc = 0;
    for (int r = 0; r < nranks; r++) acc |= masks[(size_t)r * maskWords + w];
    out[w] = acc;
  }
}

// ---- Compressed RS ----
// With the AR mask G on every rank: count_elems counts G per shard (shard s =
// sendbuff elements [s*count, (s+1)*count), the part rank s receives);
// compress writes each shard's values at G's positions to out[s*R, s*R +
// nnz_s), zero-padded to R = max over shards of nnz_s; a dense
// ncclReduceScatter of the compressed array (recvcount R); decompress expands
// the received values at G's positions in this rank's shard. Shards are walked
// in tiles of 4096 elements (64 rows of 64), lane l of a warp holding rows l
// and l+32.

#define CRS_TILE 4096
#define CRS_UNROLL 4

// Bits [e, e+valid) of mask as one word (bit i <-> element e+i); valid <= 64.
__device__ __forceinline__ uint64_t maskRow(const uint64_t* mask, size_t nWords, size_t e, long valid) {
  if (valid <= 0) return 0;
  const unsigned long long* m = (const unsigned long long*)mask;
  const size_t w = e >> 6;
  const unsigned s = (unsigned)(e & 63);
  uint64_t r = __ldg(m + w);
  if (s != 0) r = (r >> s) | (w + 1 < nWords ? (uint64_t)__ldg(m + w + 1) << (64 - s) : 0);
  return valid < 64 ? r & ((1ull << valid) - 1) : r;
}

// Rows `lane` and `lane`+32 of tile t of the segment [start, start+count).
__device__ __forceinline__ void tileRows(const uint64_t* mask, size_t nWords, size_t start, size_t count,
                                         size_t t, int lane, uint64_t& r0, uint64_t& r1) {
  const size_t e0 = t * CRS_TILE + (size_t)lane * 64, e1 = e0 + 32 * 64;
  r0 = maskRow(mask, nWords, start + e0, e0 < count ? (long)(count - e0) : 0);
  r1 = maskRow(mask, nWords, start + e1, e1 < count ? (long)(count - e1) : 0);
}

// Sum of v over the block, valid in thread 0. scratch: 32 words of shared memory.
__device__ __forceinline__ unsigned blockSum(unsigned v, unsigned* scratch) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = 0;
  if (warp == 0) {
    v = lane < (int)(blockDim.x / 32) ? scratch[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

// Places a round of tiles (warp w has tile t0 + w): every warp stores its
// tile's count, then gets its tile's first compressed position (relative to the
// round) and the round's total.
__device__ __forceinline__ void roundScan(uint64_t r0, uint64_t r1, unsigned* tileCount,
                                          unsigned& tileOff, unsigned& roundTotal) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  unsigned c = __popcll(r0) + __popcll(r1);
  for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
  if (lane == 0) tileCount[warp] = c;
  __syncthreads();
  const unsigned x = lane < nwarps ? tileCount[lane] : 0;
  unsigned before = lane < warp ? x : 0, all = x;
  for (int off = 16; off > 0; off >>= 1) {
    before += __shfl_down_sync(0xffffffffu, before, off);
    all += __shfl_down_sync(0xffffffffu, all, off);
  }
  tileOff = __shfl_sync(0xffffffffu, before, 0);
  roundTotal = __shfl_sync(0xffffffffu, all, 0);
  __syncthreads();  // tileCount is rewritten by the next round
}

// count_elems: nnz[s] = popcount of mask over shard s; block b counts shards b, b+gridDim.x, ...
__global__ void countElems(const uint64_t* mask, size_t nWords, size_t count, int nShards, unsigned* nnz) {
  __shared__ unsigned scratch[32];
  const size_t nRows = (count + 63) / 64;
  for (int s = blockIdx.x; s < nShards; s += gridDim.x) {
    const size_t start = (size_t)s * count;
    unsigned c = 0;
    #pragma unroll 8
    for (size_t i = threadIdx.x; i < nRows; i += blockDim.x) {
      c += __popcll(maskRow(mask, nWords, start + 64 * i, (long)(count - 64 * i)));
    }
    c = blockSum(c, scratch);
    if (threadIdx.x == 0) nnz[s] = c;
  }
}

// Compresses shard s of x into out[s*R, (s+1)*R) (values at the mask's
// positions, then zeros); block b compresses shards b, b+gridDim.x, ... The
// warps take tiles in rounds; within a tile, a warp walks the non-empty rows in
// order and the 32 lanes cover a row's 64 elements two apiece.
template<typename T>
__global__ void compressShards(const T* __restrict__ x, const uint64_t* mask, size_t nWords, size_t count,
                               int nShards, size_t R, T* __restrict__ out) {
  __shared__ unsigned tileCount[32];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  const unsigned bit = 2 * lane;
  const uint64_t below = (1ull << bit) - 1;
  const size_t nTiles = (count + CRS_TILE - 1) / CRS_TILE;
  for (int s = blockIdx.x; s < nShards; s += gridDim.x) {
    const size_t start = (size_t)s * count;
    T* o = out + (size_t)s * R;
    unsigned roundBase = 0;
    for (size_t t0 = 0; t0 < nTiles; t0 += nwarps) {
      const size_t t = t0 + warp;
      uint64_t r0 = 0, r1 = 0;
      if (t < nTiles) tileRows(mask, nWords, start, count, t, lane, r0, r1);
      unsigned tileOff, roundTotal;
      roundScan(r0, r1, tileCount, tileOff, roundTotal);
      unsigned base = roundBase + tileOff;
      uint64_t rows = (uint64_t)__ballot_sync(0xffffffffu, r0 != 0)
                    | (uint64_t)__ballot_sync(0xffffffffu, r1 != 0) << 32;
      const size_t tileElt = start + t * CRS_TILE;
      while (rows) {
        T v0[CRS_UNROLL], v1[CRS_UNROLL];
        bool m0[CRS_UNROLL], m1[CRS_UNROLL];
        unsigned p[CRS_UNROLL];
        #pragma unroll
        for (int u = 0; u < CRS_UNROLL; u++) {
          const int j = rows ? __ffsll((long long)rows) - 1 : -1;  // warp-uniform
          rows &= rows - 1;
          const uint64_t Rj = j >= 0
              ? (uint64_t)__shfl_sync(0xffffffffu, (unsigned long long)(j < 32 ? r0 : r1), j & 31) : 0;
          m0[u] = (Rj >> bit) & 1;
          m1[u] = (Rj >> (bit + 1)) & 1;
          p[u] = base + __popcll(Rj & below);
          base += __popcll(Rj);
          const size_t e = tileElt + (size_t)(j >= 0 ? j : 0) * 64 + bit;
          if (m0[u]) v0[u] = x[e];
          if (m1[u]) v1[u] = x[e + 1];
        }
        #pragma unroll
        for (int u = 0; u < CRS_UNROLL; u++) {
          if (m0[u]) o[p[u]] = v0[u];
          if (m1[u]) o[p[u] + m0[u]] = v1[u];
        }
      }
      roundBase += roundTotal;
    }
    for (size_t i = roundBase + threadIdx.x; i < R; i += blockDim.x) o[i] = T(0.0f);
  }
}

// Tiles [b*tpb, (b+1)*tpb) of [start, start+count) belong to block b of the decompress.
__device__ __forceinline__ void decompressRange(size_t count, size_t& tb0, size_t& tb1) {
  const size_t nTiles = (count + CRS_TILE - 1) / CRS_TILE, tpb = (nTiles + gridDim.x - 1) / gridDim.x;
  tb0 = min(blockIdx.x * tpb, nTiles);
  tb1 = min(tb0 + tpb, nTiles);
}

// blockNnz[b] = popcount of the mask over block b's tiles of [start, start+count).
__global__ void decompressCount(const uint64_t* mask, size_t nWords, size_t start, size_t count, unsigned* blockNnz) {
  __shared__ unsigned scratch[32];
  size_t tb0, tb1;
  decompressRange(count, tb0, tb1);
  const size_t rowEnd = min(tb1 * 64, (count + 63) / 64);
  unsigned c = 0;
  #pragma unroll 8
  for (size_t i = tb0 * 64 + threadIdx.x; i < rowEnd; i += blockDim.x) {
    c += __popcll(maskRow(mask, nWords, start + 64 * i, (long)(count - 64 * i)));
  }
  c = blockSum(c, scratch);
  if (threadIdx.x == 0) blockNnz[blockIdx.x] = c;
}

// out[i] = the compressed value of element start+i if the mask has it, else 0,
// for i < count. Block b starts at compressed position sum(blockNnz[0..b)).
template<typename T>
__global__ void decompressShard(const T* __restrict__ in, const uint64_t* mask, size_t nWords, size_t start,
                                size_t count, const unsigned* blockNnz, T* __restrict__ out) {
  __shared__ unsigned tileCount[32];
  __shared__ unsigned blockBase;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = blockDim.x / 32;
  const unsigned bit = 2 * lane;
  const uint64_t below = (1ull << bit) - 1;
  if (warp == 0) {
    unsigned v = 0;
    for (unsigned b = lane; b < blockIdx.x; b += 32) v += blockNnz[b];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) blockBase = v;
  }
  __syncthreads();
  size_t tb0, tb1;
  decompressRange(count, tb0, tb1);
  unsigned roundBase = blockBase;
  for (size_t t0 = tb0; t0 < tb1; t0 += nwarps) {
    const size_t t = t0 + warp;
    uint64_t r0 = 0, r1 = 0;
    if (t < tb1) tileRows(mask, nWords, start, count, t, lane, r0, r1);
    unsigned tileOff, roundTotal;
    roundScan(r0, r1, tileCount, tileOff, roundTotal);
    if (t < tb1) {  // warp-uniform
      unsigned base = roundBase + tileOff;
      for (int j0 = 0; j0 < 64; j0 += CRS_UNROLL) {
        T v0[CRS_UNROLL], v1[CRS_UNROLL];
        size_t e[CRS_UNROLL];
        #pragma unroll
        for (int u = 0; u < CRS_UNROLL; u++) {
          const int j = j0 + u;
          const uint64_t Rj = (uint64_t)__shfl_sync(0xffffffffu, (unsigned long long)(j < 32 ? r0 : r1), j & 31);
          const bool m0 = (Rj >> bit) & 1, m1 = (Rj >> (bit + 1)) & 1;
          const unsigned p = base + __popcll(Rj & below);
          base += __popcll(Rj);
          v0[u] = m0 ? in[p] : T(0.0f);
          v1[u] = m1 ? in[p + m0] : T(0.0f);
          e[u] = t * CRS_TILE + (size_t)j * 64 + bit;
        }
        #pragma unroll
        for (int u = 0; u < CRS_UNROLL; u++) {
          if (e[u] < count) out[e[u]] = v0[u];
          if (e[u] + 1 < count) out[e[u] + 1] = v1[u];
        }
      }
    }
    roundBase += roundTotal;
  }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Timer {
  cudaEvent_t start, stop;
  Timer() { CUDACHECK(cudaEventCreate(&start)); CUDACHECK(cudaEventCreate(&stop)); }
  ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};

template<typename T>
static void run(const Options& opt, int rank, int nranks, ncclComm_t comm, cudaStream_t stream) {
  const ncclDataType_t dt = DType<T>::nccl;
  const bool capture = opt.data == "capture";
  const char* channelsEnv = getenv("NCCL_CCD_CHANNELS");
  const std::string channels = channelsEnv ? channelsEnv : "default";
  const bool wantAgMask = has(opt.variants, "union") && has(opt.unionMasks, "ag");
  const bool wantArMask = has(opt.variants, "union") && has(opt.unionMasks, "ar");
  const bool wantCooAg = has(opt.variants, "mask") && has(opt.exchanges, "coo-ag");
  const bool wantCooBc = has(opt.variants, "mask") && has(opt.exchanges, "coo-bcast");
  // The AG masks feed the AG-mask union runs and check the other exchanges.
  const bool needAG = wantAgMask || has(opt.variants, "mask");
  const bool needAR = wantArMask || (has(opt.variants, "mask") && has(opt.exchanges, "orreduce"));

  // Problem sizes (recvcount per rank) and data source.
  Capture cap;
  int capRank = rank;
  std::vector<size_t> counts;
  if (capture) {
    cap = readManifest(opt.capture);
    if (nranks > cap.nranks && !opt.wrap) {
      die("capture has fewer ranks than this run; pass --wrap to reuse them (%s)", opt.capture.c_str());
    }
    capRank = rank % cap.nranks;
    if (opt.offset >= cap.numel) die("--offset is past the end of the capture");
  }
  for (double mib : opt.sizesMiB) counts.push_back((size_t)(mib * (1 << 20)) / (nranks * sizeof(T)));
  if (counts.empty()) {
    counts.push_back(opt.count ? opt.count : capture ? (cap.numel - opt.offset) / nranks : (size_t)1 << 20);
  }
  for (size_t c : counts) {
    if (c == 0) die("a requested size is too small for this many ranks");
    if (capture && opt.offset + c * nranks > cap.numel) die("--offset/--count window exceeds the capture");
  }
  const std::vector<double> densities = capture ? std::vector<double>{-1.0} : opt.densities;
  const size_t maxCount = *std::max_element(counts.begin(), counts.end());
  const size_t maxTotal = maxCount * nranks;
  const size_t maxMaskWords = (maxTotal + 63) / 64;

  if (rank == 0) {
    const double gib = 1.0 / (1 << 30);
    printf("# nranks=%d dtype=%s channels=%s kernels=%dx%d iters=%d warmup=%d", nranks, opt.dtype.c_str(),
           channels.c_str(), opt.grid, opt.block, opt.iters, opt.warmup);
    if (capture) {
      printf(" data=capture(%s, iter %ld, %d ranks%s) offset=%zu of %zu\n", opt.capture.c_str(), cap.iteration,
             cap.nranks, nranks > cap.nranks ? ", wrapped" : "", opt.offset, cap.numel);
    } else {
      printf(" data=iid(overlap %.2f, seed %lu)\n", opt.overlap, (unsigned long)opt.seed);
    }
    printf("# device memory per rank (before COO and Compressed RS buffers): %.2f GiB\n",
           (2 * maxTotal * sizeof(T) + 2 * maxCount * sizeof(T) + ((needAG ? nranks : 0) + 2) * maxMaskWords * 8) * gib);
    if (opt.csv) {
      printf("nranks,channels,size_mib,count,dtype,data,density,variant,N,order,masks,avg_us,min_us,mismatches,"
             "pass_counts\n");
    }
  }

  T *pristine, *send, *recv, *ref;
  uint64_t *mask, *agMasks = nullptr, *arMask;
  unsigned long long* counter;
  CUDACHECK(cudaMalloc(&pristine, maxTotal * sizeof(T)));
  CUDACHECK(cudaMalloc(&send, maxTotal * sizeof(T)));
  CUDACHECK(cudaMalloc(&recv, maxCount * sizeof(T)));
  CUDACHECK(cudaMalloc(&ref, maxCount * sizeof(T)));
  CUDACHECK(cudaMalloc(&mask, maxMaskWords * sizeof(uint64_t)));
  if (needAG) CUDACHECK(cudaMalloc(&agMasks, (size_t)nranks * maxMaskWords * sizeof(uint64_t)));
  CUDACHECK(cudaMalloc(&arMask, maxMaskWords * sizeof(uint64_t)));
  CUDACHECK(cudaMalloc(&counter, sizeof(unsigned long long)));
  Timer timer;
  constexpr int MaxPhases = 8;
  cudaEvent_t phaseEv[MaxPhases + 1];
  for (auto& e : phaseEv) CUDACHECK(cudaEventCreate(&e));

  for (size_t count : counts) {
  for (double density : densities) {
    const size_t total = count * nranks;
    const size_t maskWords = (total + 63) / 64;
    const double sizeMiB = (double)total * sizeof(T) / (1 << 20);

    if (capture) {
      char name[64];
      snprintf(name, sizeof(name), "/rank_%05d.bits", capRank);
      std::vector<uint8_t> bits = readBits(opt.capture + name, opt.offset, total);
      uint8_t* dBits;
      CUDACHECK(cudaMalloc(&dBits, bits.size()));
      CUDACHECK(cudaMemcpy(dBits, bits.data(), bits.size(), cudaMemcpyHostToDevice));
      fillFromBits<T><<<1024, 256, 0, stream>>>(pristine, total, dBits, (unsigned)(opt.offset % 8), rank, opt.seed);
      CUDACHECK(cudaGetLastError());
      CUDACHECK(cudaStreamSynchronize(stream));
      CUDACHECK(cudaFree(dBits));
    } else {
      fillIid<T><<<1024, 256, 0, stream>>>(pristine, total, rank, density, opt.overlap, opt.seed);
      CUDACHECK(cudaGetLastError());
    }

    // This rank's nonzeros, their max over ranks (sizes the COO buffers) and
    // the measured density averaged over ranks.
    unsigned long long nnz = 0, maxNnz = 0, sumNnz = 0;
    {
      CUDACHECK(cudaMemsetAsync(counter, 0, sizeof(*counter), stream));
      countNonzero<T><<<1024, 256, 0, stream>>>(pristine, total, counter);
      CUDACHECK(cudaMemcpyAsync(&nnz, counter, sizeof(nnz), cudaMemcpyDeviceToHost, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      MPI_Allreduce(&nnz, &sumNnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&nnz, &maxNnz, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    }
    const double measured = (double)sumNnz / ((double)total * nranks);
    if (rank == 0 && !opt.csv) {
      printf("\n# size %.1f MiB (count %zu)", sizeMiB, count);
      if (capture) printf(" window [%zu,%zu)", opt.offset, opt.offset + total);
      else printf(" density %.6g", density);
      printf(", measured density %.6f, max nnz per rank %llu\n", measured, maxNnz);
      fflush(stdout);
    }

    // Reference: dense reduce-scatter (does not modify its sendbuff).
    NCCLCHECK(ncclReduceScatter(pristine, ref, count, dt, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Times `launch` (which must enqueue on `stream`); `prep` runs untimed first.
    // Returns {avg, min} over iterations of the max time across ranks, in us.
    auto measure = [&](auto prep, auto launch) {
      double sumT = 0, minT = 1e30;
      for (int it = -opt.warmup; it < opt.iters; it++) {
        prep();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        CUDACHECK(cudaEventRecord(timer.start, stream));
        launch();
        CUDACHECK(cudaEventRecord(timer.stop, stream));
        CUDACHECK(cudaEventSynchronize(timer.stop));
        float ms;
        CUDACHECK(cudaEventElapsedTime(&ms, timer.start, timer.stop));
        double us = ms * 1e3, maxUs;
        MPI_Allreduce(&us, &maxUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (it >= 0) { sumT += maxUs; minT = std::min(minT, maxUs); }
      }
      return std::make_pair(sumT / std::max(opt.iters, 1), minT);
    };
    // Times a pipeline of nPhases phases: `launch(mark)` enqueues them on
    // `stream` and calls mark() after each, which records an event. Per
    // iteration the phase times are those of the rank with the largest total.
    // Returns {avg, min} over iterations per phase, then for the total, in us.
    auto measurePhases = [&](int nPhases, auto prep, auto launch) {
      std::vector<double> sum(nPhases + 1, 0.0), mn(nPhases + 1, 1e30), d(nPhases + 1);
      for (int it = -opt.warmup; it < opt.iters; it++) {
        prep();
        CUDACHECK(cudaStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        int k = 0;
        CUDACHECK(cudaEventRecord(phaseEv[0], stream));
        launch([&] {
          if (k == nPhases) die("pipeline has more phases than declared");
          CUDACHECK(cudaEventRecord(phaseEv[++k], stream));
        });
        if (k != nPhases) die("pipeline has fewer phases than declared");
        CUDACHECK(cudaEventSynchronize(phaseEv[nPhases]));
        float ms;
        for (int i = 0; i < nPhases; i++) {
          CUDACHECK(cudaEventElapsedTime(&ms, phaseEv[i], phaseEv[i + 1]));
          d[i] = ms * 1e3;
        }
        CUDACHECK(cudaEventElapsedTime(&ms, phaseEv[0], phaseEv[nPhases]));
        d[nPhases] = ms * 1e3;
        struct { double v; int r; } mine{d[nPhases], rank}, slowest;
        MPI_Allreduce(&mine, &slowest, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        MPI_Bcast(d.data(), nPhases + 1, MPI_DOUBLE, slowest.r, MPI_COMM_WORLD);
        if (it >= 0) for (int i = 0; i <= nPhases; i++) { sum[i] += d[i]; mn[i] = std::min(mn[i], d[i]); }
      }
      std::vector<std::pair<double, double>> out;
      for (int i = 0; i <= nPhases; i++) out.emplace_back(sum[i] / std::max(opt.iters, 1), mn[i]);
      return out;
    };
    auto restore = [&]() {
      CUDACHECK(cudaMemcpyAsync(send, pristine, total * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    };
    // Mismatching elements of a vs b (n of them), summed over ranks.
    auto mismatches = [&](auto* a, auto* b, size_t n) -> long long {
      using E = typename std::remove_const<typename std::remove_pointer<decltype(a)>::type>::type;
      unsigned long long bad = 0, allBad = 0;
      CUDACHECK(cudaMemsetAsync(counter, 0, sizeof(*counter), stream));
      countMismatch<E><<<1024, 256, 0, stream>>>(a, b, n, counter);
      CUDACHECK(cudaMemcpyAsync(&bad, counter, sizeof(bad), cudaMemcpyDeviceToHost, stream));
      CUDACHECK(cudaStreamSynchronize(stream));
      MPI_Allreduce(&bad, &allBad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      return (long long)allBad;
    };
    // recv vs the dense reference (-1: not checked).
    auto check = [&]() -> long long { return opt.check ? mismatches(recv, ref, count) : -1; };
    int passCounts = -1;  // NCCL_CCD_UNION_PASS_COUNTS of the union rows being reported (-1: not applicable)
    // Order 2 (pairs) runs only in fp32 with N >= nranks, and once per
    // --pass-counts list, since it doesn't use the tile offsets.
    auto skipUnion = [&](int order, int n, int pc) {
      return order == 2 && (n < nranks || !std::is_same<T, float>::value || pc != opt.passCounts.front());
    };
    auto report = [&](const char* name, int n, int order, const char* masks, std::pair<double, double> t,
                      long long bad) {
      if (rank != 0) return;
      if (opt.csv) {
        printf("%d,%s,%.3f,%zu,%s,%s,%.6g,%s,%d,%d,%s,%.2f,%.2f,%lld,%d\n", nranks, channels.c_str(), sizeMiB, count,
               opt.dtype.c_str(), opt.data.c_str(), capture ? measured : density, name, n, order, masks,
               t.first, t.second, bad, passCounts);
      } else {
        const char* status = bad < 0 ? "-" : bad == 0 ? "PASS" : "FAIL";
        printf("%-11s N=%-3d order=%-2d %-7s pc=%-2d avg %10.2f us  min %10.2f us  %s", name, n, order, masks,
               passCounts, t.first, t.second, status);
        if (bad > 0) printf(" (%lld mismatches)", bad);
        printf("\n");
      }
      fflush(stdout);
    };

    if (has(opt.variants, "dense")) {
      auto t = measure([] {}, [&] { NCCLCHECK(ncclReduceScatter(pristine, recv, count, dt, ncclSum, comm, stream)); });
      report("dense", 0, -1, "-", t, check());
    }

    // Sparse variants modify their sendbuff, so restore it before each iteration.
    if (has(opt.variants, "spccl")) {
      auto t = measure(restore, [&] { NCCLCHECK(ncclReduceScatterSparse(send, recv, count, dt, ncclSum, comm, stream)); });
      report("spccl", 1, -1, "-", t, check());
    }

    // ---- Extract bitmask and the mask exchanges ----
    const bool reportMask = has(opt.variants, "mask");
    auto extractOwnBitmask = [&] { extractBitmask<T><<<opt.grid, opt.block, 0, stream>>>(pristine, total, mask); };
    auto t = measure([] {}, extractOwnBitmask);
    CUDACHECK(cudaGetLastError());
    if (reportMask) report("extract", 0, -1, "-", t, -1);

    bool haveAgMasks = false;
    if (needAG) {
      t = measure([] {}, [&] { NCCLCHECK(ncclAllGather(mask, agMasks, maskWords, ncclUint64, comm, stream)); });
      haveAgMasks = true;
      if (reportMask && has(opt.exchanges, "bitmask")) report("maskAG", 0, -1, "-", t, -1);
    }

    if (wantCooAg || wantCooBc) {
      if (total > UINT32_MAX) {
        if (rank == 0) printf("# skipping COO exchanges: %zu elements do not fit 32-bit keys\n", total);
      } else if (maxNnz > 0) {
        // COO buffers: own keys (padded to the largest rank for the allgather),
        // gathered keys in both layouts, decoded masks, counts and offsets.
        uint32_t *keys, *keysAG = nullptr, *keysBC = nullptr;
        uint64_t* cooMasks;
        unsigned long long *dCount, *dCounts, *dOffsets;
        CUDACHECK(cudaMalloc(&keys, maxNnz * sizeof(uint32_t)));
        if (wantCooAg) CUDACHECK(cudaMalloc(&keysAG, (size_t)nranks * maxNnz * sizeof(uint32_t)));
        if (wantCooBc) CUDACHECK(cudaMalloc(&keysBC, sumNnz * sizeof(uint32_t)));
        CUDACHECK(cudaMalloc(&cooMasks, (size_t)nranks * maskWords * sizeof(uint64_t)));
        CUDACHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
        CUDACHECK(cudaMalloc(&dCounts, nranks * sizeof(unsigned long long)));
        CUDACHECK(cudaMalloc(&dOffsets, nranks * sizeof(unsigned long long)));
        std::vector<unsigned long long> hCounts(nranks), hOffsets(nranks);

        const thrust::counting_iterator<uint32_t> idx(0);
        const NonzeroAt<T> nonzero{pristine};
        size_t cubBytes = 0;
        void* cubTemp;
        CUDACHECK(cub::DeviceSelect::If(nullptr, cubBytes, idx, keys, dCount, (int64_t)total, nonzero, stream));
        CUDACHECK(cudaMalloc(&cubTemp, cubBytes));

        t = measure([] {}, [&] {
          CUDACHECK(cub::DeviceSelect::If(cubTemp, cubBytes, idx, keys, dCount, (int64_t)total, nonzero, stream));
        });
        report("cooEncode", 0, -1, "-", t, -1);

        // Counts are needed on the host to size the transfers, so each
        // exchange includes sharing them: a tiny ncclAllGather plus the copy
        // back, or (--count-exchange mpi) an MPI_Allgather of the host copy,
        // uploaded again for the decode.
        auto exchangeCounts = [&]() {
          if (opt.countExchange == "mpi") {
            unsigned long long mine = 0;
            CUDACHECK(cudaMemcpyAsync(&mine, dCount, sizeof(mine), cudaMemcpyDeviceToHost, stream));
            CUDACHECK(cudaStreamSynchronize(stream));
            MPI_Allgather(&mine, 1, MPI_UNSIGNED_LONG_LONG, hCounts.data(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
            CUDACHECK(cudaMemcpyAsync(dCounts, hCounts.data(), nranks * sizeof(unsigned long long),
                                      cudaMemcpyHostToDevice, stream));
            return;
          }
          NCCLCHECK(ncclAllGather(dCount, dCounts, 1, ncclUint64, comm, stream));
          CUDACHECK(cudaMemcpyAsync(hCounts.data(), dCounts, nranks * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, stream));
          CUDACHECK(cudaStreamSynchronize(stream));
        };
        auto decode = [&](const uint32_t* gathered) {
          CUDACHECK(cudaMemcpyAsync(dOffsets, hOffsets.data(), nranks * sizeof(unsigned long long),
                                    cudaMemcpyHostToDevice, stream));
          CUDACHECK(cudaMemsetAsync(cooMasks, 0, (size_t)nranks * maskWords * sizeof(uint64_t), stream));
          decodeCoo<<<dim3(256, nranks), 256, 0, stream>>>(gathered, dOffsets, dCounts, cooMasks, maskWords);
        };
        auto checkMasks = [&]() -> long long {
          return haveAgMasks ? mismatches(cooMasks, agMasks, (size_t)nranks * maskWords) : -1;
        };

        if (wantCooAg) {
          t = measure([] {}, [&] {
            exchangeCounts();
            const unsigned long long width = *std::max_element(hCounts.begin(), hCounts.end());
            NCCLCHECK(ncclAllGather(keys, keysAG, width, ncclUint32, comm, stream));
          });
          report("cooAG", 0, -1, "-", t, -1);
          const unsigned long long width = *std::max_element(hCounts.begin(), hCounts.end());
          for (int r = 0; r < nranks; r++) hOffsets[r] = r * width;
          t = measure([] {}, [&] { decode(keysAG); });
          CUDACHECK(cudaGetLastError());
          report("cooDecodeAG", 0, -1, "-", t, checkMasks());
        }

        if (wantCooBc) {
          t = measure([] {}, [&] {
            exchangeCounts();
            unsigned long long off = 0;
            for (int r = 0; r < nranks; r++) { hOffsets[r] = off; off += hCounts[r]; }
            NCCLCHECK(ncclGroupStart());
            for (int r = 0; r < nranks; r++) {
              if (hCounts[r] == 0) continue;
              NCCLCHECK(ncclBroadcast(keys, keysBC + hOffsets[r], hCounts[r], ncclUint32, r, comm, stream));
            }
            NCCLCHECK(ncclGroupEnd());
          });
          report("cooBcast", 0, -1, "-", t, -1);
          t = measure([] {}, [&] { decode(keysBC); });
          CUDACHECK(cudaGetLastError());
          report("cooDecodeBc", 0, -1, "-", t, checkMasks());
        }

        CUDACHECK(cudaFree(keys)); CUDACHECK(cudaFree(cooMasks)); CUDACHECK(cudaFree(cubTemp));
        if (keysAG) CUDACHECK(cudaFree(keysAG));
        if (keysBC) CUDACHECK(cudaFree(keysBC));
        CUDACHECK(cudaFree(dCount)); CUDACHECK(cudaFree(dCounts)); CUDACHECK(cudaFree(dOffsets));
      }
    }

    if (needAR) {
      t = measure([] {}, [&] {
        NCCLCHECK(ncclAllReduce(mask, arMask, maskWords, ncclUint64, ncclBitOr, comm, stream));
      });
      long long bad = -1;
      if (haveAgMasks) {
        // Compare against the OR of the gathered masks, computed locally.
        uint64_t* expect;
        CUDACHECK(cudaMalloc(&expect, maskWords * sizeof(uint64_t)));
        orMasks<<<1024, 256, 0, stream>>>(agMasks, nranks, maskWords, expect);
        CUDACHECK(cudaGetLastError());
        bad = mismatches(arMask, expect, maskWords);
        CUDACHECK(cudaFree(expect));
      }
      if (reportMask && has(opt.exchanges, "orreduce")) report("maskAR", 0, -1, "-", t, bad);
    }

    // ---- Union-pack reduce-scatter ----
    if (has(opt.variants, "union")) {
      std::vector<int> ns = opt.ns;
      if (ns.empty()) for (int n = 1; n <= nranks; n++) ns.push_back(n);
      for (const std::string& um : opt.unionMasks) {
        if (um != "ag" && um != "ar") die("--union-masks takes ag and/or ar");
        const uint64_t* masks = um == "ar" ? arMask : agMasks;
        setenv("NCCL_CCD_UNION_MASK", um.c_str(), 1);
        for (int pc : opt.passCounts) {
          setenv("NCCL_CCD_UNION_PASS_COUNTS", std::to_string(pc).c_str(), 1);
          for (int order : opt.orders) {
            setenv("NCCL_CCD_UNION_ORDER", std::to_string(order).c_str(), 1);
            for (int n : ns) {
              if (skipUnion(order, n, pc)) continue;
              passCounts = order == 2 ? -1 : pc;
              t = measure(restore, [&] {
                NCCLCHECK(ncclReduceScatterSparseUnion(send, recv, count, dt, ncclSum, n, masks, comm, stream));
              });
              report("union", n, order, um.c_str(), t, check());
            }
          }
        }
        passCounts = -1;
      }
    }

    // ---- Compressed RS, one pipeline ----
    if (has(opt.variants, "compressed-rs")) {
      unsigned *dShardNnz, *dBlockNnz;
      CUDACHECK(cudaMalloc(&dShardNnz, nranks * sizeof(unsigned)));
      CUDACHECK(cudaMalloc(&dBlockNnz, opt.grid * sizeof(unsigned)));
      std::vector<unsigned> shardNnz(nranks);
      // count_elems: nnz of the AR mask in every shard, read back to size the
      // RS: R = the largest shard, rounded up to 64 elements.
      auto countElemsReadback = [&]() -> size_t {
        countElems<<<opt.grid, opt.block, 0, stream>>>(arMask, maskWords, count, nranks, dShardNnz);
        CUDACHECK(cudaMemcpyAsync(shardNnz.data(), dShardNnz, nranks * sizeof(unsigned), cudaMemcpyDeviceToHost,
                                  stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        const size_t mx = *std::max_element(shardNnz.begin(), shardNnz.end());
        return std::max<size_t>((mx + 63) / 64 * 64, 64);
      };
      // An untimed run of the first phases sizes the compressed buffers.
      extractOwnBitmask();
      NCCLCHECK(ncclAllReduce(mask, arMask, maskWords, ncclUint64, ncclBitOr, comm, stream));
      const size_t R = countElemsReadback();
      if (rank == 0) {
        const auto mm = std::minmax_element(shardNnz.begin(), shardNnz.end());
        printf("# compressed-rs: recvcount %zu of %zu (shard nnz %u..%u), compressed sendbuff %.2f GiB\n", R, count,
               *mm.first, *mm.second, (double)nranks * R * sizeof(T) / (1 << 30));
      }
      T *compressed, *compressedRecv;
      CUDACHECK(cudaMalloc(&compressed, (size_t)nranks * R * sizeof(T)));
      CUDACHECK(cudaMalloc(&compressedRecv, R * sizeof(T)));
      const size_t myStart = (size_t)rank * count;
      auto ph = measurePhases(6, [] {}, [&](auto mark) {
        extractOwnBitmask();
        mark();
        NCCLCHECK(ncclAllReduce(mask, arMask, maskWords, ncclUint64, ncclBitOr, comm, stream));
        mark();
        if (countElemsReadback() != R) die("compressed-rs: the shard counts changed between iterations");
        mark();
        compressShards<T><<<opt.grid, opt.block, 0, stream>>>(pristine, arMask, maskWords, count, nranks, R,
                                                               compressed);
        mark();
        NCCLCHECK(ncclReduceScatter(compressed, compressedRecv, R, dt, ncclSum, comm, stream));
        mark();
        decompressCount<<<opt.grid, opt.block, 0, stream>>>(arMask, maskWords, myStart, count, dBlockNnz);
        decompressShard<T><<<opt.grid, opt.block, 0, stream>>>(compressedRecv, arMask, maskWords, myStart, count,
                                                                dBlockNnz, recv);
        mark();
      });
      CUDACHECK(cudaGetLastError());
      const char* names[] = {"compressedRS.extract", "compressedRS.maskAR", "compressedRS.count_elems",
                             "compressedRS.compress", "compressedRS.rs", "compressedRS.decompress"};
      const long long bad = check();
      for (int i = 0; i < 6; i++) report(names[i], 0, -1, "ar", ph[i], -1);
      report("compressedRS.total", 0, -1, "ar", ph[6], bad);
      CUDACHECK(cudaFree(compressed)); CUDACHECK(cudaFree(compressedRecv));
      CUDACHECK(cudaFree(dShardNnz)); CUDACHECK(cudaFree(dBlockNnz));
    }

    // ---- Union-pack RS with the AR mask, one pipeline ----
    if (has(opt.variants, "union-e2e")) {
      std::vector<int> ns = opt.ns;
      if (ns.empty()) for (int n = 1; n <= nranks; n++) ns.push_back(n);
      setenv("NCCL_CCD_UNION_MASK", "ar", 1);
      for (int pc : opt.passCounts) {
        setenv("NCCL_CCD_UNION_PASS_COUNTS", std::to_string(pc).c_str(), 1);
        for (int order : opt.orders) {
          setenv("NCCL_CCD_UNION_ORDER", std::to_string(order).c_str(), 1);
          for (int n : ns) {
            if (skipUnion(order, n, pc)) continue;
            passCounts = order == 2 ? -1 : pc;
            auto ph = measurePhases(3, restore, [&](auto mark) {
              extractOwnBitmask();
              mark();
              NCCLCHECK(ncclAllReduce(mask, arMask, maskWords, ncclUint64, ncclBitOr, comm, stream));
              mark();
              NCCLCHECK(ncclReduceScatterSparseUnion(send, recv, count, dt, ncclSum, n, arMask, comm, stream));
              mark();
            });
            CUDACHECK(cudaGetLastError());
            const long long bad = check();
            report("unionE2E.extract", n, order, "ar", ph[0], -1);
            report("unionE2E.maskAR", n, order, "ar", ph[1], -1);
            report("unionE2E.rs", n, order, "ar", ph[2], -1);
            report("unionE2E.total", n, order, "ar", ph[3], bad);
          }
        }
      }
      passCounts = -1;
    }
  }
  }

  CUDACHECK(cudaFree(pristine)); CUDACHECK(cudaFree(send)); CUDACHECK(cudaFree(recv)); CUDACHECK(cudaFree(ref));
  CUDACHECK(cudaFree(mask)); CUDACHECK(cudaFree(agMasks)); CUDACHECK(cudaFree(arMask));
  CUDACHECK(cudaFree(counter));
  for (auto& e : phaseEv) CUDACHECK(cudaEventDestroy(e));
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  gRank = rank;
  Options opt = parseArgs(argc, argv);

  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  int localRank, nDev;
  MPI_Comm_rank(local, &localRank);
  CUDACHECK(cudaGetDeviceCount(&nDev));
  CUDACHECK(cudaSetDevice(localRank % nDev));

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  if (opt.dtype == "f32") run<float>(opt, rank, nranks, comm, stream);
  else if (opt.dtype == "f16") run<__half>(opt, rank, nranks, comm, stream);
  else if (opt.dtype == "bf16") run<__nv_bfloat16>(opt, rank, nranks, comm, stream);
  else die("--dtype must be f32, f16 or bf16");

  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return 0;
}
