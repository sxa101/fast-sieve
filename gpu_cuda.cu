/* gpu_cuda.cu - CUDA port of the OpenCL segmented-sieve accelerator
   (gpu.c). Each thread block sieves one GPU-block (30*16384 numbers) into
   16 KiB of shared memory (256 lanes), crossing off multiples of every
   sieving prime p with p*p in the block, then counts survivors with a
   work-group popcount reduction.

   Wheel-30 layout: 8 candidate bits per 30 numbers; each candidate bit is
   cleared with a word-level shared-memory atomicAnd so lanes racing on the
   same byte are race-free (CUDA shared semantics + __syncthreads make this
   deterministic, unlike the RDNA4 OpenCL blob).

   Boundary convention (must match fastsieve.c exactly):
     - byte j covers [segLow+30j+7, segLow+30j+31]; a value v == 1 (mod 30)
       lives at offset 31 of byte (v-6)/30, i.e. one byte EARLIER.
     - a full block owns candidates up to segEnd+1 (the trailing offset-31
       candidate of its last byte); the final block counts candidates < top.
     - a prime is added while p*p <= segEnd+1, and its multiples are crossed
       up to v <= segEnd+1 (qmax = (segEnd+1)/p).

   Build:  nvcc -O3 -arch=sm_86 -Xcompiler -fopenmp gpu_cuda.cu fastsieve.c -o fastsieve
   Select device (0 = first CUDA GPU):  FASTSIEVE_DEVICE=1 ./fastsieve --gpu 1e11
*/
#include "gpu.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#define SEG_BYTES 16384u

typedef uint64_t u64;
typedef uint32_t u32;

/* ---- small helper sieve, identical to gpu.c / fastsieve.c ---- */
static u64* simple_primes(u64 limit, u64* cnt) {
  u64 words = (limit / 2 + 2 + 63) / 64;
  u64* bits = (u64*)calloc((size_t)words, 8);
  u64* out; u64 c = 0, i;
  if (!bits) { *cnt = 0; return 0; }
  for (i = 1; 2 * i + 1 <= limit; i++) {
    u64 odd = 2 * i + 1;
    if (!(bits[i >> 6] & (1ull << (i & 63))))
      if (odd * odd <= limit)
        for (u64 j = odd * odd; j <= limit; j += 2 * odd)
          bits[(j / 2) >> 6] |= 1ull << ((j / 2) & 63);
  }
  for (i = 1; 2 * i + 1 <= limit; i++) if (!(bits[i >> 6] & (1ull << (i & 63)))) c++;
  c++;
  out = (u64*)malloc(sizeof(u64) * (size_t)(c + 1));
  if (!out) { free(bits); *cnt = 0; return 0; }
  c = 0; out[c++] = 2;
  for (i = 1; 2 * i + 1 <= limit; i++) if (!(bits[i >> 6] & (1ull << (i & 63)))) out[c++] = 2 * i + 1;
  free(bits); *cnt = c; return out;
}

/* ---- kernel ---- */
__device__ __forceinline__ int bitidx(unsigned char r) {
  if (r == 1) return 7; if (r == 7) return 0; if (r == 11) return 1;
  if (r == 13) return 2; if (r == 17) return 3; if (r == 19) return 4;
  if (r == 23) return 5; if (r == 29) return 6; return 7;
}
__device__ __forceinline__ int isunit(unsigned char r) {
  return (r == 1 || r == 7 || r == 11 || r == 13 || r == 17 || r == 19 || r == 23 || r == 29);
}

/* Division by reciprocal-multiply: a / p with m = floor(2^64 / p) precomputed
   on the host. mulhi(a,m) is q-1 or q (error a*e/2^64 < e < p), so exactly one
   conditional correction recovers the exact quotient. Replaces the two
   software u64 divisions per sieving prime per block (the measured hot spot). */
__device__ __forceinline__ unsigned long long divf_m(unsigned long long a, u32 p,
                                                     unsigned long long m) {
  unsigned long long q = __umul64hi(a, m);
  if (a - q * (unsigned long long)p >= (unsigned long long)p) q++;
  return q;
}

#define GSIEVE_CHUNK 2048u   /* primes per shared-memory chunk (8 KiB) */

/* v3 kernel: phase-split crossing, v1 memory layout (16 KiB shared/CTA -> 6
   CTAs/SM). Measured lesson (gsieve2, K blocks/CTA + chunked prime list): the
   scan amortization does NOT pay for the occupancy it costs (1e11: 32.8 s vs
   14.7 s). The real waste is SIMT-serial scanning of ~sqrt(segEnd) primes per
   block with idle lanes for p > 491520/256 = 1920:
     phase 1 (p < 1920, ~296 primes): cooperative lane-strided q-loop;
     phase 2 (1920 <= p <= sqrt(segEnd+1)): ONE prime per lane - the scan is
       256x parallel and no lane idles; each lane crosses its primes' few
       multiples directly (same atomicAnd bit-clear as phase 1).
   A per-block binary search bounds the active prime set (pp <= segEnd+1). */
__device__ __forceinline__ void clear_mult(unsigned char* seg, unsigned long long v,
                                           unsigned long long segLow) {
  if (v >= segLow + 7 && v < segLow + (unsigned long long)30 * SEG_BYTES + 2) {
    unsigned char r = (unsigned char)(v % 30);
    if (isunit(r)) {
      unsigned long long idx = (v - segLow) / 30;
      if (r == 1) idx -= 1;      /* residue 1 lives at offset 31 = one byte earlier */
      if (idx < SEG_BYTES) {
        u32 wi = (u32)idx >> 2;
        u32 by = (u32)idx & 3;
        u32 wordMask = ~(((u32)1u << bitidx(r)) << (8 * by));
        atomicAnd((u32*)&seg[wi << 2], wordMask);
      }
    }
  }
}

__global__ void __launch_bounds__(256, 6)
gsieve(const u32* prim, const unsigned long long* pm, u32 np, u32 p1idx,
       unsigned long long top, unsigned long long* counters,
       unsigned char* dump = 0, unsigned int dumpBlock = 0) {
  const u32 gid = blockIdx.x;
  const u32 lid = threadIdx.x;
  const u32 L   = blockDim.x;
  const unsigned long long segLow = (unsigned long long)30 * SEG_BYTES * gid;
  const unsigned long long segEnd = segLow + (unsigned long long)30 * SEG_BYTES;

  __shared__ __align__(16) unsigned char seg[SEG_BYTES];
  for (u32 i = lid; i < SEG_BYTES; i += L) seg[i] = 0xFF;
  __syncthreads();

  /* lo = number of active sieving primes: prim[0..lo) have p*p <= segEnd+1 */
  u32 lo = 0, hi = np;
  while (lo < hi) {
    const u32 mid = (lo + hi) >> 1;
    if ((unsigned long long)prim[mid] * prim[mid] <= segEnd + 1) lo = mid + 1;
    else hi = mid;
  }

  /* phase 1: small primes, cooperative q-strides over all 256 lanes */
  const u32 lim1 = (p1idx < lo) ? p1idx : lo;
  for (u32 pi = 0; pi < lim1; pi++) {
    const u32 p = prim[pi];
    if (p < 7) continue;
    const unsigned long long pp = (unsigned long long)p * p;
    const unsigned long long mp = pm[pi];
    /* q0 = ceil(v0/p); common case p*p >= segLow+7 gives v0 = pp -> q0 = p */
    const unsigned long long q0 = (pp >= segLow + 7) ? (unsigned long long)p
                                                     : divf_m(segLow + 6 + p, p, mp);
    const unsigned long long qmax = divf_m(segEnd + 1, p, mp);
    for (unsigned long long q = q0 + lid; q <= qmax; q += L)
      clear_mult(seg, p * q, segLow);
  }

  /* phase 2: large primes, one prime per lane, multiples crossed serially */
  for (u32 pi = p1idx + lid; pi < lo; pi += L) {
    const u32 p = prim[pi];
    const unsigned long long pp = (unsigned long long)p * p;
    const unsigned long long mp = pm[pi];
    const unsigned long long q0 = (pp >= segLow + 7) ? (unsigned long long)p
                                                     : divf_m(segLow + 6 + p, p, mp);
    const unsigned long long qmax = divf_m(segEnd + 1, p, mp);
    for (unsigned long long q = q0; q <= qmax; q++)
      clear_mult(seg, p * q, segLow);
  }
  __syncthreads();

  /* count candidates < cap, where a full block owns up to segEnd+1 */
  unsigned long long cap = (segEnd < top) ? (segEnd + 2) : top;
  unsigned long long rel = cap - segLow;
  /* NOTE: rel may be 30*SEG_BYTES+2 (full block incl. trailing candidate);
     do NOT clamp rel - the nbits<=SEG_BYTES*8 clamp below suffices, and
     clamping rel here would drop the segEnd+1 candidate from the count. */
  unsigned long long nbits = 0;
  if (rel > 1) {
    unsigned long long full = rel / 30;
    unsigned long long rem  = rel % 30;
    /* CUNIT: number of wheel-30 unit residues in [1, rem) - identical to
       fastsieve.c build_count_tables(); candidate '1' is not stored. */
    const int cup[30] = {0,0,1,1,1,1,1,1,2,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6,7,7,7,7,7,7};
    nbits = full * 8 + (unsigned long long)cup[rem];
    nbits -= 1;
  }
  if (nbits > (unsigned long long)SEG_BYTES * 8) nbits = (unsigned long long)SEG_BYTES * 8;

  unsigned long long nf = nbits / 64, nb = nbits & 63;
  unsigned long long cnt = 0;
  {
    unsigned long long* w = (unsigned long long*)seg;
    for (unsigned long long i = lid; i < nf; i += L) cnt += (unsigned long long)__popcll(w[i]);
    if (nb && lid == 0) cnt += (unsigned long long)__popcll(w[nf] & ((1ull << nb) - 1));
  }
  __syncthreads();
  __shared__ unsigned long long lsum[256];
  lsum[lid] = cnt; __syncthreads();
  for (u32 s = L / 2; s > 0; s >>= 1) {
    if (lid < s) lsum[lid] += lsum[lid + s];
    __syncthreads();
  }
  if (lid == 0) counters[gid] = lsum[0];
  if (dump && gid == dumpBlock && lid == 0)
    for (u32 i = 0; i < SEG_BYTES; i++) dump[i] = seg[i];
}

/* ---- host side ---- */
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

extern "C" GpuResult gpu_sieve(uint64_t top, uint64_t* block_counts, double* secs_out) {
  GpuResult res; memset(&res, 0, sizeof(res));

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return res;
  int devidx = 0;
  const char* env = getenv("FASTSIEVE_DEVICE");
  if (env && *env) {
    int v = atoi(env);
    if (v >= 0 && v < ndev) devidx = v;
  }
  if (cudaSetDevice(devidx) != cudaSuccess) return res;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, devidx) != cudaSuccess) return res;
  snprintf(res.device, sizeof(res.device), "%s", prop.name);

  /* sieving primes <= sqrt(top) */
  u64 root = 1;
  while ((root + 1) <= top / (root + 1)) root++;
  u64 np_ = 0;
  u64* pl = simple_primes(root, &np_);
  if (!pl) return res;
  u32* prime32 = (u32*)malloc(sizeof(u32) * (size_t)(np_ ? np_ : 1));
  unsigned long long* magic = (unsigned long long*)malloc(sizeof(unsigned long long) * (size_t)(np_ ? np_ : 1));
  if (!prime32 || !magic) { free(pl); free(prime32); free(magic); return res; }
  for (u64 i = 0; i < np_; i++) {
    prime32[i] = (u32)pl[i];
    magic[i] = ~0ull / (u64)pl[i];   /* floor((2^64-1)/p) = floor(2^64/p) here */
  }
  free(pl);
  /* phase split threshold: primes below this keep the cooperative q-stride;
     T = GPU_BLOCK_VALS / 256 lanes = 1920 (larger primes go one-per-lane) */
  const u32 PHASE1_P = GPU_BLOCK_VALS / 256u;
  u32 p1idx = 0;
  while (p1idx < (u32)np_ && prime32[p1idx] < PHASE1_P) p1idx++;

  const u64 nblocks = (top + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS;

  u32* dP = 0;
  unsigned long long* dM = 0;
  unsigned long long* dC = 0;
  if (cudaMalloc(&dP, np_ * 4) != cudaSuccess) { free(prime32); free(magic); return res; }
  if (cudaMalloc(&dM, np_ * 8) != cudaSuccess) {
    cudaFree(dP); free(prime32); free(magic); return res;
  }
  if (cudaMalloc(&dC, nblocks * 8) != cudaSuccess) {
    cudaFree(dP); cudaFree(dM); free(prime32); free(magic); return res;
  }
  cudaMemcpy(dP, prime32, np_ * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dM, magic, np_ * 8, cudaMemcpyHostToDevice);
  free(prime32); free(magic);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0); cudaEventCreate(&ev1);

  /* optional single-block dump for byte-diff debugging (FASTSIEVE_DUMP_BLOCK=k) */
  unsigned char* dDump = 0;
  unsigned int dumpBlock = 0;
  unsigned char* hDump = 0;
  const char* denv = getenv("FASTSIEVE_DUMP_BLOCK");
  if (denv && *denv) {
    dumpBlock = (unsigned int)atoi(denv);
    hDump = (unsigned char*)malloc(SEG_BYTES);
    cudaMalloc(&dDump, SEG_BYTES);
  }

  cudaEventRecord(ev0);
  gsieve<<<(unsigned)nblocks, 256>>>(dP, dM, (u32)np_, p1idx,
                                     (unsigned long long)top, dC, dDump, dumpBlock);
  cudaEventRecord(ev1);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaEventSynchronize(ev1);
  float ms = 0;
  if (err == cudaSuccess && secs_out) {
    cudaEventElapsedTime(&ms, ev0, ev1);
    *secs_out = ms / 1000.0;
  }
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);

  if (err != cudaSuccess || !block_counts) {
    cudaFree(dP); cudaFree(dM); cudaFree(dC);
    return res;
  }
  err = cudaMemcpy(block_counts, dC, nblocks * 8, cudaMemcpyDeviceToHost);
  if (hDump) {
    cudaMemcpy(hDump, dDump, SEG_BYTES, cudaMemcpyDeviceToHost);
    const char* fn = "/tmp/gpublock.bin";
    FILE* f = fopen(fn, "wb");
    if (f) { fwrite(hDump, 1, SEG_BYTES, f); fclose(f); }
    fprintf(stderr, "[dump] block %u -> %s, kernel count = %llu\n",
            dumpBlock, fn, (unsigned long long)block_counts[dumpBlock]);
    cudaFree(dDump); free(hDump);
  }
  cudaFree(dP); cudaFree(dM); cudaFree(dC);
  if (err != cudaSuccess) return res;

  res.ok = 1;
  res.nblocks = nblocks;
  res.gpu_secs = ms / 1000.0;
  return res;
}

extern "C" GpuResult gpu_count(uint64_t top) {
  u64* bc = (u64*)calloc((size_t)((top + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS), 8);
  double secs = 0;
  GpuResult r = gpu_sieve(top, bc, &secs);
  if (r.ok) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < r.nblocks; i++) s += bc[i];
    if (top >= 5) s += 3; else if (top >= 3) s += 2; else if (top >= 2) s += 1;
    r.gpu_pi = s;
  }
  free(bc);
  return r;
}
