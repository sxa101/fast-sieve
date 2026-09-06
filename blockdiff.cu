/* blockdiff.cu - dump one GPU block's segment bytes vs naive CPU reference */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define SEG_BYTES 16384u

__global__ void gsieve_dump(const unsigned int* prim, unsigned int np,
                            unsigned long long top, unsigned long long* counters,
                            unsigned char* dump, unsigned int dumpBlock) {
  const unsigned int gid = blockIdx.x;
  const unsigned int lid = threadIdx.x;
  const unsigned int L   = blockDim.x;
  const unsigned long long segLow = (unsigned long long)30 * SEG_BYTES * gid;
  const unsigned long long segEnd = segLow + (unsigned long long)30 * SEG_BYTES;
  __shared__ __align__(16) unsigned char seg[SEG_BYTES];
  for (unsigned int i = lid; i < SEG_BYTES; i += L) seg[i] = 0xFF;
  __syncthreads();
  for (unsigned int pi = 0; pi < np; pi++) {
    unsigned int p = prim[pi];
    if (p < 7) continue;
    unsigned long long pp = (unsigned long long)p * p;
    if (pp > segEnd + 1) break;
    unsigned long long v0 = (pp > segLow + 7) ? pp : segLow + 7;
    unsigned long long q0 = (v0 + p - 1) / p;
    unsigned long long qmax = (segEnd + 1) / p;
    for (unsigned long long q = q0 + lid; q <= qmax; q += L) {
      unsigned long long v = p * q;
      if (v >= segLow + 7 && v < segEnd + 2) {
        unsigned char r = (unsigned char)(v % 30);
        if (r==1||r==7||r==11||r==13||r==17||r==19||r==23||r==29) {
          unsigned long long idx = (v - segLow) / 30;
          if (r == 1) idx -= 1;
          if (idx < SEG_BYTES) {
            unsigned int wi = (unsigned int)idx >> 2;
            unsigned int by = (unsigned int)idx & 3;
            int b = (r==1)?7:(r==7)?0:(r==11)?1:(r==13)?2:(r==17)?3:(r==19)?4:(r==23)?5:6;
            unsigned int wordMask = ~(((unsigned int)1u << b) << (8*by));
            atomicAnd((unsigned int*)&seg[wi << 2], wordMask);
          }
        }
      }
    }
  }
  __syncthreads();
  if (gid == dumpBlock && lid == 0)
    for (unsigned int i = 0; i < SEG_BYTES; i++) dump[i] = seg[i];
  /* keep counter write so layout matches production kernel */
  __shared__ unsigned long long lsum[256];
  lsum[lid] = 0; __syncthreads();
  if (lid == 0) counters[gid] = lsum[0];
}

static int bitOf(unsigned char r) {
  switch (r) { case 7: return 0; case 11: return 1; case 13: return 2;
    case 17: return 3; case 19: return 4; case 23: return 5; case 29: return 6;
    case 1: return 7; }
  return -1;
}

int main(int argc, char** argv) {
  unsigned long long block = (argc > 1) ? strtoull(argv[1], NULL, 10) : 8ULL;
  unsigned long long top = (block + 1) * 30ull * SEG_BYTES + 2; /* non-final cap */
  unsigned int* dP; unsigned char* dSeg; unsigned long long* dC;
  /* sieving primes up to sqrt(segEnd+1) */
  unsigned long long root = 1;
  while ((root + 2) * (root + 2) <= 30ull * SEG_BYTES * (block + 1) + 1) root++;
  /* simple sieve for primes <= root */
  unsigned int np = 0;
  unsigned int* primes = (unsigned int*)malloc(sizeof(unsigned int) * (root + 16));
  char* comp = (char*)calloc((size_t)root + 2, 1);
  for (unsigned int i = 2; i <= root; i++) {
    if (!comp[i]) { primes[np++] = i; for (unsigned long long j = (unsigned long long)i * i; j <= root; j += i) comp[j] = 1; }
  }
  free(comp);
  cudaMalloc(&dP, np * 4); cudaMalloc(&dSeg, SEG_BYTES); cudaMalloc(&dC, 8);
  cudaMemcpy(dP, primes, np * 4, cudaMemcpyHostToDevice);
  unsigned char* seg = (unsigned char*)malloc(SEG_BYTES);
  gsieve_dump<<<(unsigned)(block + 1), 256>>>(dP, np, top, dC, dSeg, (unsigned)block);
  cudaDeviceSynchronize();
  cudaMemcpy(seg, dSeg, SEG_BYTES, cudaMemcpyDeviceToHost);

  /* naive CPU reference over the same value window [segLow+7, segEnd+1] */
  unsigned long long segLow = 30ull * SEG_BYTES * block;
  unsigned long long segEnd = segLow + 30ull * SEG_BYTES;
  char* isComp = (char*)calloc((size_t)(segEnd + 2), 1); /* index by absolute value */
  for (unsigned int i = 0; i < np; i++) {
    unsigned long long p = primes[i];
    if (p < 7) continue;
    for (unsigned long long v = p * p; v <= segEnd + 1; v += p) isComp[v] = 1;
  }
  int diffs = 0;
  for (unsigned int j = 0; j < SEG_BYTES; j++) {
    for (int b = 0; b < 8; b++) {
      unsigned long long v = segLow + 30ull * j + ((b == 7) ? 31ull : 0ull);
      /* bit order: bits 0..6 -> offsets 7,11,13,17,19,23,29; bit 7 -> offset 31 */
      static const int OFF[8] = {7,11,13,17,19,23,29,31};
      v = segLow + 30ull * j + (unsigned)OFF[b];
      if (v > segEnd + 1) continue;
      int gpuBit = (seg[j] >> b) & 1;
      int cpuBit = isComp[v] ? 0 : 1;
      int unit = (v % 30 == 1 || v % 30 == 7 || v % 30 == 11 || v % 30 == 13 ||
                  v % 30 == 17 || v % 30 == 19 || v % 30 == 23 || v % 30 == 29);
      if (!unit) continue;
      if (gpuBit != cpuBit) {
        printf("v=%llu r=%u byte=%u bit=%d gpu=%d cpu=%d %s\n", v, (unsigned)(v % 30),
               j, b, gpuBit, cpuBit, cpuBit ? "(prime cleared by GPU)" : "(composite kept by GPU)");
        diffs++;
      }
    }
  }
  printf("block %llu: %d bit diffs\n", block, diffs);
  return 0;
}
