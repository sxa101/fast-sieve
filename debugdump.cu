/* debugdump.cu - dump GPU block counts and compare vs primesieve at boundaries */
#include "gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  uint64_t top = (argc > 1) ? strtoull(argv[1], NULL, 10) : 1000000000ULL;
  uint64_t nb = (top + GPU_BLOCK_VALS - 1) / GPU_BLOCK_VALS;
  uint64_t* bc = (uint64_t*)calloc((size_t)nb, 8);
  GpuResult r = gpu_sieve(top, bc, NULL);
  if (!r.ok) { fprintf(stderr, "gpu failed\n"); return 1; }
  fprintf(stderr, "device: %s nblocks: %llu\n", r.device, (unsigned long long)r.nblocks);
  /* cumulative GPU count after block k = candidates <= k*BLOCK+1 (block<last) */
  uint64_t cum = 0;
  for (uint64_t k = 0; k < r.nblocks; k++) {
    cum += bc[k];
    uint64_t boundary = (k + 1) * GPU_BLOCK_VALS + 1;
    if (k + 1 == r.nblocks) boundary = top; /* final block counts < top */
    if (argc > 2 || k < 24 || k + 6 > r.nblocks)
      printf("%llu %llu %llu\n", (unsigned long long)k, (unsigned long long)bc[k],
             (unsigned long long)cum);
  }
  return 0;
}
