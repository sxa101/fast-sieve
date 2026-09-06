/* gpu.h - OpenCL sieve-accelerator interface for fastsieve */
#ifndef GPU_H
#define GPU_H
#include <stdint.h>

#define GPU_SEG_BYTES 16384u          /* GPU local segment size (bytes) */
#define GPU_BLOCK_VALS ((uint64_t)30 * GPU_SEG_BYTES)

typedef struct {
  int       ok;         /* 1 if a GPU device executed the kernel */
  char      device[128];
  uint64_t  gpu_pi;     /* raw GPU count including 2,3,5 */
  double    gpu_secs;
  uint64_t  nblocks;    /* number of GPU blocks used */
} GpuResult;

#ifdef __cplusplus
extern "C" {
#endif

/* Run the OpenCL segmented sieve for [0, top).
   On failure ok = 0. On success fills block_counts[0..nblocks). */
GpuResult gpu_sieve(uint64_t top, uint64_t* block_counts, double* secs_out);

/* Aggregate convenience: gpu_sieve + sum. */
GpuResult gpu_count(uint64_t top);

#ifdef __cplusplus
}
#endif

#endif