/* fastsieve.h - C API for the fastsieve prime-counting engine.
 *
 * The engine (fastsieve.c) counts pi(n) with a segmented wheel-30 sieve;
 * `fastsieve --gpu` adds a CUDA/OpenCL accelerator whose result is audited
 * against the exact CPU engine, with automatic fallback, so every function
 * here returns the EXACT answer on any machine that runs the CPU path.
 *
 * Linking: compile fastsieve.c together with your program, e.g.
 *   cl /O2 /arch:AVX2 /Oi /openmp fastsieve.c your_program.c
 * (add gpu.c for the GPU path).  Or:
 *   cl /O2 /arch:AVX2 /Oi /openmp /FASTLINK fastsieve.c examples/api_demo.c
 *
 * Thread-safety: the engine builds its lookup tables lazily.  Call
 * fastsieve_init() once (serial) before first use when you plan to call the
 * API from multiple threads; otherwise a short single-threaded init is done
 * automatically per process.  Each individual call is otherwise independent.
 */
#ifndef FASTSIEVE_H
#define FASTSIEVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FASTSIEVE_MAX_N ((uint64_t)8e15)   /* largest supported upper bound */

/* Errors (negative return values). */
enum {
  FASTSIEVE_OK        = 0,
  FASTSIEVE_ERR_RANGE = -1,   /* argument outside the supported range */
  FASTSIEVE_ERR_ARGS  = -2,   /* bad callback / null pointer / inconsistent lo>hi */
  FASTSIEVE_ERR_NOMEM = -3
};

typedef struct fastsieve_config {
  int         threads;       /* CPU threads; <= 0 selects the engine default (1).   */
  int         use_gpu;       /* nonzero: try the CUDA/OpenCL accelerator (exact,    */
                             /* audited, CPU fallback on mismatch/unavailable).     */
  uint64_t    sieve_bytes;   /* segment size in bytes (rounded down to a power of   */
                             /* two); 0 selects the engine default (256 KiB).       */
  double      med_factor;    /* medium/big prime crossover as a multiple of the     */
                             /* segment; <= 0 selects the engine default (3.0).     */
} fastsieve_config;

/* Prime output callback for fastsieve_generate.  Return nonzero to stop the
 * scan early; fastsieve_generate then returns the number of primes emitted so
 * far.  Primes are delivered in strictly ascending order. */
typedef int (*fastsieve_prime_cb)(uint64_t prime, void* user);

/* Idempotent engine initialization (also happens lazily on first call). */
void            fastsieve_init(void);

/* Version string, e.g. "1.0.0". */
const char*     fastsieve_version(void);

/* Number of primes p with lo <= p <= hi, or a negative FASTSIEVE_ERR_*.
 * lo == 2 uses the fastest available engine (GPU + OpenMP); lo > 2 uses the
 * single-threaded CPU generator (parallelizable yourself via fastsieve_generate
 * over disjoint sub-ranges).  hi must be <= FASTSIEVE_MAX_N. */
int64_t         fastsieve_count(uint64_t lo, uint64_t hi,
                                const fastsieve_config* cfg);

/* pi(n): number of primes <= n.  Equivalent to fastsieve_count(2, n, cfg). */
int64_t         fastsieve_pi(uint64_t n, const fastsieve_config* cfg);

/* Returns 1 if n is prime, 0 otherwise, negative FASTSIEVE_ERR_* on bad n. */
int             fastsieve_isprime(uint64_t n, const fastsieve_config* cfg);

/* k-th (1-based) prime >= start, or a negative FASTSIEVE_ERR_*.  k must be
 * >= 1.  Exact; computed by bracketing + binary search over fastsieve_pi
 * (a handful of full-range counts). */
int64_t         fastsieve_nth_prime(uint64_t k, uint64_t start,
                                    const fastsieve_config* cfg);

/* Generator: calls cb for every prime in [lo, hi] in ascending order.
 * Returns the number of primes emitted (positive), or a negative error.
 * When cb stops the scan early the returned count is the emitted-so-far. */
int64_t         fastsieve_generate(uint64_t lo, uint64_t hi,
                                   fastsieve_prime_cb cb, void* user,
                                   const fastsieve_config* cfg);

#ifdef __cplusplus
}
#endif

#endif /* FASTSIEVE_H */