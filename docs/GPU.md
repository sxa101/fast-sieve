# GPU accelerator (`--gpu`)

## Implementation (`gpu.c` / `gpu_cuda.cu`)

One work-group/block (256 lanes) per **GPU block** of `30Ã‚Â·16384 = 491520`
numbers, sieved **locally**:

* the block lives in shared/local memory; candidate bits are cleared with a
  **word-level atomic AND** on the 32-bit words of the segment so lanes racing
  on the same byte never lose updates;
* crossing uses the same wheel-30 math as the CPU engine, including
  `residue 1 Ã¢â€ â€™ offset 31, one byte earlier`;
* counting = per-word popcount + group tree reduction Ã¢â€ â€™ one `ulong` per block.

Two backends:
* `gpu.c` Ã¢â‚¬â€œ OpenCL (AMD/NVIDIA), loads `OpenCL.dll` dynamically;
* `gpu_cuda.cu` Ã¢â‚¬â€œ CUDA port (verified bit-exact on RTX 3090 / RTX 3080 Ti).

## Correctness by construction

`fastsieve --gpu` never trusts the kernel blindly: it **audits** the per-block
counts against ~512 sampled exact CPU segments (first + last always included);
any mismatch Ã¢â€¡â€™ transparent CPU fallback. The printed `Ãâ‚¬(n)` is exact on every
machine; full GPU speed is used whenever the kernel is bit-exact.

## Status (2026-09-06/07, both hosts)

The deterministic residual that the audit used to catch (~Ã‚Â±1/per boundary
prime) was **root-caused as logical block-boundary bugs**, not a race or a
driver issue:

* the kernel `cup[]` table was `CUNIT[r] Ã¢Ë†â€™ 1` (phantom candidate `1`
  subtracted twice) Ã¢â‚¬â€ `nbits` short by one for `rem Ã¢Ë†Ë† [2,6]`;
* `rel` was clamped to `30Ã‚Â·SEG_BYTES`, dropping exactly the trailing
  `segEnd+1` candidate of full blocks; the crossing side had the same holes
  (`v < segEnd` guard, `qmax = segEnd/p`, `pp >= segEnd` break).

Fixed in both kernels. **Result: audit passes with no fallback on both hosts** Ã¢â‚¬â€
bit-exact on the RX 9070 XT (OpenCL) and on the RTX 3090 / 3080 Ti (CUDA),
24/24 sweep on both paths.

## Performance

### CUDA (RTX 3090 / 3080 Ti, kernel-only time Ã¢â‚¬â€ venus)

The v3 **phase-split kernel** (`gpu_cuda.cu`) replaces the naive per-block
prime scan: division-free `q0`/`qmax` (reciprocal multiply + `q0 = p` shortcut),
small primes cross via the cooperative lane-strided loop, large primes
one-per-lane with a per-block binary search over the active set. It keeps the
16 KiB shared segment and 6 CTA/SM occupancy.

| n     | CPU 12t (venus) | primesieve 12t | CUDA GPU kernel (3090) | (3080 Ti) |
|-------|----------------:|----------------:|-----------------------:|----------:|
| 1e9   | 0.07 s          | 0.02 s          | 0.016 s                | 0.017 s   |
| 1e10  | 0.50 s          | 0.29 s          | 0.16 s                 | 0.17 s    |
| 1e11  | Ã¢â€°Ë†4 s            | Ã¢â€°Ë†2.6 s          | 1.6 s                  | 1.7 s     |
| 1e12  | 257 s           | 163 s           | **17.9 s**             | 18.2 s    |

**The CUDA GPU wins everywhere** Ã¢â‚¬â€ 1e12 in 17.9 s (47Ãƒâ€” the initial port, 9Ãƒâ€”
faster than 12-thread primesieve, 14Ãƒâ€” faster than our CPU). Bit-exact: full
audit 0 mismatches, `./tests.sh ./fastsieve --gpu` 24/24, boundary primes exact.
Full design + rejected alternatives in [RESUME_VENUS.md Ã‚Â§9](RESUME_VENUS.md).

### OpenCL (RX 9070 XT, kernel-only time Ã¢â‚¬â€ dev box)

Still the v1-style kernel (no v3 optimizations yet): exact (audit passes, no
fallback), but only beats the CPU below the crossover Ã¢â€°Ë† 5Ãƒâ€”10Ã¢ÂÂ¸.

| n     | CPU 12 threads | GPU kernel (OpenCL, gfx1201) |
|-------|---------------:|-----------------------------:|
| 1e8   | 0.038 s        | 0.007 s                      |
| 1e9   | 0.056 s        | 0.066 s                      |
| 1e10  | 0.38 s         | 1.30 s                       |
| 1e11  | 3.76 s         | 26.1 s                       |

Open work item: port the v3 phase-split optimizations (reciprocal-multiply,
phase-split prime classes, pre-sieve) to `gpu.c`/OpenCL and re-benchmark.

## venus/CUDA build

```
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp -march=native -O3" \
     fastsieve.c gpu_cuda.cu -o fastsieve
FASTSIEVE_DEVICE=0 ./fastsieve --gpu 1e11
```