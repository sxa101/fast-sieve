# GPU accelerator (`--gpu`)

## Implementation (`gpu.c` / `gpu_cuda.cu`)

One work-group/block (256 lanes) per **GPU block** of `30 * 16384 = 491520`
numbers, sieved **locally**:

* the block lives in shared/local memory; candidate bits are cleared with a
  **word-level atomic AND** on the 32-bit words of the segment so lanes racing
  on the same byte never lose updates;
* crossing uses the same wheel-30 math as the CPU engine, including
  `residue 1 -> offset 31, one byte earlier`;
* counting = per-word popcount + group tree reduction -> one `ulong` per block.

Two backends:
* `gpu.c` – OpenCL (AMD/NVIDIA), loads `OpenCL.dll` dynamically;
* `gpu_cuda.cu` – CUDA port (verified bit-exact on RTX 3090 / RTX 3080 Ti).

## Correctness by construction

`fastsieve --gpu` never trusts the kernel blindly: it **audits** the per-block
counts against ~512 sampled exact CPU segments (first + last always included);
any mismatch -> transparent CPU fallback. The printed `pi(n)` is exact on every
machine; full GPU speed is used whenever the kernel is bit-exact.

## Status (2026-09-06/07, both hosts)

The deterministic residual that the audit used to catch was **root-caused as
logical block-boundary bugs**, not a race or a driver issue:

* the kernel `cup[]` table was `CUNIT[r] - 1` (phantom candidate `1`
  subtracted twice) - `nbits` short by one for `rem in [2,6]`;
* `rel` was clamped to `30*SEG_BYTES`, dropping exactly the trailing
  `segEnd+1` candidate of full blocks; the crossing side had the same holes
  (`v < segEnd` guard, `qmax = segEnd/p`, `pp >= segEnd` break).

Fixed in both kernels. **Result: audit passes with no fallback on both hosts** -
bit-exact on the RX 9070 XT (OpenCL, verified up to 1e12) and on the
RTX 3090 / 3080 Ti (CUDA); 24/24 sweep on both paths.

## Performance

### CUDA (RTX 3090, kernel-only time - Linux reference host)

The v3 **phase-split kernel** replaces the naive per-block prime scan:
division-free `q0`/`qmax` (reciprocal multiply + `q0 = p` shortcut), small primes
cross via the cooperative lane-strided loop, large primes one-per-lane with a
per-block binary search over the active set. It keeps the 16 KiB shared segment
and 6 CTA/SM occupancy.

| n     | CPU 12 threads | primesieve 12t | CUDA GPU (3090) | CUDA GPU (3080 Ti) |
|-------|---------------:|---------------:|----------------:|-------------------:|
| 1e9   | 0.05 s         | 0.02 s         | 0.016 s         | 0.017 s            |
| 1e10  | 0.42 s         | 0.40 s         | 0.16 s          | 0.17 s             |
| 1e11  | 4.7 s          | 10.5 s         | 1.6 s           | 1.7 s              |
| 1e12  | 218 s          | 163 s          | **17.9 s**      | 18.2 s             |

(Reference-host CPU/primesieve columns are min-of-N on a shared box, ±~20%;
CPU column post-L1-chunk-fix.) **The CUDA GPU wins everywhere** - 1e12 in
17.9 s (47x the initial port, 9x faster than 12-thread primesieve, 12x faster
than that host's CPU). Bit-exact: full
audit 0 mismatches, `./tests.sh ./fastsieve --gpu` 24/24, boundary primes exact.
Full design and rejected alternatives in
[docs/GPU_DEVELOPMENT.md §9](GPU_DEVELOPMENT.md).

### OpenCL (RX 9070 XT, kernel-only time - dev box, re-measured 2026-09-07)

Now the **v3 phase-split kernel** (reciprocal-multiply division, small primes
cooperative / large one-per-lane): exact (audit passes, no fallback - verified
up to 1e12). CPU columns are the post-L1-chunk-fix dev-box numbers (best-of-2).

| n     | CPU 12 threads | primesieve 12t | GPU kernel (OpenCL, gfx1201) |
|-------|---------------:|---------------:|-----------------------------:|
| 1e8   | 0.020 s        | 0.012 s        | 0.006 s                      |
| 1e9   | 0.048 s        | 0.032 s        | 0.044 s                      |
| 1e10  | 0.31 s         | 0.229 s        | 0.35 s                       |
| 1e11  | 3.5 s          | 2.49 s         | 2.53 s                       |
| 1e12  | 47 s           | 30.0 s         | 26.3 s                       |

The GPU beats the CPU engine from ~1e11 up on this box (the only win-for-CPU
cell is 1e10, a 0.35 s vs 0.31 s photo-finish). The 1e12 kernel went
**715.5 s -> 26.3 s** when the v1 kernel was replaced by
the v3 phase split (≈27x); CUDA on the 3090 is still 1.5x faster (17.9 s).
For reference, v1 (naive) timings were: 1e9 0.063, 1e10 1.11, 1e11 26.2 s.

Future exercise (Linux-only): a **ROCm/HIP** backend that recompiles the CUDA
kernel `gpu_cuda.cu` nearly verbatim (it is already exact and tuned); not
buildable here because the RX 9070 XT is only reachable from Windows, where
ROCm is not available.

## Linux/CUDA build

```
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp -march=native -O3" \
     fastsieve.c gpu_cuda.cu -o fastsieve
FASTSIEVE_DEVICE=0 ./fastsieve --gpu 1e11
```