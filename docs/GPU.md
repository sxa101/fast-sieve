# GPU accelerator (`--gpu`)

## Implementation (`gpu.c` / `gpu_cuda.cu`)

One work-group/block (256 lanes) per **GPU block** of `30·16384 = 491520`
numbers, sieved **locally**:

* the block lives in shared/local memory; candidate bits are cleared with a
  **word-level atomic AND** on the 32-bit words of the segment so lanes racing
  on the same byte never lose updates;
* crossing uses the same wheel-30 math as the CPU engine, including
  `residue 1 → offset 31, one byte earlier`;
* counting = per-word popcount + group tree reduction → one `ulong` per block.

Two backends:
* `gpu.c` – OpenCL (AMD/NVIDIA), loads `OpenCL.dll` dynamically;
* `gpu_cuda.cu` – CUDA port (verified bit-exact on RTX 3090 / RTX 3080 Ti).

## Correctness by construction

`fastsieve --gpu` never trusts the kernel blindly: it **audits** the per-block
counts against ~512 sampled exact CPU segments (first + last always included);
any mismatch ⇒ transparent CPU fallback. The printed `π(n)` is exact on every
machine; full GPU speed is used whenever the kernel is bit-exact.

## Status (2026-09-06, both hosts)

The deterministic residual that the audit used to catch (~±1/per boundary
prime) was **root-caused as logical block-boundary bugs**, not a race or a
driver issue:

* the kernel `cup[]` table was `CUNIT[r] − 1` (phantom candidate `1`
  subtracted twice) — `nbits` short by one for `rem ∈ [2,6]`;
* `rel` was clamped to `30·SEG_BYTES`, dropping exactly the trailing
  `segEnd+1` candidate of full blocks; the crossing side had the same holes
  (`v < segEnd` guard, `qmax = segEnd/p`, `pp >= segEnd` break).

Fixed in both kernels. **Result: the audit now passes on this RX 9070 XT
(gfx1201, OpenCL) with no fallback, and 24/24 sweep is bit-exact on venus
(CUDA).**

## Performance (dev box, AMD Ryzen 5 5600G 12-thread, Radeon RX 9070 XT)

| n     | CPU 12 threads | GPU kernel (OpenCL, gfx1201) | wall (incl. audit) |
|-------|---------------:|-----------------------------:|-------------------:|
| 1e8   | 0.038 s        | 0.007 s                      | 0.03 s             |
| 1e9   | 0.056 s        | 0.066 s                      | 0.08 s             |
| 1e10  | 0.38 s         | 1.30 s                       | 1.5 s              |
| 1e11  | 3.76 s         | 26.1 s                       | 27 s               |

**Crossover ≈ 5×10⁸.** Below it the GPU wins by ~5×; above it, the current
(theoretically naive) kernel is progressively slower: per-block work grows as
`π(√segEnd)` (a per-prime serial scan with two 64-bit integer divisions each)
while the block's 256-wide parallelism does not grow. venus/CUDA shows the
same shape (kernel 1e12 = 836 s vs 257 s CPU-12t).

The kernel is **correctness-first and naive** — it intentionally omits every
CPU-engine trick. The optimization plan for the venus session is in
[RESUME_VENUS.md → §9](RESUME_VENUS.md).

## venus/CUDA build

```
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp -march=native -O3" \
     fastsieve.c gpu_cuda.cu -o fastsieve
FASTSIEVE_DEVICE=0 ./fastsieve --gpu 1e11
```