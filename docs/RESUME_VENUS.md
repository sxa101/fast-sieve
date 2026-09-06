# RESUME on `venus` – full session context dump

This file is a self-contained handoff so a fresh agent session on the CUDA
reference host **venus** (RTX 3090 + RTX 3080) can resume `fastsieve` without
repeating weeks of archaeology. Copy the repo to venus, read this file, then
give the agent the "System prompt" below verbatim.

---

## 1. Project one-liner

`fastsieve` = a correct, from-scratch segmented sieve of Eratosthenes (CPU
wheel-30 engine, exact to 10^12, plus an audited OpenCL GPU accelerator) that
counts `π(n)`. Comparable in speed to primesieve but ~1.5–1.7× behind its
unrolled-instantiable loops; the GPU path is the researched headroom.

Repo layout:

```
fastsieve.c   CPU engine (wheel-30, pre-sieve, 3 prime classes, OpenMP slices)
gpu.c/.h      OpenCL GPU accelerator + audit hooks
gpusieve.c    standalone OpenCL demo/diagnostic (dev only)
tests.ps1     system suite (oracle of exact π(n))
build.bat/tests.bat, README.md, LICENSE, docs/{DESIGN,GPU,WHEEL210,RESUME_VENUS}.md
```

## 2. Environment facts

* **Dev box (this repo's origin):** Windows 10/11, MSVC 2022, AMD Ryzen 5 5600G
  (12 threads, Zen3, AVX2, L1D 32K, L2 512K, L3 16MB) + **AMD Radeon RX 9070 XT**
  (`gfx1201`, RDNA4) via **OpenCL** only (no CUDA on AMD). OpenCL headers from
  `KhronosGroup/OpenCL-Headers`; kernel loads `OpenCL.dll` dynamically.
* **venus:** Linux, **CUDA runtime**, **NVIDIA RTX 3090 and RTX 3080**.
  → `nvcc`, no OpenCL needed; port `gpu.c` to CUDA (see §7).
* Reference oracle available on both: `primesieve` CLI (build from
  kimwalisch/primesieve, use `-t1`/`-t12 --no-status N` for π and `lo hi` for
  ranges). Always diff against it.

## 3. Do NOT relive these traps (all discovered the hard way)

1. **Residue-1 offset:** a value `v ≡ 1 (mod 30)` is stored at *offset 31* of
   byte `(v-6)/30` – one byte EARLIER than `v/30`. Any code computing a byte
   index as `(v - low)/30` clears the wrong byte for these values
   (deterministic ±1/block). Correct: `idx = (v - low)/30; if (v%30==1) idx--;`
   (or subtract the true offset).
2. **`+6` convention / trailing byte:** byte j covers
   `[low+30j+7, low+30j+31]`; a prime must be added to the segment that
   *stores* its `p²`, which may be the trailing `low+30B+1` candidate of the
   *previous* segment (boundary primes `786431²`, `786433²`). Add primes while
   `p² ≤ low + 30·B + 1`. Only the true first segment restores small primes
   (`PRIMEBITS` on bytes 0–7, guarded by `lo==0`).
3. **Frame carry when a step ≥ B:** a crossing step can jump whole segments;
   the carry index is relative to the *next* segment and must be advanced
   **exactly one segment per call**. Never `while (i>=B) i-=B` (clears one
   frame too early) – use `if (i>=B) { P[i].i = i-B; continue; }`.
4. **Slices:** an interior slice starts **one segment earlier** (`segStart =
   lo - span`) and only *counts* candidates ≥ `lo`, so its crossing state equals
   the continuous run and the `lo+1` trailing candidate is counted exactly once
   (`cap = hi+1`, last slice `cap = n`).
5. **First-byte slots:** candidates begin at value 7 (value 1 has no slot);
   primes 2,3,5 are added out-of-band (`+3`). Counting uses
   `nbits = (rel/30)*8 + CUNIT[rel%30] - 1`.
6. **GPU races:** clearing shared bytes across lanes is race-free only with
   word-level local **atomics** (`atomic_and` on `__local uint`, building a
   per-byte word mask). Residual ~±0.4/block still exists on the RDNA4 OpenCL
   blob after the residue-1 fix – the exact remaining root cause is an open
   TODO (see §6).

## 4. Correctness verification workflow (used throughout)

1. Build with MSVC (`build.bat`) or gcc/clang on venus.
2. Sweep: `100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12` at
   `-t 1` and `-t 12`, compare `pi()` to primesieve.
3. Boundary primes: `618473717761` (= 786431², π=23688293324) and
   `618476863489` (= 786433², π=23688409284).
4. `powershell -ExecutionPolicy Bypass .\tests.ps1` (or a bash port on venus) –
   oracle table lives in `tests.ps1`.
5. When hunting a cross bug: build a naive bit-reference for one segment and
   byte-diff, then bisect to the exact differing candidate; deterministic =
   logic bug, but the AMD residual was *not* eliminated that way on OpenCL.

## 5. Measured state of the world (dev box)

| n     | CPU 1 thread | CPU 12 threads | primesieve 1 | primesieve 12 |
|-------|-------------:|---------------:|-------------:|--------------:|
| 1e9   | 0.17 s       | 0.05 s         | 0.11 s       | 0.04 s        |
| 1e10  | 1.9 s        | 0.36 s         | 1.1 s        | 0.25 s        |
| 1e12  | 290 s        | 52 s           | 167 s        | 31 s          |

GPU (OpenCL, RX 9070 XT, unreliably-exact proto): 1e8 in ≈7 ms
(~14–16 Gcandidates/s). Reference-class GPU sieves: CUDASieve GTX1080 counted
1e12 in 12.5 s → modern cards have several× more headroom; our CUDA port on
venus should target ≈4–10× the 12-thread CPU here.

## 6. GPU open TODOs (in priority order for venus)

1. **CUDA port of `gpu.c` kernel** (translate OpenCL C → CUDA C, threads
   256/block, `__shared__ seg[16384]`, `atomicAnd` on shared `uint`, popcount
   via `__popcll`, one block per `30·16384` sub-range). CUDA's well-defined
   shared semantics + `__syncthreads` should make the kernel bit-exact → the
   audit passes and full speed is used. Verify with §4 sweep + audit.
2. **Nail the remaining OpenCL/RDNA4 residual** (~±0.4/block, deterministic):
   audit currently catches & falls back. Best lead: verify each block content
   against the CPU reference for the FIRST differing block (the `--blockdump`
   idea) rather than patching blind.
3. **GPU pre-sieve (AND pattern init)** to cut work an extra ~2×.
4. **Re-land wheel-210** on CPU (see `docs/WHEEL210.md` checklist) now that the
   residue-1/offset-31 and frame-carry traps are documented.

## 7. venus/CUDA quickstart

```
git clone <repo> fastsieve && cd fastsieve
# CPU:  gcc -O3 -march=native -fopenmp fastsieve.c gpu.c -o fastsieve_cpu
# CUDA: nvcc -O3 -arch=sm_86 gpu_cuda.cu fastsieve.c -o fastsieve
# (add gpu_cuda.c + a --gpu dispatch; keep the audit-fallback logic in fastsieve.c)
# sanity:
./fastsieve_cpu 1e10 && ./fastsieve_cpu -t 12 1e12
./fastsieve --gpu 1e11          # expect audit to pass on Turing/Ampere
```

---

## System prompt (paste to the venus agent)

```
You are resuming the "fastsieve" project. Read (in order):
  README.md, docs/DESIGN.md, docs/GPU.md, docs/WHEEL210.md, docs/RESUME_VENUS.md.
Mission (in priority order):
 1) Build the CPU engine on this Linux/CUDA host (gcc -O3 -march=native -fopenmp),
    verify pi() exactly matches primesieve for 1e8..1e12 at -t1 and -t12, incl.
    boundary primes 618473717761 and 618476863489.
 2) Port the OpenCL GPU kernel in gpu.c to CUDA (RTX 3090/3080, sm_86), keeping
    the audit-and-fallback design: GPU per-block counts must match the exact CPU
    engine on ~512 sampled segments + first/last, else fall back to CPU. Target:
    GPU exact on CUDA (shared-memory + atomicAnd + __syncthreads), measure and
    benchmark vs primesieve -tN.
 3) Fix the recorded OpenCL/RDNA4 deterministic residual (~±0.4/block) if time
    allows, and/or re-land wheel-210 per docs/WHEEL210.md.
 4) Update docs/RESUME_VENUS.md and the benchmark table with venus results,
    then commit + push to the public repo.

Rules: never claim exactness without the byte/oracle check described in
docs/RESUME_VENUS.md §4; if a GPU result equals the CPU oracle in the sweep,
say so; otherwise the audit MUST fall back and you must report it. Keep the
exact π(n) contract of the tool (printed value must always be exact).
```