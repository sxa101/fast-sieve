# GPU development & CUDA port notes – full session context dump

This file is a self-contained handoff so a fresh agent session on the CUDA
reference host (**RTX 3090** + **RTX 3080**) can resume `fastsieve` without
repeating weeks of archaeology. Copy the repo to that host, read this file, then
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
build.bat/tests.bat, README.md, LICENSE, docs/{DESIGN,GPU,GPU_DEVELOPMENT,WHEEL210}.md
```

## 2. Environment facts

* **Dev box (this repo's origin):** Windows 10/11, MSVC 2022, AMD Ryzen 5 5600G
  (12 threads, Zen3, AVX2, L1D 32K, L2 512K, L3 16MB) + **AMD Radeon RX 9070 XT**
  (`gfx1201`, RDNA4) via **OpenCL** only (no CUDA on AMD). OpenCL headers from
  `KhronosGroup/OpenCL-Headers`; kernel loads `OpenCL.dll` dynamically.
* **CUDA reference host:** Linux, **CUDA runtime**, **NVIDIA RTX 3090 and RTX 3080**.
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

1. Build with MSVC (`build.bat`) or gcc/clang on the reference host.
2. Sweep: `100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12` at
   `-t 1` and `-t 12`, compare `pi()` to primesieve.
3. Boundary primes: `618473717761` (= 786431², π=23688293324) and
   `618476863489` (= 786433², π=23688409284).
4. `powershell -ExecutionPolicy Bypass .\tests.ps1` (or a bash port on Linux) –
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

**AMD GPU (OpenCL, RX 9070 XT, kernel-only, after the 2026-09-06 boundary-bug
fix):** crosscheck

| n     | CPU 12t | GPU kernel (gfx1201) |
|-------|--------:|---------------------:|
| 1e8   | 0.038 s | 0.007 s             |
| 1e9   | 0.056 s | 0.066 s             |
| 1e10  | 0.38 s  | 1.30 s              |
| 1e11  | 3.76 s  | 26.1 s              |

Crossover ≈ 5×10⁸ (GPU faster below, slower above). Audit passes everywhere →
the fixed OpenCL kernel is bit-exact on RDNA4 with no fallback.

GPU (OpenCL, RX 9070 XT, unreliably-exact proto): 1e8 in ≈7 ms
(~14–16 Gcandidates/s). Reference-class GPU sieves: CUDASieve GTX1080 counted
1e12 in 12.5 s → modern cards have several× more headroom.

## 6. GPU open TODOs (status after the 2026-09-06 CUDA session)

1. ✅ **CUDA port of `gpu.c` kernel — DONE, bit-exact.** `gpu_cuda.cu`
   (256 threads/block, `__shared__` 16 KiB segment, word-level `atomicAnd`,
   `__popcll` counting, one block per 30·16384 sub-range). Audit passes with
   **0 mismatches** on RTX 3090 and RTX 3080 Ti; full sweep (1e2..1e12, -t1/-t12,
   incl. boundary primes 786431²/786433²) is **24/24 exact** via
   `./tests.sh ./fastsieve --gpu`.
2. ✅ **RDNA4 OpenCL residual — ROOT-CAUSED (not a race, not the driver).**
   Two *logical* block-boundary bugs, now fixed in BOTH `gpu.c` and
   `gpu_cuda.cu`:
   - the kernel's `cup[]` table was `CUNIT[r] − 1` (the phantom candidate `1`
     was subtracted twice): `nbits` came out one short for `rem ∈ [2,6]`;
   - `if (rel > 30·SEG_BYTES) rel = 30·SEG_BYTES` clamped away exactly the
     `cap = segEnd + 2` case of every full block, so the trailing candidate
     `segEnd+1` (offset-31 bit of the last byte) was dropped from the count.
     The crossing side had the same holes (`v < segEnd` guard,
     `qmax = segEnd/p`, `pp >= segEnd` break) — composites/primes equal to
     `segEnd+1` were silently skipped.
   Together this produced the deterministic ~±1-per-boundary-prime residual
   that the audit kept catching. `fastsieve.c`'s audit window now uses the same
   cap convention (`segHi+2` when `segHi < n`). ✅ Dev box re-ran it
   (2026-09-06b): OpenCL bit-exact on the RX 9070 XT, no fallback.
3. ✅ **CUDA kernel performance — DONE (CUDA round 2, §9).** Division-free
   reciprocal multiply + `q0 = p` shortcut + v3 phase-split crossing
   (small primes cooperative / large primes one-per-lane). 1e12: 836 s →
   **17.9 s** on the 3090 (18.2 s on the 3080 Ti) — beats 12-thread primesieve
   (163 s) and CPU-12t (257 s). Multi-block-CTA design measured and REJECTED
   (see §9.2 B). Remaining micro-ideas in §9.2 E.
4. ⬜ **GPU pre-sieve (AND pattern init)** — open, LOW priority: at v3 speeds
   the AND-pattern init costs L2 bandwidth of the same order as the crossings
   it saves (see §9.2 D).
5. ⬜ **Re-land wheel-210** on CPU (see `docs/WHEEL210.md` checklist) — still
   reverted, still honest.

## 7. Linux/CUDA quickstart (verified verbatim)

```
# oracle
git clone -q --depth 1 https://github.com/kimwalisch/primesieve.git /tmp/primesieve
cmake -S /tmp/primesieve -B /tmp/primesieve/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local && \
cmake --build /tmp/primesieve/build -j12 && cmake --install /tmp/primesieve/build
export PATH=$HOME/.local/bin:$PATH

# CPU
gcc -O3 -march=native -fopenmp fastsieve.c gpu.c -o fastsieve_cpu
# CUDA (device select via FASTSIEVE_DEVICE=0/1; debug block dump via
#       FASTSIEVE_DUMP_BLOCK=k -> /tmp/gpublock.bin)
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp -march=native -O3" \
     fastsieve.c gpu_cuda.cu -o fastsieve
# sanity + full sweep
./fastsieve_cpu 1e10 && ./fastsieve_cpu -t 12 1e12
./tests.sh ./fastsieve_cpu        # 24/24
./tests.sh ./fastsieve --gpu      # 24/24, audit 0 mismatches
```

---

## 8. CUDA session report (2026-09-06)

Host: Linux, i5-11600 (12t), gcc 13.3, CUDA 13.1, RTX 3080 Ti (CUDA dev 1) +
RTX 3090 (CUDA dev 0). Oracle: primesieve 12.16 built to `~/.local`.

**What was done**

* Portability: `fastsieve.c` + `gpu.c` now build on Linux/gcc (`_WIN32` guards,
  `clock_gettime` timing, `__builtin_popcountll`, OpenCL stub returns ok=0).
  MSVC/Windows path untouched.
* CUDA port: `gpu_cuda.cu` (see §6.1) — exact on both cards, audit passes
  everywhere.
* **Found and fixed the block-boundary logic bug family** (§6.2) in the GPU
  kernels and aligned `fastsieve.c`'s audit window with the cap convention.
  Debug tooling left in the repo: `FASTSIEVE_DUMP_BLOCK=k` (production-kernel
  block dump), `debugdump.cu`, `blockdiff.cu`, `blockcmp.c`.
* `tests.sh`: bash port of the test suite (24 checks, needs primesieve CLI).
* Full verification: CPU sweep 24/24, GPU sweep 24/24 (3090), spot-checked
  exact on 3080 Ti (1e8..1e11), boundary primes exact on both paths.

**CUDA reference-host benchmarks** (GPU columns = round-2 v3 kernel; round-1 numbers in §9.0)

| n     | ours CPU 1t | ours CPU 12t | psieve 1t | psieve 12t | ours GPU 3090 | ours GPU 3080 Ti |
|-------|------------:|-------------:|----------:|-----------:|--------------:|-----------------:|
| 1e9   | 0.21 s      | 0.07 s       | 0.10 s    | 0.02 s     | 0.016 s       | 0.017 s          |
| 1e10  | 2.4 s       | 0.50 s       | 1.2 s     | 0.29 s     | 0.16 s        | 0.17 s           |
| 1e11  | –           | –            | –         | –          | 1.64 s        | 1.69 s           |
| 1e12  | 510 s       | 257 s        | 267 s     | 163 s      | 17.9 s        | 18.2 s           |

All GPU numbers are **exact π values with audit passed** (kernel-only time via
CUDA events; wall time adds the audit's 512 CPU segments). Status: the GPU
kernel now beats 12-thread primesieve from ~1e10 upward.

---

## 9. CUDA round 2 — kernel optimization v2 (DESIGN → IMPLEMENTED 2026-09-06c)

### 9.0 Round-2 results (all exact, audit 0 mismatches, 24/24 gate)

| n     | v1+A (div-free) | v3 phase-split | v0 baseline | target |
|-------|----------------:|---------------:|------------:|-------:|
| 1e9   | 0.027 s         | 0.016 s        | 0.027 s     | –      |
| 1e10  | 0.493 s         | 0.162 s        | 0.93 s      | –      |
| 1e11  | ~15 s           | 1.64 s         | 29.4 s      | ≤3–4 s |
| 1e12  | ~440 s          | 17.9 s         | 836 s       | ≤150 s |

**Success criteria met with 5–8× margin.** 3080 Ti spot-check exact 1e8..1e12
(1e12 kernel 11.1 s). Boundary primes exact, audit clean, both cards.

### 9.1 Why the GPU was slow (measured, not guessed)

Per-block work = an **outer scan over all `π(√segEnd)` sieving primes**, each
with two **64-bit integer divisions** (`q0 = ceil(v0/p)`, `qmax = (segEnd+1)/p`)
plus a primality-unchecked `%30`/`/30` pair (those are constant-divisor, cheap),
and a **word-level shared `atomicAnd`** per multiple. The block only has
`491520` numbers, so for any prime `p > 491520/256 = 1920` fewer than 256
multiples exist and most lanes are idle; for `p` near `√n` the scan finds no
multiples at all but the outer iteration is still paid. Summed over
`n/491520` blocks this serial-scan + division overhead dominates:
`1e12 → 836 s (3090)` vs `257 s CPU-12t`.

### 9.2 What was done (ordered; each step gated on audit 0-mismatch + sweep exact)

**A. Remove the two runtime divisions — DONE, kept.** Host precomputes
`m_p = ⌊2^64 / p⌋`; kernel `q = __umul64hi(a, m); if (a − q·p ≥ p) q++;`
(proof: with `e = 2^64 − m·p < p`, mulhi = ⌊(a − a·e/2^64)/p⌋ ∈ {q−1, q}, so
exactly one correction is exact for all `a < 2^64`). Plus the `q0 = p`
shortcut when `p² ≥ segLow+7` (the common case; only early blocks take the
general ceil path). Measured: 1e10 0.93 → 0.493 s (1.9×), exact everywhere.

**B. Multi-block CTAs + chunked shared prime list — IMPLEMENTED, REJECTED on
measurement (keep the lesson).** `gsieve2` (K blocks/CTA, prime list streamed
through shared in 2048-prime chunks, per-block `done[]` flags) was exact but
**slower everywhere**: 1e9 0.081 s vs 0.027 s; 1e10 1.43 s vs 0.493 s; 1e11
32.8 s vs ~15 s. Root cause: K×16 KiB segments blow the shared budget
(sm_86 = 100 KiB/SM) → 1 CTA/SM at K=4 vs 6 CTAs/SM for v1 — occupancy
starvation costs more than the K× scan amortization saves. **Lesson: on this
kernel, never trade shared-memory footprint for work amortization.**

**B′. v3 phase-split — DONE, this is the shipped kernel.** Same 16 KiB/CTA
layout and occupancy as v1, but the prime scan is split at
`T = 491520/256 = 1920`:
* phase 1 (`p < 1920`, ~296 primes): v1's cooperative lane-strided q-loop —
  these primes have ≥256 multiples per block, so no lane idles;
* phase 2 (`1920 ≤ p ≤ √(segEnd+1)`): **one prime per lane** (`pi = p1idx +
  lid; pi < lo; pi += 256`) — the outer scan becomes 256× parallel and the
  lane-idle tail disappears; each lane crosses its primes' few multiples
  directly (same `atomicAnd` bit-clear helper `clear_mult`).
* per-block binary search (17 steps) bounds the active set: `lo` = first index
  with `p² > segEnd+1`.
Measured (3090, kernel-only): 1e9 0.016 s | 1e10 0.162 s | 1e11 1.64 s |
1e12 17.9 s (47× vs baseline; CPU-12t 257 s beaten by 14×).

**C. CUDASieve-style buckets — NOT NEEDED.** With B′ the outer scan is no
longer serial; profiling headroom went from ">20%" to irrelevant.

**D. GPU pre-sieve (AND-pattern init) — OPEN, low priority.** The 16 tables
(~22 KiB) live in global memory; a block's init would cost ~256 KiB of
coalesced L2 reads ≈ the same order as the crossings it saves (p ≤ 163 is
~25–30% of clears). Poor ROI at current speeds; revisit only if the kernel
becomes init-bound.

**E. Micro — partially done:** `__launch_bounds__(256, 6)` pins v1 occupancy;
`clear_mult` factors the bit-clear; `%30`/`/30` left to the compiler's constant
magic; `uint4` clears + occupancy tuning unexplored.

### 9.3 Success criteria — ALL MET

* audit **0 mismatches** on 3090 and 3080 Ti; `tests.sh ./fastsieve --gpu` 24/24 ✓
* kernel-only: 1e11 = 1.64 s (≤3–4 s ✓), 1e12 = 17.9 s (≤150 s ✓)
* tables updated (§8, README) and pushed ✓

### 9.4 Simplified pseudocode for the inner loop after A+B

```
kernel gsieve2(prim[], m[], np, K, ..., top, counters[])
  gid, K blocks at base = gid*K
  __shared__ u32 sp[NP];            // load primes once per CTA
  if (np) { for (i=lane; i<np; i+=256) sp[i]=prim[i]; }
  __syncthreads();
  for (k=0;k<K;k++){
    segLow = 30*SEG*(gid*K + k); segEnd=segLow+30*SEG;
    init seg[] = 0xFF (or pre-sieved pattern, D)
    __syncthreads();
    for (pi=0; pi<np; pi++){
      p=sp[pi]; if (p<7) continue;
      pp=(ulong)p*p; if (pp>segEnd+1) break;
      q0 = (pp>=segLow+7) ? p : div_by_mul(segLow+7+p-1, p, m[pi]);   // A
      qmax = div_by_mul(segEnd+1, p, m[pi]);                         // A
      for (q=q0+lane; q<=qmax; q+=256){ ...clear v=p*q as today... }
    }
    __syncthreads(); count -> counters[gid*K+k]; (restore PRIMEBITS for block0)
  }
```

---

## System prompt (paste to the next GPU agent)

```
You are resuming the "fastsieve" project. Read (in order):
  README.md, docs/DESIGN.md, docs/GPU.md, docs/WHEEL210.md, docs/GPU_DEVELOPMENT.md.
State: CPU engine exact (gcc build, 24/24 sweep incl. boundary primes); the v3
phase-split CUDA kernel (gpu_cuda.cu) is BIT-EXACT on RTX 3090/3080 Ti and
FAST: 1e11 kernel 1.64 s, 1e12 kernel 17.9 s (3090) - beats 12-thread
primesieve from ~1e10 up. The OpenCL/RDNA4 residual was root-caused as logical
block-boundary bugs and is fixed; OpenCL re-validated bit-exact on the dev box.
Do NOT re-try multi-block CTAs with K segments per CTA - measured and rejected
(§9.2 B: shared-memory footprint kills occupancy more than amortization pays).
Mission (in priority order):
 1) Re-land wheel-210 on the CPU engine per docs/WHEEL210.md (frame-carry
    semantics documented; residue-1/offset-31 traps in §3). Gate: 24/24 on
    ./tests.sh ./fastsieve_cpu AND ./tests.sh ./fastsieve --gpu unchanged,
    benchmark vs the wheel-30 numbers in §8.
 2) Optional: GPU pre-sieve AND-pattern init (§9.2 D) - only if a profile
    shows the kernel is init-bound; keep the audit + 24/24 gate.
 3) Update docs/GPU_DEVELOPMENT.md + benchmark tables with new results, then
    commit + push.

Rules: never claim exactness without the byte/oracle check in
docs/GPU_DEVELOPMENT.md §4; keep the audit-and-fallback design and the exact
π(n) contract (printed value must always be exact).
```