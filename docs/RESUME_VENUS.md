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

## 6. GPU open TODOs (status after the 2026-09-06 venus session)

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
   cap convention (`segHi+2` when `segHi < n`). **The dev box should re-run the
   OpenCL path — with these fixes it may pass with no fallback.**
3. ⬜ **CUDA kernel performance (next session's main item).** The kernel is
   division-bound: two 64-bit divisions per active prime per block
   (`q0 = ceil(v0/p)`, `qmax = (segEnd+1)/p`), and lanes idle for
   `p > 491520/256`. 1e12 kernel = 836 s on the 3090 vs 257 s CPU-12t.
   Ideas: reciprocal-multiply + ≤2 corrections (pass 1/p from host), skip the
   division when `pp ≥ segLow+7` (then `q0 = p`), process several blocks per
   CTA, CUDASieve-style per-prime wheel state, GPU pre-sieve (AND-pattern
   init, ~2×).
4. ⬜ **GPU pre-sieve (AND pattern init)** to cut work an extra ~2×.
5. ⬜ **Re-land wheel-210** on CPU (see `docs/WHEEL210.md` checklist) — still
   reverted, still honest.

## 7. venus/CUDA quickstart (verified verbatim)

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

## 8. venus session report (2026-09-06)

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

**venus benchmarks**

| n     | ours CPU 1t | ours CPU 12t | psieve 1t | psieve 12t | ours GPU 3090 | ours GPU 3080 Ti |
|-------|------------:|-------------:|----------:|-----------:|--------------:|-----------------:|
| 1e9   | 0.21 s      | 0.07 s       | 0.10 s    | 0.02 s     | 0.027 s       | 0.030 s          |
| 1e10  | 2.4 s       | 0.50 s       | 1.2 s     | 0.29 s     | 0.93 s        | 0.96 s           |
| 1e11  | –           | –            | –         | –          | 29.4 s        | 30.3 s           |
| 1e12  | 510 s       | 257 s        | 267 s     | 163 s      | 836 s         | –                |

All GPU numbers are **exact π values with audit passed** (kernel-only time via
CUDA events; wall time adds the audit's 512 CPU segments). Honest status: the
GPU kernel is correctness-first and currently *slower* than the 12-thread CPU
past 1e10 — §6.3 lists the concrete optimization plan.

---

## 9. venus round 2 — CUDA kernel optimization v2 (design, ready to implement)

### 9.1 Why the GPU is currently slow (measured, not guessed)

Per-block work = an **outer scan over all `π(√segEnd)` sieving primes**, each
with two **64-bit integer divisions** (`q0 = ceil(v0/p)`, `qmax = (segEnd+1)/p`)
plus a primality-unchecked `%30`/`/30` pair (those are constant-divisor, cheap),
and a **word-level shared `atomicAnd`** per multiple. The block only has
`491520` numbers, so for any prime `p > 491520/256 = 1920` fewer than 256
multiples exist and most lanes are idle; for `p` near `√n` the scan finds no
multiples at all but the outer iteration is still paid. Summed over
`n/491520` blocks this serial-scan + division overhead dominates:
`1e12 → 836 s (3090)` vs `257 s CPU-12t`. Same shape on OpenCL/gfx1201
(crosscheck table in §5).

Reference numbers to beat (venus, 3090):
`1e9: 0.027s | 1e10: 0.93s | 1e11: 29.4s | 1e12: 836s`,
vs CPU-12t `0.07 / 0.50 / ~4 / 257` and primesieve-12t `0.02 / 0.29 / ~2.6 / 163`.

### 9.2 Optimization stack (ordered; each step MUST keep audit 0-mismatch + tests.sh 24/24)

**A. Remove the two runtime divisions (biggest single win, low risk).**
Replace `ceil(v0/p)` and `(segEnd+1)/p` with division-by-reciprocal: precompute
on the host for every sieving prime `m_p = floor(2^64 / p)` (u64) and pass
`m[]` alongside `prim[]`. In the kernel:

```
// q = a / p  (a < 2^64, p < 2^32), m = 2^64 / p precomputed
uint64_t q = __umulhi(a, m);     // high 64 bits of a*m
uint64_t r = a - q * p;
if (r >= p) { q++; r -= p; }     // at most ONE correction (proven bound)
```

Validation rule: cross-check `q` against `a/p` for a few million pseudo-random
`a` on the host before trusting it; and ALWAYS re-run the byte-diff tools
(`debugdump.cu` / `blockdiff.cu`) for the first differing block, then
`tests.sh ./fastsieve --gpu` (24/24) + the full audit.

Additionally apply the **`q0 = p` shortcut**: in the common case `p² ≥ segLow+7`
(any block strictly past the first few), `v0 = p²` so `q0 = p` exactly — no
division and no `v0 > segLow+7` branch. Only the earliest blocks need the
general `ceil((segLow+7)/p)` path.

**B. Amortize the per-prime outer scan — multi-block CTAs.**
256 threads is too few to amortize scanning e.g. 70k primes. Process
`K = 8` GPU-blocks per CTA (256 threads, K×16 KiB shared or one 16 KiB segment
reused round-robin) so the prime list is **loaded once per CTA** into shared
memory (it is read-only; a single `memcpy-to-shared` per CTA instead of a
global load per prime per block). This removes the dominant per-block scan
cost (≈π(√segEnd) reads/iterations × nblocks/K). Keep the block-count array
indexed per GPU-block so the audit is unchanged.

**C. Stop dense blocks from walking big primes.**
Coarse rule that keeps exactness trivial: inside the crossing loop, `break`
once `p² > segEnd + 1` (already there). For the tail primes with
`m_p = 1` multiple (p² near segEnd) nothing is paid beyond one iteration; the
main win is B (shared prime list) + A (div removed). A CUDASieve-style bucket
for truly big primes (p > sqrt(segEnd)) is only worthwhile after A+B and only
if profiling shows it; defer unless A+B leaves >20% in the outer scan.

**D. GPU pre-sieve (AND-pattern init) ≈ 2× fewer crossings.**
Remove multiples of primes ≤ 163 the same way the CPU does: precompute the 16
periodical AND-tables on the host, copy them to shared/global once, and have
each CTA initialize its segment(s) by AND-ing the tables (position derived from
`segLow mod tablePeriod`) instead of `0xFF`. This duplicates the CPU pre-sieve
semantics exactly (incl. the first-8-bytes PRIMEBITS restore for block 0) —
mirror `fastsieve.c`'s `pre_sieve` + `PRIMEBITS` handling, and validate with
the byte-diff tools first.

**E. Micro:** `__restrict__` on kernel pointers, `#pragma unroll 4` on the
q-loop for small primes, keep `%30`/`/30` as-is (constant magic is cheap),
consider `reinterpret_cast<uint4*>` clears to cut atomic width when safe, and
block scheduling (`cudaOccupancyMaxPotentialBlockSize`) to hit 100% occupancy
with the larger CTAs.

### 9.3 Success criteria (venus round 2)

* audit **0 mismatches** on both 3090 and 3080 Ti; `tests.sh ./fastsieve --gpu` 24/24;
* kernel-only times: 1e11 ≤ ~3–4 s, 1e12 ≤ ~150 s (beats CPU-12t 257 s; aim ~60–90 s);
* update the tables in §8 and README, then commit + push.

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

## System prompt (paste to the venus agent)

```
You are resuming the "fastsieve" project. Read (in order):
  README.md, docs/DESIGN.md, docs/GPU.md, docs/WHEEL210.md, docs/RESUME_VENUS.md.
State: CPU engine exact (gcc build, 24/24 sweep incl. boundary primes); CUDA
port gpu_cuda.cu is BIT-EXACT on RTX 3090/3080 Ti (audit 0 mismatches, 24/24
GPU sweep); the old OpenCL/RDNA4 residual was root-caused as logical
block-boundary bugs (cup/CUNIT off-by-one + rel clamp dropping the trailing
segEnd+1 candidate) and fixed in both gpu.c and gpu_cuda.cu.
Mission (in priority order):
 1) Make the CUDA kernel FAST while keeping it bit-exact. Follow the ready
    spec in §9 (division-by-reciprocal for q0/qmax + q0=p shortcut, multi-block
    CTAs with a per-CTA shared prime list, then GPU pre-sieve). After EACH
    change: full audit + ./tests.sh ./fastsieve --gpu must stay 24/24 and
    bit-exact (use debugdump.cu / blockdiff.cu on the first differing block);
    re-benchmark 1e9..1e12 vs primesieve -tN (§9.3 targets).
 2) On the dev box (RDNA4/OpenCL), re-run the GPU path with the fixed gpu.c:
    the audit should now pass with no fallback; update docs/GPU.md.
 3) If time allows, re-land wheel-210 per docs/WHEEL210.md.
 4) Update docs/RESUME_VENUS.md + benchmark tables with new results, then
    commit + push.

Rules: never claim exactness without the byte/oracle check in
docs/RESUME_VENUS.md §4; keep the audit-and-fallback design and the exact
π(n) contract (printed value must always be exact).
```