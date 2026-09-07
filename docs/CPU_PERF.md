# CPU performance investigation — closing the gap to primesieve

Date: 2026-09-07, Linux/CUDA host (i5-11600, 12 threads, gcc 13 -O3 -march=native).
Oracle: primesieve 12.16, same compiler. **Caveat: the host is a shared machine with
bursty neighbor load (loadavg spikes to 9+); all numbers are min-of-N wall
times, ratios are trustworthy, absolute values are ~±20%.**

## 1. The gap, measured

Single thread, before any change:

| n     | ours   | primesieve | ratio |
|-------|-------:|-----------:|------:|
| 1e9   | 0.204 s | 0.099 s    | 2.06  |
| 1e10  | 2.413 s | 1.241 s    | 1.94  |
| 1e11  | 29.3 s  | 16.1 s     | 1.82  |

(The historical 1.5–1.7× was measured on the Zen3/MSVC dev box; gcc on this
Rocket Lake starts at ~1.9×.)

## 2. Where the time goes (rdtscp phase profile, 1e10, 1 thread)

| phase                    | cycles  | share |
|--------------------------|--------:|------:|
| small-prime crossing     | 4.12 G  | 61%   |
| medium crossing          | 2.18 G  | 32%   |
| pre-sieve (AVX2 AND)     | 0.43 G  | 6%    |
| big-prime buckets        | 0.08 G  | 1%    |
| popcount counting        | 0.06 G  | 1%    |

93% of runtime is `cross_flat`. Any fix must target the crossing loops.

## 3. What was tried

### 3.1 L1-chunk size — **ADOPTED, the big win (+40–50%)**

`L1_CHUNK` was `(1u << 18)` = 256 KiB — equal to the whole segment, so the
"L1-chunked" small-prime crossing never chunked at all: each small sieving
prime swept a 256 KiB working set from L2 (~14-cycle hits). With 32 KiB
chunks (= L1D) the sweeps stay resident in L1 (~4-cycle hits):

| L1_CHUNK | 1e9    | 1e10   |
|----------|-------:|-------:|
| 256 KiB  | 0.192 s| 2.275 s|
| 128 KiB  | 0.178 s| 2.132 s|
| 64 KiB   | 0.142 s| 1.782 s|
| **32 KiB** | **0.127 s** | **1.623 s** |
| 16 KiB   | 0.156 s| 1.934 s|
| 8 KiB    | 0.215 s| 2.515 s|

32 KiB = L1_BYTES is exactly right (it is also not an accident that
primesieve's EratSmall uses an L1-sized work chunk). Non-power-of-two 48 KiB
was pathological (3.19 s). Resulting single-thread gap:

| n     | ours (fixed) | primesieve | ratio |
|-------|-------------:|-----------:|------:|
| 1e9   | 0.127 s      | 0.099 s    | 1.28  |
| 1e10  | 1.623 s      | 1.241 s    | 1.31  |
| 1e11  | 24.9 s       | 22.5 s     | 1.11  |

Segment-size re-sweep with the new chunking confirms 256 KiB stays optimal
(128: 2.00 s, 256: 1.94 s, 512: 1.97 s, 1024: 2.13 s at 1e10).

At 12 threads the fixed engine now *beats* primesieve at 1e11 (4.7 s vs
10.5 s min-of-3) and matches it at 1e10 (0.40–0.43 s vs 0.40 s).

### 3.2 64-case Duff's-device crossing — implemented, PARKED

Generated a fully-unrolled specialized 8-store cycle for each (residue class,
wheel state) pair with cv/cb/mask immediates (`tools/gen_wheel_duff.py` →
`wheel_duff.h`, mirrors `build_cross_tables()` exactly). Two generator bugs
were caught by a 200k-trial randomized A/B harness of old-vs-new cross_flat
(cumulative offsets double-added for o2..o7 AND for strd — both fixed;
absolutely verify generated constants with an A/B harness before trusting a
generator!).

Result: only **+5% at 1 thread**, inconsistent at 12 threads (sometimes much
worse — the ~30 KB of switch code pressures the uop cache/L1i, notably with
2 hyperthreads per core). With the L1-chunk fix in place the loop is
L1-latency-bound, not decode-bound, so immediates barely matter. **Parked**:
the generator and this analysis stay in the repo; `fastsieve.c` keeps the
compact table-driven loop.

### 3.3 Not tried yet (ordered by expected value)

1. **Wheel-210 re-land** (docs/WHEEL210.md): −14% crossings is the largest
   remaining *structural* lever; the frame-carry bug that reverted it is now
   well understood (same family as the GPU boundary bugs fixed this session).
2. **PGO** (`-fprofile-generate/use`): free 5–15% typically on branchy sieves;
   needs a profile-build step in build.bat / Makefile.
3. **smallMax / medFactor retune** with 32 KiB chunks (smallMax is still
   `0.2·L1_BYTES = 6.5k`; the optimum may move with the chunk change).
4. **Wider pre-sieve** (primes ≤ 233): pre-sieve is only 6% of runtime, so the
   ceiling here is a few percent — but crossing skips compound with it.
5. Micro: counting and buckets are already ~1% each; nothing to win.

## 4. Where this leaves us

Single-thread ~1.1–1.3× behind primesieve (from ~1.9×); 12-thread at parity
or ahead at ≥ 1e10. The remaining single-thread gap is consistent with
primesieve's remaining bag of tricks (hand-tuned per-CPU prefetch, medium-prime
loop specialization). Next session: wheel-210 re-land is the item with real
structural upside.
