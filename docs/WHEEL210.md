# Wheel-210 engine — status

Branch `wheel210`. The mod-210 crossing engine (48 residues coprime to 210,
48 bits per 6-byte block, span 35·B numbers per B-byte segment) is now
**bit-exact vs primesieve 12.16 across the full gate**, single- and
multi-threaded, including the 1e12 sweep and the 786431²/786433² boundary
primes. It is NOT merged to main: benchmarks (below) show it ~1.6–2.1×
slower than the wheel-30 engine, so it stays a branch until the crossing
loop is redesigned or the experiment is abandoned.

## Verification gate (all green, 2026-09-08)

* `tests.sh`: 24/24 — pi(n) exact for n = 1e2..1e12 at -t1 AND -t12, plus
  the boundary primes 786431² = 618473717761 (23688293324) and
  786433² = 618476863489 (23688409284).
* `api_test`: 62/62, re-run on a synchronously built binary (the first
  post-fix relaunch raced `gcc -o api_test` against the api_test exec —
  process-hazard note, see below).
* `--lo` interval spot checks vs `primesieve lo hi`: windows starting and
  ending exactly on segment boundaries (9174900·k), the countLo-on-boundary
  case [9174900, 9274900], windows in the pend band (6.2e11..1e12), and
  multi-segment emit diffs vs `primesieve -p` (byte-exact prime lists).
* Above-1e12 spot windows (critical-review follow-up): [1.2e12, +2e8],
  [1.5e12, +2e8], [2e13, +2e8] and [8e15−2e8, 8e15] all exact. The 8e15
  window exercises the extremes: root = 89.4M, pend entries waiting
  thousands of segments, bucket windows of ~2000 segments.

## CRITICAL: main (wheel-30) is wrong above n ≈ 2e12

The same pend one-early bug exists on main — its first-multiple offset is
< 7p (quotient bump ≤ 6), so pend only engages for p > 30B/7 ≈ 1.12M,
i.e. n ≳ 1.26e12, just above main's verified ceiling of 1e12. Confirmed
empirically against primesieve (window [n, n+2e8], main @5823076):

| n     | main error | main + pend reorder |
|-------|-----------:|--------------------:|
| 1.5e12 | 0         | 0                   |
| 2e12  | +678       | 0                   |
| 3e12  | +6,431     | 0                   |
| 5e12  | +24,157    | 0                   |
| 1e13  | +70,048    | 0                   |
| 2e13  | +129,072   | 0                   |
| 8e15  | +5,894,756 | (not run)           |

The three-line fix (test `i < B` before decrementing, decrement only when
not pushed — commit c6b2c7f) applied to a copy of main makes 2e12/2e13
exact and regression-clean at pi(1e11) and [618.4G, 618.5G]. **Main should
take this fix**; every pi/count/generate/nth_prime answer above ≈2e12 from
the current main is suspect, and the API advertises FASTSIEVE_MAX_N = 8e15.

Error-magnitude note: the naive "every multiple of every pended prime is
missed" model overpredicts ~20x — a composite p×q with p pended-broken is
usually still cleared by a co-factor ≤ root from a correct chain; the
visible residue comes from multiples whose only ≤root factors are
themselves broken (semiprime-ish p·q with q > root).

Process note (why the first post-fix api_test result needed a re-run): the
relaunch command built api_test and exec'd it from background jobs started
in the same shell line — a build/exec race. The result happened to be
valid (the stale 06:51 binary would have failed pi(1e12) -t12 by ~+239M
via the then-unfixed bucket bug, so a 62/62 PASS proves the new binary
ran), but the race was real; the clean re-run above settles it.

## The bugs that finished it (each verified by oracle diff)

The last session left one known failure (countLo on a segment boundary)
with a mis-localized root cause. Re-deriving the counting from first
principles and pushing the gate to 1e12 for the first time surfaced six
real bugs — all in the slice/boundary machinery the wheel-30 code shares
by structure but never exercises in the tested ranges:

1. **`count_segment` rel ≤ 1 wrap** (the actual crash behind the old
   "t12 segfaults at 1e7+"): a slice whose cap lands at segLow+1 makes
   nbits = CNT210[1] − 1 = −1 wrap in u64 → nf ≈ 2.9e17 → segfault in
   popcount. Guard: `rel <= 1 → return 0`.
2. **`count_segment` full-branch phantom overcount**: the trailing bit of
   every segment stores the NEXT segment's residue-1 candidate (the
   "phantom", value segLow+35B+1). The full branch must require
   `segLow + 35B + 1 < n`; with the old `< n` a slice cap landing exactly
   on the phantom's value counted a value == cap (+1).
3. **`cbits_prefix` rel clamp**: clamping rel to 35B before the formula
   yields nbits = B·8−1, dropping the phantom whenever up > segLow+35B+2 —
   this was the [9174900, 9274900] −1 (the missed prime was the phantom
   9174901, stored as the previous segment's last bit — NOT 9174937 as the
   old session notes claimed). Fix: never clamp rel; clamp nbits to B·8.
   The bit-prefix formula `G(rel) = 48·⌊rel/210⌋ + CNT210[rel mod 210] − 1`
   with the rel ≤ 1 guard and the nbits clamp is now exact by construction
   for every rel.
4. **Emit decode missing +low**: g is segment-relative; multi-segment
   generate emitted garbage values after segment 0 (count path unaffected).
5. **Big-prime bucket unit slip (30→210 port bug)**: i0/nxt are BYTE
   indices; the bucket split divided by bps = B/6 (blocks) instead of B
   (bytes). Wheel-30 has 1 byte per block so main's divide-by-B was right;
   the port silently kept the wrong divisor → every bucket crossing landed
   6× off (+239M composites standing at 1e12). First exercised at n ≥
   (3B)² ≈ 6.2e11 — below that the bucket class is empty.
6. **Pend migration one segment early**: pend.i is relative to the segment
   where the prime was ADDED; the migration decremented B before testing,
   pushing the first crossing one segment early — the prime's whole chain
   then cleared bits at value − span (+31859 at 1e12 t12: missed composites
   AND spuriously cleared primes). Fix: test `i < B` first, decrement only
   if not pushed. **This bug is latent on main's wheel-30 too**: its first
   multiple offset is < 7p (quotient bump ≤ 6), which stays inside one
   warm-up segment (30B) for p ≤ 1.12M, i.e. n ≤ ~1.3e12 — above main's
   tested ceiling. Port the same reordering to main when convenient.

Why the old session saw none of this: its verified range stopped at 1e9
single-thread (no buckets, no pend, no 1e12), and the t12 spot checks at
1e6/1e7 predated the last commit's code.

## Benchmarks (honest numbers; shared host, min-of-N)

Same host and compiler as docs/CPU_PERF.md (i5-11600, gcc 13 -O3
-march=native); wheel-30 = main @5823076 from a worktree. Seconds, best of
3 / 2 / 1 (1e9–1e10 / 1e11 / 1e12):

| n    | t  | fs210  | fs30  | primesieve | 210/30 | 210/ps |
|------|----|-------:|------:|-----------:|-------:|-------:|
| 1e9  | 1  | 0.324 | 0.154 | 0.098      | 2.10   | 3.31   |
| 1e9  | 12 | 0.073 | 0.036 | 0.022      | 2.03   | 3.32   |
| 1e10 | 1  | 3.87  | 1.93  | 1.24       | 2.01   | 3.12   |
| 1e10 | 12 | 0.66  | 0.36  | 0.28       | 1.81   | 2.35   |
| 1e11 | 1  | 46.5  | 26.9  | 18.6       | 1.73   | 2.50   |
| 1e11 | 12 | 27.2  | 14.9  | 12.4       | 1.83   | 2.19   |
| 1e12 | 1  | 706.7 | 416.1 | 251.1      | 1.70   | 2.81   |
| 1e12 | 12 | 347.5 | 216.5 | 162.4      | 1.60   | 2.14   |

The −14% crossing cut (6/7 wheel ratio) did NOT materialize: per segment
both grids store the same 2.097M candidate bits, so the 210 engine does
0.857× the segments but pays ~2.34× per segment → net ~2× slower.

## Where the 2.3×/segment goes (rdtsc phase profile, 1e10 t1)

Phase shares of measured cycles are the same shape as wheel-30
(small 63%, medium 32%, pre-sieve 4%, count 0.5%), but ~40% of total
cycles fall outside the phase timers (add/pend/loop overhead — itself a
lead). Measured micro-findings:

* `tinc`'s `% 48` and the sub-batch `(st+8) % 48` → branchless: only
  +2.5% (tried, not the bottleneck; left as-is since `% 48` documents the
  wheel invariant).
* Prime suspects (unproven): the 48-entry u16 table rows are 240 B/prime
  vs wheel-30's 8-entry rows that fit a register (gcc hoists a whole row;
  the 210 loop streams L2 per store), and the 16 pre-sieve tiles total
  ~466 KB vs wheel-30's ~78 KB (L2- vs L1-resident).

## Follow-up leads (ordered)

1. Crossing-loop redesign for the 48-state tables: per-prime packed rows,
   or revive the parked Duff generator (`tools/gen_wheel_duff.py`,
   `wheel_duff.h`) — immediates should pay far more here than the 5% they
   gave on wheel-30, because table pressure is the binding constraint.
2. Migrate pend primes into the BUCKET lists, not med (they are p > medMax
   by construction; med makes them sweep every segment whole-range) —
   also explains the poor t12 scaling (2× at 12 threads).
3. Account for the ~40% out-of-timer cycles before trusting any micro
   optimization.
4. Port the pend-migration reordering (bug 6) to main — **confirmed wrong
   there above n ≈ 2e12, see the section above**; the three-line fix is
   verified (main + fix exact at 2e12/2e13).
5. GPU: kernels are wheel-30; a 210 port is deliberate follow-up work.

## Debug tooling that earned its keep (in /tmp, recreate if lost)

`FASTSIEVE_DUMP_SEG=n` dumps segment n after crossing; an absolute-space
naive comparator (cross primes ≤ sqrt(hi+1) from q = max(p, LO/p), bump to
coprime residue — start q near LO or it iterates billions) byte-diffs the
dump and prints wrong bits with values; factoring the kept composites
identifies the desynced prime class instantly; a TRACE_P env hook in
prime_init_state / pend-migration / cross_flat follows one prime's chain
through a slice. The counting formula was re-derived symbolically
(G(rel) above) instead of patched case-by-case — that is what exposed the
clamp/guard/condition bugs as one family.
