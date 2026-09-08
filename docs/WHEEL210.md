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

## Bottleneck hunt (2026-09-08, second session)

Twin rdtsc instrumentation of both engines (same session, same clock)
isolated the gap to the SMALL-prime crossing: ~5.6 cycles/crossing vs
wheel-30's ~1.9. Disassembly showed why: wheel-30's 8-state cycle aligns
with its 8-store batch, so all offset/state math hoists out of the store
loop; the 210 wheel's 48-state cycle forced seven `%48` magic-number
division sequences plus the full offset chain to be recomputed for every
8 stores, with register spills.

Three changes landed (all verified against the full gate):

1. Branchless state advance (`tinc`, sub-batch `st+8`).
2. Packed per-state step table `ST210[k][t]` (u32: low16 = CB, high16 =
   CV) — one load per state instead of two arrays.
3. **Rotated fast path**: doubled rows `ST2/BM2[k][t..t+47]` make a full
   48-state cycle a contiguous window from ANY starting state t; a full
   cycle starts and ends at state t, so the whole offset vector hoists
   out of the store loop — the 210 analogue of wheel-30's aligned cycle.

Result (min-of-N, same-host): the gap narrowed from ~2.0× to 1.22–1.45×:

| n    | t  | fs210  | fs30  | primesieve | 210/30 | 210/ps |
|------|----|-------:|------:|-----------:|-------:|-------:|
| 1e9  | 1  | 0.224 | 0.155 | 0.099      | 1.45   | 2.26   |
| 1e9  | 12 | 0.048 | 0.037 | 0.022      | 1.30   | 2.18   |
| 1e10 | 1  | 2.77  | 1.93  | 1.24       | 1.43   | 2.23   |
| 1e10 | 12 | 0.458 | 0.362 | 0.317      | 1.27   | 1.44   |
| 1e11 | 1  | 41.9  | 30.0  | 16.9       | 1.40   | 2.47   |
| 1e11 | 12 | 17.9  | 13.7  | 12.7       | 1.30   | 1.41   |
| 1e12 | 1  | 535.8 | 415.2 | 254.8      | 1.29   | 2.10   |
| 1e12 | 12 | 255.5 | 209.0 | 156.3      | 1.22   | 1.63   |

Measured micro-findings:

* `%48` → branchless alone: only −14% (the division sequences matter but
  are not the whole story; the hoisting is).
* The rotated path is a win for primes with ≥ ~48 multiples per call; the
  legacy sub-batch path (with cmov indices + packed tables) is kept for
  the rest.
* The med class improved 1.59× → ~1.25×; small crossing 2.5× → ~1.66×.

## Follow-up leads (ordered)

1. **Per-segment offset reuse** (attempted, reverted): the per-call
   `off[48]` build (~200 cycles) still taxes low-multiple primes.  The
   plan — build once per prime per SEGMENT, let mid-segment chunks
   overshoot into the next chunk (AND is idempotent), serial-tail only the
   last chunk — has an UNRESOLVED carried-state bug: the (byte, state)
   pair desyncs at segment boundaries (probe: byte ≡ 1 mod 6 carried with
   state 0, which requires byte ≡ 5).  Needs a single-prime cycle-exact
   simulator before re-attempting; the rest of the design was sound.
2. Port the pend-migration reordering (bug 6) to main — **confirmed wrong
   there above n ≈ 2e12, see the section above**; the three-line fix is
   verified (main + fix exact at 2e12/2e13).
3. GPU: kernels are wheel-30; a 210 port is deliberate follow-up work.

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
