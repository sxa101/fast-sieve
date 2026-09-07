# Wheel-210 session context (paste as system prompt for the next agent)

You are resuming the wheel-210 development branch of "fastsieve" (repo
/home/sa/fast-sieve, branch `wheel210`, head 105dae6). Read first:
README.md, docs/DESIGN.md, docs/WHEEL210.md, docs/CPU_PERF.md,
docs/RESUME_VENUS.md (background), then this file.

## Mission

Finish the wheel-210 crossing engine (mod-210 wheel, 48 residues, 6 bytes
per block, span 35*B per segment) so it is bit-exact vs primesieve
everywhere, then benchmark vs primesieve and vs the wheel-30 engine
(main branch). Success = tests.sh-equivalent sweep exact (1e2..1e12,
t1 and t12, boundary primes 786431^2/786433^2), then honest benchmarks.
Do NOT merge to main until the full gate passes.

## Current state (all on the branch, committed)

pi(n) exact single-thread for n = 1e2 .. 1e9 (verified vs primesieve
12.16 at ~/.local/bin/primesieve). Multi-segment --lo slices verified at
1e6/1e8/6.5e8. t12 exact at 1e6/1e7. Boundary primes (p^2 = 1 mod 210,
p = 211..23561) exact. GPU disabled on this branch (kernels are
wheel-30); --gpu falls back to CPU and stays exact.

## The ONE known remaining failure

`./fs210 --lo 9174900 --hi 9274900` (any slice whose countLo lands
EXACTLY on a segment boundary): pi = 6286, primesieve = 6287 (-1; the
missed prime is 9174937 = the first candidate of segment 1, at
segment-relative byte 0, bit 7).

Root cause (localized, fix is ~2 lines): the slice is
sieve_slice(segStart=0, countLo=9174900, cap=9274901). Segment 1's count
comes from cbits_prefix(cap - low = 100101) MINUS cbits_prefix(countLo -
low = 0). cbits_prefix with rel = 0 computes
nbits = 48*0 + CNT210[0] - 1 = -1, which wraps in u64 and (after the
nbits > 8B clamp) makes cbits_prefix return the popcount of the WHOLE
buffer instead of 0. Segment 1's contribution then = popcount[0,22856] -
popcount[all] = garbage; net effect measured: segment 1 contributes 0
instead of 6287 - wait, measured total was 6286 = 6287 - 1, so precisely
one prime went missing via the rel=0 wrap interacting with the final
partial-segment cbits. FIX: in cbits_prefix, after computing nbits, if
((i64)nbits < 0) return 0 (or clamp nbits to 0 before the nf/nb split).
Then re-run: the failing window, the full sweep, t12 sweep, the 618G
boundary prime (t12 showed +1 there too, likely the same boundary-slice
cbits issue via a different slice split), and benchmark.

## Wheel-210 design (as implemented)

- Grid: 48 residues coprime to 210 stored 48 bits per 6-byte block;
  bit m of block j (m 0..46) = value 210j + RES210[m+1]; bit 47 =
  210j + 211 (the next block's residue-1 candidate: the wheel-30
  offset-31 trick generalized; MIDX210[r] = 47 for r == 1). Byte j
  covers [low+210j+11, low+210j+211]; first stored candidate of a
  segment is low+11; value low+1 is the phantom (belongs to the
  previous segment's last bit). Span = 35*B numbers per B-byte segment
  (B must be a multiple of 6: 262140 = 4 mod 8 - NOT 8-aligned, which
  matters for popcount tails).
- Crossing tables (built in build_cross_tables210): CV210[k][t] = 6*dq
  (u16; bytes/number = 1/35 so the sp-coefficient is 6*dq), CB210[k][t]
  = byte remainder via bg-floor (bg210(v1 + rk*dq)/8 - bg210(v1)/8),
  BM210[k][t] = clear mask for the multiple's bit. State t indexes the
  quotient's residue: q = RES210[t] mod 210. ALL state advances must be
  % 48 - & 47 is bitwise AND and silently collapses the cycle (47 is
  not 48-1 in bitmask terms; wheel-30's & 7 only worked because 8 is a
  power of two).
- NEXT210[q] = next coprime residue >= q (right-to-left sweep;
  NEXT210[0] = 1). Used by prime_init_state to bump q to a coprime
  residue before computing the first multiple.
- Out-of-band primes are {2,3,5,7} (+4, thresholds n>=7/5/3/2); the
  pre-sieve groups drop 7 (vacuous on the 210 grid); pre-sieve tiles
  are 16 blocks = 96 bytes with +16-block padded tables; the scalar
  tail must ASSIGN from table 0 then AND the rest (never &= into
  untouched bytes).
- Counting: count_segment full branch popcounts ALL B bytes including
  the tail (B = 4 mod 8; B/8 words drop the last 4 bytes = real
  candidates - fixed, keep it). Partial branch: nbits = (rel/210)*48 +
  CNT210[rel%210] - 1 (no rr correction; the -1 removes the phantom
  candidate 1; correct for rr >= 2 and, with the cbits rel=0 fix above,
  for the rel=0 boundary case).
- Emit (generate path): byte->value decode v = 210*(g/48) +
  (g%48==47 ? 211 : RES210[1+g%48]) where g = 48*(j/6) + 8*(j%6) + bit.
  NOTE: for multi-segment slices with lo > 0 this decode is missing the
  +low offset - the emitted VALUES are wrong for segments after the
  first (the count path is unaffected; single-segment slices with
  segStart=0 are unaffected). Check whether this matters for the API
  (fastsieve_generate) before shipping; fastsieve_count uses
  fs_generate_core -> sieve_slice counting, not the emit values.

## Bugs already fixed this session (do not regress; each has a story)

1. Builder state cycle: & 47 vs % 48 (above).
2. CV210 type: u8 -> u16 (6*dq overflows u8 for dq >= 43); cross_flat
   pointer must be const u16*.
3. NEXT210 builder init-order: the marker init NEXT210[r] = 210 ran
   inside the fill loop, so every non-coprime index was re-wiped to the
   marker AFTER a coprime had filled it; prime_init_state then jumped q
   by +76 onto a non-candidate multiple (90875190 % 210 == 0) and
   desynced that prime's whole crossing. THIS was the historical
   wheel-210 "slice-band residual" mechanism.
4. NEXT210 fill semantics: marker-guarded ascending fill left
   NEXT210[q] = 11 for all q (first coprime's pass won) -> init bump
   went BACKWARD below the segment start -> negative rel -> i0 wrapped
   to a huge u32 -> prime went to pend and never crossed. Fixed with
   the right-to-left sweep.
5. count_segment tail bytes (B%8 = 4): full-branch popcount dropped the
   last 4 bytes of every full segment (~8 primes/segment).
6. Counting formula: a wrong-direction tail correction (+1 for rr<2)
   was applied then reverted - the correct form is plain
   48q + CNT210[rr] - 1 (with the cbits rel=0 fix from above handling
   the boundary).

## Debug tooling (proven, reuse these)

- FASTSIEVE_DUMP_SEG=n ./fs210 ... : dumps segment n's sieve bytes to
  /tmp/segdump.bin (hook is in sieve_slice, after crossing, before
  counting).
- /tmp/segcmp3.c <hi> <lo>: absolute-space naive comparator. Builds the
  true primality pattern for [lo, lo+35B) (cross multiples >= p*p for
  primes <= sqrt(hi+1)), byte-diffs vs /tmp/segdump.bin, prints wrong
  bits with values. NOTE: its bg/LO reference must be a candidate
  (LO+11), and its crossing loop must cover the engine's prime set
  (primes <= sqrt(hi+1)) - both were bug sources during debugging.
- /tmp/sim2.c: bit-level simulator, recurrence (tables) vs independent
  value walk for all (k, t0, sp), 600 steps, compares (byte,bit) clears.
  CAUTION: any & 47 / shared-index corruption makes it pass vacuously
  (both sides share the bug) - keep the naive side table-free.
- /tmp/printprimes.c: fastsieve_generate printer; diff vs
  `primesieve -p --no-status lo hi` catches emit-path bugs.
- TRACE env hooks in /tmp/fsdbg.c builds (per-prime step traces).

## Verification gate (run before ANY claim of exactness)

1. ./fs210 n for n = 1e2..1e12 vs primesieve (t1).
2. --lo windows: [1e6,1.1e6], [1e8,1.001e8], [6.5e8,6.51e8],
   [9174900,9274900] (the boundary case), [9e6,1e7].
3. Boundary primes 618473717761 / 618476863489 (= 786431^2 / 786433^2),
   t1 AND t12.
4. -t 12 sweep 1e6..1e9.
5. generate/emit spot check vs primesieve -p on a multi-segment window
   with lo > 0 (exposes the emit +low question above).
Everything must be exact - the engine has no heuristic slack.

## Then: benchmarks (the actual point)

vs primesieve -t1/-t12 and vs the wheel-30 build (main branch, build
from a main worktree): 1e9, 1e10, 1e11, 1e12. Wheel-30 numbers on this
host (i5-11600, gcc 13): see docs/CPU_PERF.md (ours ~1.1-1.3x behind
primesieve single-thread; parity at t12 >= 1e10). Wheel-210 should cut
~14% of crossings; report honest numbers either way. Also re-check
cache behavior: the 210 grid changes bytes/number from 8/30 to 6/210·8 =
0.8/30 (14% less memory traffic).

## Rules

- Never claim exactness without the oracle diff (primesieve) on the
  full gate.
- The engine must stay exact-by-construction (no heuristic slack); the
  GPU path stays disabled on this branch until the CPU wheel-210 is
  proven and the 30->210 GPU port is a deliberate follow-up.
- Commit incrementally on wheel210 with messages that explain the WHY.
- Update docs/WHEEL210.md with the final state (the re-landing
  checklist there is now mostly done; rewrite it as a status doc).
