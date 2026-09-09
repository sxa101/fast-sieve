# Wheel-210 experiment – history and findings

## Status (2026-09-09)

* The wheel-210 crossing engine was **revived on branch `wheel210`** and is
  bit-exact across the full gate (pi sweep to 1e12, single- and multi-thread,
  the 786431²/786433² boundary primes, spot windows up to 8e15). It remains a
  branch, **not merged**: measured 1.22–1.45× **slower** than the wheel-30
  engine here, so the re-landing checklist below is superseded by that
  branch's write-up (docs/WHEEL210.md on `wheel210`).
* **CRITICAL (found via the 210 work): the wheel-30 engine on main was wrong
  above n ≈ 2e12.** The pend-migration decremented the carry index *before*
  testing `i < B`, pushing a pended prime's first crossing one segment early.
  Dormant until a prime's first-multiple offset can exceed one segment
  (p ≳ 1.1M, i.e. n ≳ 1.3e12) — just above the old test ceiling, while the
  API advertises 8e15. Error (window [n, n+2e8]):

  | n     | error |
  |-------|------:|
  | 1.5e12 | 0    |
  | 2e12  | +678 |
  | 3e12  | +6,431 |
  | 5e12  | +24,157 |
  | 1e13  | +70,048 |
  | 2e13  | +129,072 |
  | 8e15  | +5,894,756 |

  Fixed by commit `4ffa6f3` (test `i < B` first, decrement only when not
  pushed); regression windows above 2e12 added in `cf1688d` (tests.sh /
  tests.ps1 / api_test.c). Full root-cause write-up: `wheel210` branch,
  docs/WHEEL210.md.

## Goal

A mod-210 wheel skips multiples whose quotient is divisible by 7 (in addition
to 2,3,5). Since 7 ≤ 163 is already pre-sieved, skipping those multiples is
free and harmless, cutting the number of crossings by
`(48/210)/(8/30) = 6/7 ≈ −14%`.

## What was done and what was found

* The 48-step wheel-210 tables (byte deltas `dk·sp + cb`, per prime-residue
  class and step) were generated from first principles and **verified
  bit-exact** against a naive reference on a full segment.
* The one critical framing subtlety was found **empirically** and fixed: a
  crossing step whose byte-delta `≥ B` can jump whole segments; the carry
  index must advance **exactly one segment per call** (never normalize with
  `while (i>=B) i-=B`, which cleared one frame too early). A
  multi-frame, single-prime test across 6000 frames then passed.

## Why it was reverted

After integration, multi-segment **sliced/multithreaded** runs over the
big-sparse-prime band (≈6.5e11+) still showed a small deterministic residual
(~1–4 composites/segment), reproducible at the segment level, while the
single-thread full-range run stayed exact. The wheel-210 crossing math was
proven correct; the residual was a *carry/frame accounting interaction*
inside the slice machinery.

Because correctness outranks a 14% win, the engine ships with the **wheel-30**
crossing (verified exact to 1e12, single & multi-thread) and the wheel-210 work
is documented here for a focused follow-up.

## Re-landing checklist (for a fresh session)

1. Restore the wheel-210 tables + crossing (`UNITS210/KMAP210/NXT210/DK/CB210/BM210`,
   sparse/dense split). The design is captured in the git history and in the
   source comments of the wheel-30 tables block.
2. Reproduce the slice-band residual with **one isolated over-lapped slice**
   (`sieve_slice(segStart, countLo, cap, ...)`), comparing per-segment to the
   `primesieve lo hi` interval (reference-trustworthy).
3. Track the carry of sparse primes across many slice-internal segments, in
   particular when `step ≥ B` lands at the first segment of a slice.
4. Re-run `tests.ps1` and the 1e12 multi-thread parity run.