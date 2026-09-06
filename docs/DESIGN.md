# Design

## Layout

The sieve array is a **bit-packed wheel-30** bitmap. Every byte holds 8 flags,
one per candidate residue mod 30, covering 30 consecutive numbers:

```
byte j  ↔  numbers 30·j + {7, 11, 13, 17, 19, 23, 29, 31}
```

Because 2, 3 and 5 divide none of these residues, multiples of 2, 3 and 5 do
not exist in the representation at all (memory and crossing work reduced by
3.75×). The bit for a value `v` lives in byte `(v-6)/30` (the `+6` convention)
and the residue `1` is stored as offset 31, one byte *earlier* than
`v/30` suggests – a trap that historically broke naive implementations.

## Segmented pipeline, per segment

1. **Pre-sieve** – AND 16 periodical lookup tables (period = product of a
   subset of primes ≤ 163) into the fresh segment via AVX2. This removes, for
   free, the crossing work of every prime ≤ 163 (~half of all crossings).
2. **Add sieving primes** – primes whose `p²` is inside the current segment can
   start crossing here; a prime with `p²` at the trailing (`+1 mod 30`) byte of
   the segment is added while `p² ≤ low + 30·B + 1` (this is the fix for the
   `786431²` boundary case).
3. **Cross** – three classes:
   * *small/medium*: every prime visited each segment; its multiples are
     crossed with a wheel-30 step table. After the last multiple, the prime
     stores the index *just past* it, which is exactly the first index of the
     next segment – so **no per-segment re-initialization ever happens**.
   * *big* (p > sieve-bytes × 3): Oliveira e Silva's bucket sieve – each prime
     is only visited on the segment containing its next multiple.
4. **Count** – hardware popcount; segments that end on `±1 mod 30` frame
   boundaries use an explicit candidate-count formula.

The wheel step tables (`cv, cb, bit` per prime-residue × step) are generated
at startup from first principles.

## Multithreading

The range is split into independent, one-segment-overlapping slices; each slice
starts one segment early but only *counts* from its base, so its crossing state
equals the single-thread run and the `lo+1` trailing candidate is counted
exactly once. Slices are distributed with OpenMP.

## GPU accelerator

See [GPU.md](GPU.md). A work-group per sub-segment sieves 30·16384 numbers in
local memory (256 lanes), clearing candidate bits with word-level local atomic
AND (race-free) and counting with a work-group popcount reduction. The GPU
result is **audited** against exact CPU segments; mismatch ⇒ CPU fallback.

## Verification

`tests.ps1` (see README) validates against hard-coded `π(n)` oracles across
single/multi-thread CPU and GPU, including the boundary primes `786431²`,
`786433²` and GPU block boundaries.