# fastsieve C API - design notes

## Motivation

The sieve engine (fastsieve.c) grew as a CLI that prints `pi(n)`. Every
downstream use needs to *call* the engine, not parse its stdout. The API is a
thin, dependency-free C surface over the same code path the CLI runs, so the
exactness guarantee (CPU fallback, GPU audit) holds identically for callers.

## Files

* `fastsieve.h`      - the public header (single include, self-contained).
* `fastsieve.c`      - the engine plus the API implementation (same file, so
                       there is no separate translation unit to drift).
  * Build the CLI as before: `build.bat`.
  * Build a program against the library:
    ```
    cl /O2 /arch:AVX2 /Oi /openmp fastsieve.c gpu.c your_program.c
    ```
    (Linux: `gcc -O3 -march=native -fopenmp fastsieve.c your_program.c`.)
    Set `FASTSIEVE_NO_MAIN` when your program has its own `main()`,
    e.g. `build.bat` does this for `examples/api_demo.c`.
* `examples/api_demo.c` - small self-check demo (also a smoke test).

## Function set

| Function | Purpose |
|----------|---------|
| `fastsieve_init`        | idempotent engine init (tables built once) |
| `fastsieve_version`     | version string |
| `fastsieve_pi(n, cfg)`  | pi(n), exact |
| `fastsieve_count(lo,hi,cfg)` | number of primes in [lo, hi] |
| `fastsieve_isprime(n,cfg)`   | exact primality test |
| `fastsieve_nth_prime(k,start,cfg)` | k-th (1-based) prime >= start |
| `fastsieve_generate(lo,hi,cb,user,cfg)` | enumerate primes in ascending order |

`fastsieve_config` selects threads, GPU, segment size and the
medium/big crossover; `NULL` means engine defaults.

## Design choices

* **`const fastsieve_config*` (never globals).** The CLI's knobs are exported as
  one optional struct so callers pay nothing by default and can configure per
  call. Thread-count `<=0` = engine default (1), `sieve_bytes=0` = 256 KiB,
  `med_factor<=0` = 3.0 - mirrors the CLI.
* **Exact > fast, always.** `fastsieve_pi` routes through the audited GPU path
  when `use_gpu` is set and transparently falls back to CPU on any mismatch or
  missing device. Callers can never observe a wrong count.
* **`int64_t` returns; negative = error.** Range errors
  (`n > FASTSIEVE_MAX_N`, `lo > hi`, `k == 0`) are distinguishable from valid
  results without a separate status out-param.
* **Generation via callback, ascending order.** `cb` returning nonzero stops
  the scan early; the returned count is the emitted-so-far. Useful for "print
  the first N primes" without buffering.
* **Latent bug fixed while exposing the API.** The counting convention is
  "candidate values `< cap`". The CLI passed `cap = n`, so a *prime* upper
  bound `n` was silently dropped (`pi(13)` used to be 5). The API core now
  sieves to `n + 1`, so every function is exact at arbitrary prime bounds;
  validated by `pi(10000000019) = pi(10000000018) + 1` on CPU and GPU.
* **Thread-safety.** Per-call state only; the shared lookup tables are built
  lazily. Call `fastsieve_init()` once from a single thread before multithreaded
  use. Each call is then independent and may run concurrently.

## Performance semantics (who pays what)

* `fastsieve_pi(n)` and `fastsieve_count(2, n)` use the fastest engine
  (OpenMP threads, optional audited GPU) - `pi(1e12)` ≈ tens of seconds on GPU.
* `fastsieve_count(lo, hi)` for `lo > 2` and `fastsieve_generate` use the
  single-threaded CPU generator over `[lo, hi]`. For large ranges, split the
  range across threads/processes yourself (each call is independent).
* `fastsieve_nth_prime` is exact via bracketing + binary search over
  `fastsieve_pi` (≈ a handful of full-range counts), so it is *not* the
  cheapest way to find a tiny prime; it is the simple/correct way.

## Evolution candidates (deferred, not required)

* Streaming iterator state (`fastsieve_iterator_t`) so `generate` doesn't need a
  per-prime callback.
* Parallel range enumeration inside `generate` with ordered merges.
* `fastsieve_provesieve` / pre-sieved standalone priming for tiny queries.
* HIP/ROCm backend (see docs/GPU.md) would automatically flow into
  `fastsieve_pi`/`fastsieve_count(2, ...)` through the existing audit.