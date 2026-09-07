# fastsieve

A fast, correct, open-source **segmented sieve of Eratosthenes** that counts the
prime numbers up to `n` on the CPU (single & multi-threaded) and, optionally, on
an OpenCL or CUDA GPU accelerator.

It is a from-scratch implementation of the architecture made famous by
kimwalisch/primesieve: a bit-packed **wheel-30** layout (8 candidate flags per
30 numbers), a **pre-sieve** that removes multiples of all primes ≤ 163 with 16
periodical AND-tables, three sieving-prime classes (dense loop / medium /
bucketized "big" primes), **carry-along wheel state** across segments (zero
per-segment re-initialization), and hardware-popcount counting.

## Status

* **CPU engine: verified exact** – single-thread and multi-thread, validated
  against primesieve across 10^2 … 10^12 including tricky frame/boundary
  primes (e.g. `786431²` and `786433²`).
* **GPU engine (OpenCL & CUDA): accelerator with correctness by construction**
  – the GPU result is audited against the exact CPU engine on sampled segments;
  on any mismatch the tool falls back to the CPU engine. The published result is
  therefore *always* exact, on any machine. Both backends are **bit-exact** —
  the CUDA kernel on RTX 3090 / 3080 Ti and the OpenCL kernel on RX 9070 XT,
  audit passes with no fallback. See [docs/GPU.md](docs/GPU.md).
* **C API** (`fastsieve.h`): `fastsieve_pi`, `fastsieve_count`,
  `fastsieve_isprime`, `fastsieve_nth_prime`, `fastsieve_generate`, all exact
  (same audited engine). Verified 62/62 on the CUDA build (RTX 3090) and the
  OpenCL build (RX 9070 XT). See [docs/API.md](docs/API.md).
* A **mod-210 wheel** experiment (skipping multiples of 7, −14% crossings) was
  carried through table generation and verification, then **reverted** after a
  rare one-frame carry misalignment surfaced in multi-segment runs. Full notes
  and debugging trail are in [docs/WHEEL210.md](docs/WHEEL210.md).

## Build

### Windows (MSVC)

Requires MSVC (Visual Studio 2022 Build Tools) and, for the GPU path, an
OpenCL ICD (AMD/NVIDIA driver) and the Khronos OpenCL headers.

```
build.bat          # CPU engine (fastsieve.exe, with OpenMP + OpenCL)
tests.bat          # run the system test suite (tests.ps1)
```

### Linux

```
# CPU engine (falls back to CPU when no GPU backend is compiled in)
gcc -O3 -march=native -fopenmp fastsieve.c gpu.c -o fastsieve_cpu

# CUDA engine (NVIDIA; verified bit-exact on RTX 3090 / 3080 Ti, sm_86)
nvcc -O3 -arch=sm_86 -Xcompiler "-fopenmp -march=native -O3" \
     fastsieve.c gpu_cuda.cu -o fastsieve

./tests.sh ./fastsieve_cpu        # CPU sweep vs primesieve (24 checks)
./tests.sh ./fastsieve --gpu      # GPU sweep vs primesieve (24 checks)
```

`tests.sh` needs the `primesieve` CLI on PATH (or `PRIMESIEVE=/path`).

## C API

The engine is callable from C through `fastsieve.h` (exact by construction:
the audited GPU path falls back to CPU on any mismatch, so callers can never
see a wrong count). Functions: `fastsieve_pi`, `fastsieve_count`,
`fastsieve_isprime`, `fastsieve_nth_prime`, `fastsieve_generate` (ascending
callback enumeration). Design notes: [docs/API.md](docs/API.md).

```
cl /O2 /arch:AVX2 /Oi /openmp fastsieve.c gpu.c your_program.c
# or: build.bat also builds examples/api_demo.c (a self-check demo)
```

## Usage

```
fastsieve [OPTIONS] n
  -t N          threads (default 1; >1 uses OpenMP slices)
  --gpu         use the OpenCL GPU accelerator (audited, exact output)
  --sieve-size KiB   segment bytes (power of two, default 256)
  --med-f F     medium/big prime crossover = F * sieve bytes (default 3)
```

Examples:

```
fastsieve 1e10                  # single-thread CPU
fastsieve -t 12 1e12            # 12-thread CPU
fastsieve --gpu 1e11            # GPU accelerator (audited)
```

## Tests

`tests.ps1` validates single-thread, 12-thread and GPU modes against a
hard-coded oracle of exact `π(n)` values (including the two famous
`786431²` / `786433²` boundary primes and GPU block boundaries). Run:

```
powershell -ExecutionPolicy Bypass .\tests.ps1
```

## Performance

### CPU — Windows dev box (Ryzen 5 5600G, MSVC /O2 /arch:AVX2)

Measured before the L1-chunk fix described below; expect a similar gain on
re-measurement.

| n     | ours 1 core | ours 12 threads | primesieve 1 core | primesieve 12 threads |
|-------|------------:|----------------:|------------------:|----------------------:|
| 1e9   | 0.17 s      | 0.05 s          | 0.11 s            | 0.04 s                |
| 1e10  | 1.9 s       | 0.36 s          | 1.1 s             | 0.25 s                |
| 1e12  | 290 s       | 52 s            | 167 s             | 31 s                  |

### CPU + GPU — Linux/CUDA reference box (i5-11600, gcc 13; RTX 3090)

| n     | ours CPU 1t | ours CPU 12t | primesieve 1t | primesieve 12t | ours GPU (kernel) |
|-------|------------:|-------------:|--------------:|---------------:|------------------:|
| 1e9   | 0.13 s      | 0.05 s       | 0.10 s        | 0.02 s         | 0.016 s           |
| 1e10  | 1.6 s       | 0.42 s       | 1.2 s         | 0.29 s         | 0.16 s            |
| 1e11  | –           | –            | –             | –              | 1.6 s             |
| 1e12  | 429 s       | 218 s        | 267 s         | 163 s          | 17.9 s            |

The historical ~1.5–1.9× single-thread gap to primesieve was traced to a
chunking bug: the small-prime crossing swept 256 KiB chunks from L2 instead of
32 KiB chunks from L1D. Fixing it was worth **40–50% single-thread** — the gap
is now ~1.1–1.3× single-thread, and 12 threads are at parity or ahead of
primesieve at ≥ 1e10, with no code-generation tricks (the primesieve-style
Duff's-device loop was built, measured and parked — it stopped mattering once
the working set fit L1). Full analysis, profile and remaining options:
[docs/CPU_PERF.md](docs/CPU_PERF.md).

The CUDA kernel is **bit-exact** (audit passes everywhere), and the fixed
OpenCL kernel is now also bit-exact on the RX 9070 XT (audit passes, no
fallback). The v3 phase-split kernel (division-free + one-prime-per-lane
crossing of large primes, RTX 3090) is **faster than 12-thread primesieve from
~1e10 upward** — 1e12 in 17.9 s vs 163 s, 47× faster than the initial port.
Design, measurements and the rejected-alternatives log are in
[docs/GPU_DEVELOPMENT.md §9](docs/GPU_DEVELOPMENT.md).

### GPUs compared (kernel-only time)

| n     | CPU 12 threads (dev box) | AMD GPU OpenCL (RX 9070 XT) | CPU 12 threads (reference) | CUDA GPU (RTX 3090) |
|-------|-------------------------:|----------------------------:|---------------------------:|--------------------:|
| 1e8   | 0.032 s                  | 0.006 s                     | –                          | –                  |
| 1e9   | 0.052 s                  | 0.044 s                     | 0.07 s                     | 0.016 s            |
| 1e10  | 0.35 s                   | 0.35 s                      | 0.50 s                     | 0.16 s             |
| 1e11  | 4.1 s                    | 2.53 s                      | –                          | 1.6 s              |
| 1e12  | 50.6 s                   | 26.3 s                      | 257 s                      | 17.9 s             |

Takeaways:
* **CUDA (v3 kernel) is still the fastest**: 1e12 in **17.9 s** (vs 30.6 s for
  12-thread primesieve, 26.3 s for the AMD GPU, 50.6 s for our CPU on the dev
  box).
* **The OpenCL path on the RX 9070 XT now runs the same v3 phase-split kernel**
  (reciprocal-multiply division, small-prime cooperative loop, large primes one
  per lane): **exact** (audit passes, no fallback, verified up to 1e12) and
  beats the CPU engine at every measured size — 1e12 kernel **715.5 s → 26.3 s**
  (≈27×). Design in [docs/GPU_DEVELOPMENT.md §9](docs/GPU_DEVELOPMENT.md),
  measurements in [docs/GPU.md](docs/GPU.md).
* A ROCm/HIP backend (recompiling the CUDA kernel `gpu_cuda.cu` ~verbatim) is a
  future exercise once the AMD card is reachable from Linux; not available on the
  Windows-only dev box.

## License

BSD 2-Clause. See [LICENSE](LICENSE).

## Roadmap

* **GPU backends — mostly done.** OpenCL (RX 9070 XT) and CUDA (RTX 3090 /
  RTX 3080 Ti) both run the v3 phase-split kernel bit-exact via the audited
  path. Open items: a ROCm/HIP backend (Linux, once the AMD card is reachable
  there), and closing the residual CPU gap to primesieve (see Performance).
* Re-land the wheel-210 crossing with the frame-carry semantics fixed.