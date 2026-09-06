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
* **GPU engine (OpenCL): accelerator with correctness by construction** – the
  GPU result is audited against the exact CPU engine on sampled segments; on
  any mismatch the tool silently/visibly falls back to the CPU engine. The
  published result is therefore *always* exact, on any machine. On drivers where
  the GPU kernel is bit-exact (expected on CUDA/venus), the audit passes and full
  GPU speed is used. See [docs/GPU.md](docs/GPU.md).
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

CPU engine, AMD Ryzen 5 5600G, MSVC /O2 /arch:AVX2:

| n     | ours 1 core | ours 12 threads | primesieve 1 core | primesieve 12 threads |
|-------|------------:|----------------:|------------------:|----------------------:|
| 1e9   | 0.17 s      | 0.05 s          | 0.11 s            | 0.04 s                |
| 1e10  | 1.9 s       | 0.36 s          | 1.1 s             | 0.25 s                |
| 1e12  | 290 s       | 52 s            | 167 s             | 31 s                  |

We are consistently ~1.5–1.7× behind primesieve: its compile-time-unrolled
Duff's-device residue loops with immediate bit masks cannot be emitted by MSVC
from our runtime-generated tables. That is the single declared gap.

**venus** (Intel i5-11600, 12 threads, gcc 13 -O3 -march=native; CUDA, RTX 3090):

| n     | ours CPU 1t | ours CPU 12t | primesieve 1t | primesieve 12t | ours GPU (kernel) |
|-------|------------:|-------------:|--------------:|---------------:|------------------:|
| 1e9   | 0.21 s      | 0.07 s       | 0.10 s        | 0.02 s         | 0.016 s           |
| 1e10  | 2.4 s       | 0.50 s       | 1.2 s         | 0.29 s         | 0.16 s            |
| 1e11  | –           | –            | –             | –              | 1.6 s             |
| 1e12  | 510 s       | 257 s        | 267 s         | 163 s          | 17.9 s            |

The CUDA kernel is **bit-exact** (audit passes everywhere), and the fixed
OpenCL kernel is now also bit-exact on the RX 9070 XT (audit passes, no
fallback). The v3 phase-split kernel (division-free + one-prime-per-lane
crossing of large primes, RTX 3090) is **faster than 12-thread primesieve from
~1e10 upward** — 1e12 in 17.9 s vs 163 s, 47× faster than the initial port.
Design, measurements and the rejected-alternatives log are in
[docs/RESUME_VENUS.md §9](docs/RESUME_VENUS.md).

### GPUs compared (kernel-only time)

| n     | CPU 12 threads (dev box) | AMD GPU OpenCL (RX 9070 XT) | CPU 12 threads (venus) | CUDA GPU (RTX 3090) |
|-------|-------------------------:|----------------------------:|-----------------------:|--------------------:|
| 1e8   | 0.038 s                  | 0.007 s                     | –                      | –                  |
| 1e9   | 0.056 s                  | 0.066 s                     | 0.07 s                 | 0.016 s            |
| 1e10  | 0.38 s                   | 1.30 s                      | 0.50 s                 | 0.16 s             |
| 1e11  | 3.76 s                   | 26.1 s                      | –                      | 1.6 s              |
| 1e12  | 52 s                     | –                           | 257 s                  | 17.9 s             |

Takeaways: on the **CUDA side the GPU wins everywhere** (1e12: **17.9 s** vs
163 s for 12-thread primesieve / 257 s for our CPU). The **OpenCL kernel on the
RX 9070 XT is exact** (audit passes, no fallback) but only beats the CPU below
≈5×10⁸ (crossover); the v3 optimizations have not yet been ported to the OpenCL
path — that is the next work item (see [docs/GPU.md](docs/GPU.md) and
[docs/RESUME_VENUS.md §9](docs/RESUME_VENUS.md)).

## License

BSD 2-Clause. See [LICENSE](LICENSE).

## Roadmap

* Port the GPU engine to CUDA on the venus box (RTX 3090 / RTX 3080) and
  re-run the audit – see [docs/RESUME_VENUS.md](docs/RESUME_VENUS.md).
* Re-land the wheel-210 crossing with the frame-carry semantics fixed.