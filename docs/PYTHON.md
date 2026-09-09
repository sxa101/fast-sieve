# Python API

The package lives in `python/` (it is kept out of the repository root so it
cannot collide with the `fastsieve` CLI binary the C build produces). It is a
small, dependency-free `ctypes` wrapper around the native C API. It does not
reimplement the sieve, so Python calls retain the same exactness and optional
GPU fallback as the C and command-line interfaces.

## Build the native library

Build a shared library from the repository root (the engine has no Python
dependencies):

```sh
# Linux - CPU-only shared library
gcc -O3 -mavx2 -fopenmp -fPIC -shared fastsieve.c gpu.c -o libfastsieve.so
```

On Windows, use a Visual Studio developer prompt and export a DLL with a
`fastsieve.def` that lists the seven `fastsieve_*` entry points:

```bat
cl /LD /O2 /arch:AVX2 /Oi /openmp /DFASTSIEVE_NO_MAIN fastsieve.c gpu.c /Fe:fastsieve.dll /link /DEF:fastsieve.def
```

## Install and load

Install the Python package from `python/`:

```sh
python -m pip install ./python
```

The native library is looked up in this order: the `FASTSIEVE_LIBRARY`
environment variable, the package directory, then its parent. The repo-root
build output is not next to the package, so point at it explicitly, or copy
the library next to the package:

```sh
FASTSIEVE_LIBRARY=/path/to/libfastsieve.so python -m pip install ./python
# or: cp libfastsieve.so python/fastsieve/
```

## Usage

```python
import fastsieve

fastsieve.pi(1_000_000)                 # 78498
fastsieve.count(10, 30)                 # 6
fastsieve.is_prime(97)                  # True
fastsieve.nth_prime(1000)               # 7919
list(fastsieve.generate(10, 30))        # [11, 13, 17, 19, 23, 29]

fastsieve.pi(
    1_000_000_000,
    fastsieve.Config(threads=12, use_gpu=True),
)
```

`FASTSIEVE_MAX_N` is exported for callers that need to validate input before a
long-running operation. Native range, argument, and allocation errors are
raised as `FastSieveError` or `MemoryError`.
