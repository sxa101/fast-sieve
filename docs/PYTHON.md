# Python API

The Python package is a small, dependency-free `ctypes` wrapper around the
native C API. It does not reimplement the sieve, so Python calls retain the
same exactness and optional GPU fallback as the C and command-line interfaces.

## Build the native library

On Linux, build a shared CPU library from the repository root:

```sh
gcc -O3 -mavx2 -fopenmp -fPIC -shared fastsieve.c gpu.c -o libfastsieve.so
```

On Windows, use the Visual Studio developer prompt and export a DLL:

```bat
cl /LD /O2 /arch:AVX2 /Oi /openmp fastsieve.c gpu.c /Fe:fastsieve.dll
```

Set `FASTSIEVE_LIBRARY` when the library is not next to the Python package:

```sh
FASTSIEVE_LIBRARY=/path/to/libfastsieve.so python -m pip install .
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
