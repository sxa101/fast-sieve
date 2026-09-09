"""ctypes bindings for the fastsieve C API."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

FASTSIEVE_MAX_N = 8_000_000_000_000_000
FASTSIEVE_OK = 0
FASTSIEVE_ERR_RANGE = -1
FASTSIEVE_ERR_ARGS = -2
FASTSIEVE_ERR_NOMEM = -3


class FastSieveError(ValueError):
    """An argument was rejected by the native fastsieve API."""


class LibraryNotFoundError(ImportError):
    """The fastsieve shared library could not be found."""


class _Config(ctypes.Structure):
    _fields_ = [
        ("threads", ctypes.c_int),
        ("use_gpu", ctypes.c_int),
        ("sieve_bytes", ctypes.c_uint64),
        ("med_factor", ctypes.c_double),
    ]


@dataclass(frozen=True)
class Config:
    """Optional per-call tuning for the native engine.

    ``None`` (the default for every function) uses the engine defaults.
    """

    threads: int = 0
    use_gpu: bool = False
    sieve_bytes: int = 0
    med_factor: float = 0.0

    def _native(self) -> _Config:
        if self.sieve_bytes < 0:
            raise ValueError("sieve_bytes must be non-negative")
        if self.med_factor < 0:
            raise ValueError("med_factor must be non-negative")
        return _Config(
            self.threads,
            int(self.use_gpu),
            self.sieve_bytes,
            self.med_factor,
        )


def _library_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("fastsieve.dll", "libfastsieve.dll")
    if sys.platform == "darwin":
        return ("libfastsieve.dylib", "fastsieve.dylib")
    return ("libfastsieve.so", "fastsieve.so")


def _load_library() -> ctypes.CDLL:
    requested = os.environ.get("FASTSIEVE_LIBRARY")
    candidates = [Path(requested)] if requested else []
    package_dir = Path(__file__).resolve().parent
    root = package_dir.parent
    candidates.extend(package_dir / name for name in _library_names())
    candidates.extend(root / name for name in _library_names())
    for candidate in candidates:
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    searched = ", ".join(str(path) for path in candidates)
    raise LibraryNotFoundError(
        "fastsieve native library not found; build it and set "
        f"FASTSIEVE_LIBRARY (searched: {searched})"
    )


def _configure(lib: ctypes.CDLL) -> ctypes.CDLL:
    config_ptr = ctypes.POINTER(_Config)
    lib.fastsieve_init.argtypes = []
    lib.fastsieve_init.restype = None
    lib.fastsieve_version.argtypes = []
    lib.fastsieve_version.restype = ctypes.c_char_p
    lib.fastsieve_pi.argtypes = [ctypes.c_uint64, config_ptr]
    lib.fastsieve_pi.restype = ctypes.c_int64
    lib.fastsieve_count.argtypes = [ctypes.c_uint64, ctypes.c_uint64, config_ptr]
    lib.fastsieve_count.restype = ctypes.c_int64
    lib.fastsieve_isprime.argtypes = [ctypes.c_uint64, config_ptr]
    lib.fastsieve_isprime.restype = ctypes.c_int
    lib.fastsieve_nth_prime.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        config_ptr,
    ]
    lib.fastsieve_nth_prime.restype = ctypes.c_int64
    callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p)
    lib.fastsieve_generate.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        callback,
        ctypes.c_void_p,
        config_ptr,
    ]
    lib.fastsieve_generate.restype = ctypes.c_int64
    lib._fastsieve_callback_type = callback
    return lib


_native: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _native
    if _native is None:
        _native = _configure(_load_library())
        _native.fastsieve_init()
    return _native


def _config(
    config: Optional[Config],
) -> Tuple[Optional[ctypes.POINTER(_Config)], Optional[_Config]]:
    native = config._native() if config is not None else None
    return (ctypes.pointer(native) if native is not None else None), native


def _check(value: int) -> int:
    if value == FASTSIEVE_ERR_RANGE:
        raise FastSieveError("value is outside FASTSIEVE_MAX_N")
    if value == FASTSIEVE_ERR_ARGS:
        raise FastSieveError("invalid fastsieve arguments")
    if value == FASTSIEVE_ERR_NOMEM:
        raise MemoryError("fastsieve could not allocate memory")
    return value


def version() -> str:
    """Return the native engine version."""

    return _lib().fastsieve_version().decode("ascii")


def pi(n: int, config: Optional[Config] = None) -> int:
    """Return the number of primes less than or equal to ``n``."""

    cfg, _keepalive = _config(config)
    return _check(int(_lib().fastsieve_pi(n, cfg)))


def count(lo: int, hi: int, config: Optional[Config] = None) -> int:
    """Return the number of primes in the inclusive range ``[lo, hi]``."""

    cfg, _keepalive = _config(config)
    return _check(int(_lib().fastsieve_count(lo, hi, cfg)))


def is_prime(n: int, config: Optional[Config] = None) -> bool:
    """Return whether ``n`` is prime."""

    cfg, _keepalive = _config(config)
    result = int(_lib().fastsieve_isprime(n, cfg))
    return bool(_check(result))


def nth_prime(k: int, start: int = 2, config: Optional[Config] = None) -> int:
    """Return the ``k``-th prime greater than or equal to ``start``."""

    cfg, _keepalive = _config(config)
    return _check(int(_lib().fastsieve_nth_prime(k, start, cfg)))


def generate(
    lo: int,
    hi: int,
    config: Optional[Config] = None,
) -> Iterator[int]:
    """Yield primes in ascending order in the inclusive range ``[lo, hi]``."""

    native = _lib()
    cfg, _keepalive = _config(config)
    values: list[int] = []

    @native._fastsieve_callback_type
    def callback(prime: int, _user: int) -> int:
        values.append(prime)
        return 0

    result = _check(int(native.fastsieve_generate(lo, hi, callback, None, cfg)))
    if result != len(values):
        raise RuntimeError(
            f"native generator emitted {result} primes but callback received "
            f"{len(values)}"
        )
    yield from values
