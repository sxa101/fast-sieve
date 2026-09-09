"""Python bindings for the fastsieve prime-counting engine.

The native library is loaded lazily. Set ``FASTSIEVE_LIBRARY`` to its path, or
place a platform shared library named ``fastsieve`` next to this package.
"""

from ._api import (
    FASTSIEVE_MAX_N,
    Config,
    FastSieveError,
    LibraryNotFoundError,
    count,
    generate,
    is_prime,
    nth_prime,
    pi,
    version,
)

__all__ = [
    "FASTSIEVE_MAX_N",
    "Config",
    "FastSieveError",
    "LibraryNotFoundError",
    "count",
    "generate",
    "is_prime",
    "nth_prime",
    "pi",
    "version",
]
