"""Generate reproducible pseudorandom matrices from two text sources."""

from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
from numpy.typing import NDArray

_SEED_DOMAIN = b"naive-block-encoding/random-matrix/v1\0"
_SOURCE_NAMES = ("random_strings_a.txt", "random_strings_b.txt")


def random_matrix(
    dimension: int,
    *,
    index: int = 0,
    complex_entries: bool = False,
    sources: tuple[str, str] | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Return a square pseudorandom matrix seeded by two text sources.

    By default, use the bundled ``data/random_strings_a.txt`` and
    ``data/random_strings_b.txt`` files in that order. ``sources`` can instead
    supply a tuple of two nonempty, non-whitespace-only strings. Source text is
    data only: no instructions or expressions in it are executed.

    Normalize CRLF and CR line endings to LF, preserving all other text,
    including trailing newlines. Seed construction uses SHA-256, starting with
    ``b"naive-block-encoding/random-matrix/v1\\0"``. Append each UTF-8 source in
    order, preceded by its byte length as an unsigned 8-byte big-endian integer.
    Append the ASCII decimal ``index`` with the same length framing. Interpret
    the digest as a big-endian integer and initialize a local NumPy PCG64
    generator with it. No global random state is read or changed.

    Real entries are generated as ``2 * rng.random((dimension, dimension)) - 1``
    and lie in ``[-1, 1)``. For complex output, draw the imaginary components
    independently using the next samples and the same formula. Return float64
    or complex128 entries without normalization or padding. The same arguments
    reproduce the same matrix; changing ``index`` selects another pseudorandom
    matrix. This generator is intended for numerical examples, not cryptography.

    Args:
        dimension: Positive integer matrix dimension. NumPy integers are allowed;
            booleans are rejected.
        index: Nonnegative integer selecting a reproducible matrix. NumPy
            integers are allowed; booleans are rejected.
        complex_entries: Whether to generate independent imaginary components.
        sources: Optional pair of text sources replacing the bundled files.

    Raises:
        ValueError: Invalid dimension, index, complex_entries, or source strings.
        OSError: A bundled source file cannot be read.
    """
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)):
        raise ValueError("dimension must be a positive integer")
    if dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be a nonnegative integer")
    if index < 0:
        raise ValueError("index must be a nonnegative integer")
    if not isinstance(complex_entries, bool):
        raise ValueError("complex_entries must be a boolean")

    if sources is None:
        data = resources.files("naive_block_encoding").joinpath("data")
        sources = tuple(data.joinpath(name).read_text(encoding="utf-8") for name in _SOURCE_NAMES)
    if not isinstance(sources, tuple) or len(sources) != 2:
        raise ValueError("sources must be a tuple of exactly two non-whitespace-only strings")
    if any(not isinstance(source, str) or not source.strip() for source in sources):
        raise ValueError("sources must be a tuple of exactly two non-whitespace-only strings")

    digest = hashlib.sha256(_SEED_DOMAIN)
    for source in sources:
        encoded = source.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    encoded_index = str(int(index)).encode("ascii")
    digest.update(len(encoded_index).to_bytes(8, "big"))
    digest.update(encoded_index)

    generator = np.random.Generator(np.random.PCG64(int.from_bytes(digest.digest(), "big")))
    shape = (int(dimension), int(dimension))
    matrix = 2 * generator.random(shape) - 1
    if complex_entries:
        matrix = matrix + 1j * (2 * generator.random(shape) - 1)
    return matrix
