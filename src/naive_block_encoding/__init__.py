"""Generate block-encoding circuits by dense unitary dilation."""

from .encoding import BlockEncoding, block_encode
from .matrix_generation import random_matrix

__all__ = ["BlockEncoding", "block_encode", "random_matrix"]
