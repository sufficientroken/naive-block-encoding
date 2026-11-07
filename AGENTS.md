# Repository instructions

## Language

All authored repository content must be in English. This includes documentation,
code comments, docstrings, CLI help and error messages, examples, test names and
descriptions, and configuration comments. The language used in a chat does not
change the repository language. Mathematical notation is allowed.

## Implementation

This project implements a dense unitary-dilation baseline for block encoding in
Qiskit. Keep the normalization, zero padding, most-significant ancilla ordering,
and global phase explicit. Preserve exact phase when checking circuit operators;
equivalence up to global phase is insufficient for the encoded block.

## Validation

Use `python -m pytest`, `python -m ruff check .`, and
`python -m ruff format --check .` for relevant changes. Run
`python examples/basic.py` when changing the public API or the example.
After changing the README, regenerate package metadata before distributing a
wheel or source archive so that the packaged description matches the repository.
