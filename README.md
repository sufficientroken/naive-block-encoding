# Naive Block Encoding with Qiskit

Generate a Qiskit circuit whose upper-left block is a normalized input matrix. The implementation accepts finite, nonempty, real or complex square matrices, including non-Hermitian and singular matrices. It uses one ancilla qubit and pads the matrix with zeros when its dimension is not a power of two.

The construction is deliberately simple: form a dense unitary dilation on a classical computer, then use Qiskit's [`UnitaryGate`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UnitaryGate) to create the circuit. Optional synthesis produces elementary gates. This repository is intended for small examples, numerical verification, and comparisons with more efficient constructions.

## Installation

Requirements: Python 3.10 or later, Qiskit 2.2 or later within the 2.x series, and NumPy 1.26 or later within the 1.x or 2.x series. No IBM Quantum account or hardware connection is required.

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/basic.py
```

On macOS or Linux, replace the activation command with `source .venv/bin/activate`. If PowerShell prevents script activation, use `.\.venv\Scripts\python.exe` directly in place of `python` after creating the environment.

## Python API

```python
import numpy as np

from naive_block_encoding import block_encode

matrix = np.array([[1.0, 0.5j], [0.25, -0.5]], dtype=complex)
encoding = block_encode(matrix)

print("Normalization:", encoding.alpha)
print("Ancilla index:", encoding.ancilla_qubit)
print(encoding.circuit.draw(output="text"))

# Extract the block with the ancilla in state zero at both input and output.
np.testing.assert_allclose(
    encoding.encoded_block(), matrix / encoding.alpha, atol=1e-12
)

# Create a new circuit expressed in u and cx gates.
synthesized = encoding.transpile()
print(synthesized.draw(output="text"))
```

`block_encode(matrix, alpha=None)` returns a `BlockEncoding` object:

| Attribute or method | Meaning |
| --- | --- |
| `circuit` | A `QuantumCircuit` containing one `UnitaryGate` |
| `alpha` | The normalization coefficient |
| `input_dimension` | The original matrix dimension, `d` |
| `padded_dimension` | The next power of two at least as large as `d`, called `N` |
| `num_system_qubits` | The number of system qubits, `log2(N)` |
| `ancilla_qubit` | The index of the sole ancilla, `log2(N)` |
| `encoded_block()` | The `N x N` upper-left block of the actual circuit operator |
| `transpile(...)` | A new circuit synthesized into the requested gate basis |

The default normalization is the spectral norm of the input matrix, or `1.0` for the zero matrix. You can supply a larger coefficient, for example `block_encode(matrix, alpha=2.0)`. An explicit coefficient must be real, finite, positive, and at least the spectral norm, allowing floating-point roundoff.

Automatic normalization rounds upward by one representable floating-point value when needed to keep the normalized matrix a contraction. If the spectral norm exceeds the range of `float64`, rescale the input matrix before encoding it. Invalid matrix shapes, non-finite entries, and invalid coefficients raise `ValueError`.

`transpile()` defaults to `basis_gates=("u", "cx")` and `optimization_level=1`. It preserves the global phase and leaves the original circuit unchanged. It does not perform backend-specific placement or routing; use Qiskit's backend-aware transpilation before hardware execution.

For dimensions other than powers of two, `encoded_block()` includes the zero padding. To recover the original matrix, use:

```python
d = encoding.input_dimension
recovered = encoding.alpha * encoding.encoded_block()[:d, :d]
```

The circuit contains no initialization or measurement instructions, so it can be composed with other circuits. Although the `BlockEncoding` fields are frozen, the underlying Qiskit circuit remains mutable.

## Command-line usage

Input files contain a JSON matrix. Write complex entries as Python-style strings:

```json
[
  [1.0, "0.5j"],
  [0.25, -0.5]
]
```

The repository includes this matrix in `examples/matrix.json`.

```powershell
# Print the circuit diagram.
python -m naive_block_encoding --matrix examples/matrix.json

# Save a diagram after synthesis into u and cx gates.
python -m naive_block_encoding --matrix examples/matrix.json --decompose --output output/circuit.txt

# Save OpenQASM 2; synthesis into u3 and cx gates is automatic.
python -m naive_block_encoding --matrix examples/matrix.json --format qasm --output output/circuit.qasm

# Save the circuit in Qiskit's QPY format.
python -m naive_block_encoding --matrix examples/matrix.json --format qpy --output output/circuit.qpy
```

The installed `naive-block-encoding` command accepts the same arguments as `python -m naive_block_encoding`.

| Option | Behavior |
| --- | --- |
| `--matrix PATH` | Read a JSON square matrix; required |
| `--alpha FLOAT` | Set the normalization coefficient |
| `--format text\|qasm\|qpy` | Select the output format; defaults to `text` |
| `--output PATH` | Write to a file; required for `qpy` |
| `--decompose` | Synthesize text or QPY output into `u` and `cx` gates |
| `--optimization-level 0\|1\|2\|3` | Set the synthesis optimization level; defaults to `1` |

Text and QASM output go to standard output when `--output` is omitted. Parent directories are created automatically. Every file export writes a JSON sidecar at `<output path>.json`, such as `output/circuit.qasm.json`. It records the normalization, dimensions, ancilla index, global phase, and circuit statistics. The CLI also prints this metadata to standard error. Output paths that would overwrite the input matrix are rejected.

## Reloading circuits and preserving phase

QPY preserves the circuit's global phase and metadata, making it convenient for reuse in Qiskit:

```python
from qiskit import qpy

with open("output/circuit.qpy", "rb") as stream:
    circuit = qpy.load(stream)[0]

alpha = circuit.metadata["alpha"]
```

[OpenQASM 2 cannot represent a whole-program global phase](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qasm2#openqasm-2-phase-conventions). Restore that phase after loading a QASM export to preserve the encoded block, including its phase:

```python
import json
from pathlib import Path

from qiskit import qasm2

circuit = qasm2.load("output/circuit.qasm")
metadata = json.loads(Path("output/circuit.qasm.json").read_text(encoding="utf-8"))
circuit.global_phase += metadata["global_phase"]
alpha = metadata["alpha"]
```

QASM exports use `u3` and `cx` for compatibility with standard `qelib1.inc`. The first line also contains the metadata as a JSON comment prefixed with `// naive-block-encoding: `, so exports written to standard output retain the phase and normalization information.

## Mathematical construction

For an input matrix $A$ of dimension $d$, pad with zeros to obtain $A_{\mathrm{pad}}$ of dimension $N = 2^{\lceil \log_2 d \rceil}$. Choose $\alpha \geq \lVert A \rVert_2$ and define $B = A_{\mathrm{pad}}/\alpha$. The circuit implements the unitary dilation

$$
U = \begin{pmatrix}
B & \sqrt{I - BB^\dagger} \\
\sqrt{I - B^\dagger B} & -B^\dagger
\end{pmatrix}.
$$

The square roots are the positive semidefinite matrix square roots. One singular value decomposition supplies both defect operators, using the left and right singular vectors with the same singular values. This construction applies to complex, non-Hermitian matrices.

Preparing and projecting the ancilla in state zero gives

$$
(\langle 0| \otimes I_N) U (|0\rangle \otimes I_N)
= \frac{A_{\mathrm{pad}}}{\alpha}.
$$

In exact arithmetic, this is an $(\alpha, 1, 0)$ block encoding of $A_{\mathrm{pad}}$. The implementation has floating-point and gate-synthesis errors. For the block-encoding definition and its applications, see [Gilyen, Su, Low, and Wiebe](https://arxiv.org/abs/1806.01838).

Qiskit uses lower qubit indices for less significant bits. The system occupies indices `0` through `n - 1`, and the ancilla occupies index `n`, where `n = log2(N)`. Consequently, `Operator(circuit).data[:N, :N]` selects the desired block. The ancilla appears on the bottom wire in the default circuit drawing. See [Qiskit's bit-ordering guide](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering).

A `1 x 1` matrix uses no system qubits and one ancilla qubit. A matrix with a dimension other than a power of two uses the padded system space.

For a normalized system state $|\psi\rangle$, the probability of observing the ancilla in state zero is

$$
p_0 = \frac{\lVert A_{\mathrm{pad}}|\psi\rangle \rVert^2}{\alpha^2}.
$$

When this probability is nonzero, the conditional system state is proportional to $A_{\mathrm{pad}}|\psi\rangle$. `examples/basic.py` verifies this relation with a Qiskit statevector.

## Cost and scope

For padded dimension `N`, dense construction requires approximately `O(N^3)` classical time and `O(N^2)` memory. Generic unitary synthesis also grows exponentially with the number of qubits. `encoded_block()` constructs a dense circuit operator and has the same small-instance scope.

The implementation does not provide sparse-access oracles, LCU constructions, or efficient state-preparation-based block encodings. Use it as a reference implementation for small matrices.

## Development and verification

Repository layout:

```text
src/naive_block_encoding/  Circuit construction, Python API, and CLI
examples/                  Runnable example and JSON input matrix
tests/                     Numerical and command-line tests
.github/workflows/         Continuous integration configuration
pyproject.toml             Package dependencies and development settings
MANIFEST.in                Source distribution contents
AGENTS.md                  Repository contribution instructions
```

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest
python examples/basic.py
```

The tests check unitary correctness, recovery of the encoded matrix, non-Hermitian complex inputs, zero padding, scalar and zero matrices, ancilla ordering, statevector postselection, phase preservation after synthesis, CLI exports, QPY and QASM round trips, invalid inputs, and extreme numerical scales.

GitHub Actions is configured to run linting, formatting checks, tests, and the example on Windows and Ubuntu with Python 3.10 and 3.12. The local verification environment uses Windows, Python 3.12.14, Qiskit 2.6.2, and NumPy 2.5.4.

Repository documentation, comments, docstrings, examples, and project-authored messages are written in English.
