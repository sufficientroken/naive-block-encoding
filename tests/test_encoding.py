"""Behavioral tests of the block encoded by the generated Qiskit circuit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from naive_block_encoding import BlockEncoding, block_encode


def assert_unitary(circuit: QuantumCircuit, atol: float = 1e-10) -> None:
    unitary = Operator(circuit).data
    identity = np.eye(unitary.shape[0])
    assert_allclose(unitary.conj().T @ unitary, identity, atol=atol, rtol=0)
    assert_allclose(unitary @ unitary.conj().T, identity, atol=atol, rtol=0)


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[1.0, 0.5], [0.5, -2.0]]),
        np.array([[1 + 2j, 3 - 1j], [-0.25j, -2 + 0.5j]]),
        np.array([[0, 1j], [0, 0]]),
        np.eye(4, dtype=complex),
    ],
    ids=["hermitian", "complex-nonhermitian", "rank-deficient", "unitary"],
)
def test_generated_unitary_encodes_input(matrix: np.ndarray) -> None:
    original = matrix.copy()
    encoding = block_encode(matrix)

    assert isinstance(encoding, BlockEncoding)
    assert isinstance(encoding.circuit, QuantumCircuit)
    assert encoding.alpha == pytest.approx(np.linalg.norm(matrix, ord=2))
    assert encoding.input_dimension == matrix.shape[0]
    assert encoding.padded_dimension == matrix.shape[0]
    assert encoding.circuit.num_qubits == encoding.num_system_qubits + 1
    assert encoding.ancilla_qubit == encoding.num_system_qubits
    assert encoding.circuit.num_clbits == 0
    assert_allclose(encoding.encoded_block(), matrix / encoding.alpha, atol=1e-12)
    assert_allclose(matrix, original, atol=0, rtol=0)
    assert_unitary(encoding.circuit)


def test_non_power_of_two_dimension_is_zero_padded() -> None:
    matrix = np.array([[1, 2j, 3], [0.5, -2, 1j], [2 - 1j, 0, 0.25]])
    encoding = block_encode(matrix)
    expected = np.zeros((4, 4), dtype=complex)
    expected[:3, :3] = matrix / encoding.alpha

    assert encoding.input_dimension == 3
    assert encoding.padded_dimension == 4
    assert encoding.num_system_qubits == 2
    assert encoding.ancilla_qubit == 2
    assert encoding.circuit.num_qubits == 3
    assert_allclose(encoding.encoded_block(), expected, atol=1e-12)
    assert_unitary(encoding.circuit)


@pytest.mark.parametrize("value", [2.0, -3.0, 0.25j, 1 + 2j])
def test_scalar_matrix_uses_only_the_ancilla(value: complex) -> None:
    encoding = block_encode([[value]])

    assert encoding.input_dimension == encoding.padded_dimension == 1
    assert encoding.num_system_qubits == 0
    assert encoding.ancilla_qubit == 0
    assert encoding.circuit.num_qubits == 1
    assert encoding.alpha == pytest.approx(abs(value))
    assert_allclose(encoding.encoded_block(), [[value / abs(value)]], atol=1e-12)
    assert_unitary(encoding.circuit)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_zero_matrix_has_finite_normalization(dimension: int) -> None:
    encoding = block_encode(np.zeros((dimension, dimension)))

    assert encoding.alpha == 1.0
    assert_allclose(encoding.encoded_block(), 0, atol=1e-12)
    assert_unitary(encoding.circuit)


def test_explicit_normalization_is_preserved() -> None:
    matrix = np.array([[1j, 2], [0, -1]])
    encoding = block_encode(matrix, alpha=5.0)

    assert encoding.alpha == 5.0
    assert_allclose(encoding.encoded_block(), matrix / 5, atol=1e-12)
    assert_unitary(encoding.circuit)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_finite_scales_preserve_the_normalized_matrix(scale: float) -> None:
    reference = np.array([[1 + 1j, -2], [0.5j, 1 - 0.25j]])
    reference_norm = np.linalg.norm(reference, ord=2)
    encoding = block_encode(reference * scale)

    assert np.isfinite(encoding.alpha)
    assert encoding.alpha / scale == pytest.approx(reference_norm)
    assert_allclose(encoding.encoded_block(), reference / reference_norm, atol=1e-12)
    assert_unitary(encoding.circuit)


def test_subnormal_complex_scalar_is_normalized_to_a_contraction() -> None:
    tiny = 5e-324
    encoding = block_encode([[complex(tiny, tiny)]])
    block = encoding.encoded_block()

    # sqrt(2) * tiny lies between representable floats; normalization must round up.
    assert encoding.alpha == 2 * tiny
    assert np.linalg.norm(block, ord=2) <= 1
    assert_allclose(block, [[complex(tiny / encoding.alpha, tiny / encoding.alpha)]])
    assert encoding.alpha * block[0, 0].real == tiny
    assert encoding.alpha * block[0, 0].imag == tiny
    assert_unitary(encoding.circuit)


def test_subnormal_alpha_below_true_complex_norm_is_rejected() -> None:
    tiny = 5e-324
    with pytest.raises(ValueError):
        block_encode([[complex(tiny, tiny)]], alpha=tiny)


def test_numpy_complex_normalization_is_rejected() -> None:
    with pytest.raises(ValueError):
        block_encode(np.eye(2), alpha=np.complex128(1 + 1j))


def test_ancilla_zero_branch_applies_the_matrix_to_a_state() -> None:
    matrix = np.array([[1 + 1j, 2, 0], [0, -1j, 3], [1, 0.5, -2]])
    encoding = block_encode(matrix, alpha=6.0)
    psi = np.array([1, -2j, 2], dtype=complex) / 3
    padded_psi = np.pad(psi, (0, encoding.padded_dimension - len(psi)))
    initial = Statevector(np.concatenate([padded_psi, np.zeros_like(padded_psi)]))

    final = initial.evolve(encoding.circuit)
    expected = np.pad(matrix @ psi / encoding.alpha, (0, 1))
    success_probability = np.linalg.norm(expected) ** 2

    assert_allclose(final.data[: encoding.padded_dimension], expected, atol=1e-12)
    assert final.probabilities([encoding.ancilla_qubit])[0] == pytest.approx(
        success_probability, abs=1e-12
    )


def test_synthesized_circuit_preserves_the_block_including_global_phase() -> None:
    matrix = np.array([[1 + 2j, 0.5], [-1j, 2 - 1j]])
    encoding = block_encode(matrix)

    synthesized = encoding.transpile(basis_gates=("u", "cx"), optimization_level=1)

    assert isinstance(synthesized, QuantumCircuit)
    assert synthesized.num_qubits == encoding.circuit.num_qubits
    assert set(synthesized.count_ops()) <= {"u", "cx"}
    # Operator.equiv allows a global phase; block encoding needs the actual phase.
    actual = Operator(synthesized).data
    assert_allclose(actual, Operator(encoding.circuit).data, atol=1e-8, rtol=0)
    assert_allclose(actual[:2, :2], matrix / encoding.alpha, atol=1e-8, rtol=0)
    assert_unitary(synthesized, atol=1e-8)


@pytest.mark.parametrize(
    "matrix",
    [
        [],
        np.empty((0, 0)),
        1,
        [1, 2],
        [[1, 2, 3], [4, 5, 6]],
        np.ones((2, 2, 2)),
        [[np.nan]],
        [[np.inf]],
        [[complex(1, np.inf)]],
        [["not-a-number"]],
        [[1, 2], [3]],
    ],
    ids=[
        "empty-list",
        "empty-matrix",
        "scalar",
        "vector",
        "rectangular",
        "tensor",
        "nan",
        "infinity",
        "complex-infinity",
        "nonnumeric",
        "ragged",
    ],
)
def test_invalid_matrices_raise_value_error(matrix: object) -> None:
    with pytest.raises(ValueError):
        block_encode(matrix)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, -np.inf, 0.5])
def test_invalid_normalization_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError):
        block_encode(np.eye(2), alpha=alpha)
