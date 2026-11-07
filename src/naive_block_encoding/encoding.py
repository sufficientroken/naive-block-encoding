"""Construct one-ancilla block encodings through dense unitary dilation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


@dataclass(frozen=True)
class BlockEncoding:
    """A circuit and the normalization needed to interpret its encoded block.

    The last (most significant) qubit is the sole ancilla. The circuit contains
    no initialization or measurement, so it can be composed with other circuits.
    ``frozen`` protects metadata; the Qiskit circuit itself remains mutable.
    """

    circuit: QuantumCircuit
    alpha: float
    input_dimension: int
    padded_dimension: int

    @property
    def num_system_qubits(self) -> int:
        """Return the number of qubits used for the padded input space."""
        return self.padded_dimension.bit_length() - 1

    @property
    def ancilla_qubit(self) -> int:
        """Return the index of the most significant qubit, the sole ancilla."""
        return self.num_system_qubits

    def encoded_block(self) -> NDArray[np.complex128]:
        """Extract <0|U|0> from the actual circuit, including zero padding.

        This forms the full dense Operator and is intended for small examples.
        """
        return Operator(self.circuit).data[: self.padded_dimension, : self.padded_dimension].copy()

    def transpile(
        self,
        *,
        basis_gates: Sequence[str] = ("u", "cx"),
        optimization_level: int = 1,
    ) -> QuantumCircuit:
        """Synthesize a new circuit into a gate basis, preserving global phase.

        No backend routing or hardware execution is performed. Generic synthesis
        is expensive: this implementation is a small-matrix reference baseline.
        """
        return transpile(
            self.circuit,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
            seed_transpiler=0,
        )


def block_encode(matrix: ArrayLike, *, alpha: float | None = None) -> BlockEncoding:
    """Construct a dense block encoding of a finite, nonempty square matrix.

    Zero-pad A to dimension N = the next power of two. With B = A_pad / alpha,
    set B_adj = B.conj().T and construct the block matrix

        U = [[B,                  sqrt(I - B @ B_adj)],
             [sqrt(I - B_adj @ B),             -B_adj]].

    Projecting the ancilla onto |0> at both input and output selects B, so
    U[:N, :N] = A_pad / alpha to floating-point precision.

    The input may be complex and need not be Hermitian or invertible. The
    generated circuit has no measurements or state preparation instructions.

    One SVD constructs both defect operators consistently. Classical cost is
    O(N^3) time and O(N^2) space, before generic Qiskit gate synthesis.

    Args:
        matrix: A finite, nonempty square array of real or complex numbers.
        alpha: Positive normalization at least ||A||_2, within roundoff tolerance.
            Defaults to ||A||_2, rounded up by one representable float when
            needed to retain a contraction. Defaults to 1 for the zero matrix.

    Returns:
        The circuit, its normalization, and the original and padded dimensions.

    Raises:
        ValueError: Invalid matrix or normalization, or a norm that cannot be
            represented as a finite float64 value.
    """
    try:
        array = np.array(matrix, dtype=np.complex128, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrix must contain numeric entries") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a nonempty square two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("matrix entries must be finite")

    dimension = array.shape[0]
    padded_dimension = 1 << (dimension - 1).bit_length()
    padded = np.zeros((padded_dimension, padded_dimension), dtype=np.complex128)
    padded[:dimension, :dimension] = array

    # Scale real/imaginary parts separately to avoid overflow in abs(complex)
    # and underflow in complex division for very small input magnitudes.
    scale = float(max(np.max(np.abs(array.real)), np.max(np.abs(array.imag))))
    scaled = padded.copy()
    if scale:
        scaled.real /= scale
        scaled.imag /= scale
    left, singular_values, right_adjoint = np.linalg.svd(scaled, full_matrices=False)
    norm = scale * float(singular_values[0])
    if not np.isfinite(norm):
        raise ValueError("matrix spectral norm exceeds the float64 range; rescale the matrix")

    if alpha is None:
        normalization = norm if norm > 0 else 1.0
        # The norm of subnormal inputs can round down by a substantial fraction.
        # Move up one representable float when needed to retain a contraction.
        if norm and singular_values[0] * (scale / normalization) > 1.0:
            with np.errstate(over="ignore"):
                rounded_up = float(np.nextafter(normalization, np.inf))
            if np.isfinite(rounded_up):
                normalization = rounded_up
    else:
        if np.iscomplexobj(alpha):
            raise ValueError("alpha must be a finite positive real number")
        try:
            normalization = float(alpha)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha must be a finite positive real number") from exc
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError("alpha must be a finite positive real number")
        if normalization < norm and (norm - normalization) / norm > 1e-14:
            raise ValueError(f"alpha ({normalization:g}) must be at least ||A||_2 ({norm:g})")

    normalized_singular_values = singular_values * (scale / normalization)
    if normalized_singular_values[0] > 1.0 + 1e-14:
        raise ValueError("alpha is too small to make the normalized matrix a contraction")
    # Computing B componentwise also handles subnormal (very small) real inputs.
    block = padded.copy()
    block.real /= normalization
    block.imag /= normalization
    defect_values = np.sqrt(np.maximum(0.0, 1.0 - normalized_singular_values**2))
    left_defect = (left * defect_values) @ left.conj().T
    right = right_adjoint.conj().T
    right_defect = (right * defect_values) @ right_adjoint
    unitary = np.block([[block, left_defect], [right_defect, -block.conj().T]])

    num_system_qubits = padded_dimension.bit_length() - 1
    ancilla = AncillaRegister(1, "anc")
    if num_system_qubits:
        system = QuantumRegister(num_system_qubits, "sys")
        circuit = QuantumCircuit(system, ancilla, name="block_encoding")
    else:
        circuit = QuantumCircuit(ancilla, name="block_encoding")
    circuit.append(UnitaryGate(unitary, label="U_A"), circuit.qubits)
    circuit.metadata = {
        "alpha": normalization,
        "input_dimension": dimension,
        "padded_dimension": padded_dimension,
        "ancilla_qubit": num_system_qubits,
    }
    return BlockEncoding(circuit, normalization, dimension, padded_dimension)
