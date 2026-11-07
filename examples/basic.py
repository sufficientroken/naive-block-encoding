"""Build, synthesize, and verify a complex block-encoding circuit."""

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from naive_block_encoding import block_encode


def main() -> None:
    matrix = np.array([[1.0, 0.5j], [0.25, -0.5]], dtype=complex)
    encoding = block_encode(matrix)
    dimension = encoding.padded_dimension

    print(f"alpha: {encoding.alpha:.12g}")
    print(f"system qubits: {encoding.num_system_qubits}")
    print(f"ancilla qubit index: {encoding.ancilla_qubit}")
    print("\nUnitary-gate circuit:")
    print(encoding.circuit.draw(output="text"))

    recovered = encoding.alpha * encoding.encoded_block()
    np.testing.assert_allclose(recovered, matrix, atol=1e-12, rtol=1e-12)
    print("\nRecovered matrix:", recovered, sep="\n")
    print(f"reconstruction error: {np.linalg.norm(recovered - matrix):.3e}")

    decomposed = encoding.transpile()
    np.testing.assert_allclose(
        Operator(decomposed).data,
        Operator(encoding.circuit).data,
        atol=1e-10,
        rtol=1e-10,
    )
    print("\nCircuit decomposed into u and cx:")
    print(decomposed.draw(output="text", fold=100))
    print("gate counts:", dict(decomposed.count_ops()))

    # Statevector indices 0..N-1 have the most significant (ancilla) bit set to 0.
    psi = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
    initial = np.concatenate([psi, np.zeros(dimension, dtype=complex)])
    final = Statevector(initial).evolve(encoding.circuit).data
    postselected = final[:dimension]
    expected = matrix @ psi / encoding.alpha
    np.testing.assert_allclose(postselected, expected, atol=1e-12, rtol=1e-12)
    probability = float(np.vdot(postselected, postselected).real)
    print(f"\nP(ancilla = 0): {probability:.12g}")


if __name__ == "__main__":
    main()
