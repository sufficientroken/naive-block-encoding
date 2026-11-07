"""Generate a Qiskit block-encoding circuit from a square matrix in a JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qiskit import qasm2, qpy
from qiskit.exceptions import QiskitError

from .encoding import block_encode


def main(argv: list[str] | None = None) -> int:
    """Generate a circuit and retain its normalization and global-phase metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path, help="Path to the input JSON matrix")
    parser.add_argument("--alpha", type=float, help="Positive normalization >= the spectral norm")
    parser.add_argument(
        "--format",
        choices=("text", "qasm", "qpy"),
        default="text",
        help="Output format: text diagram, OpenQASM 2, or QPY (default: text)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output path; required for the binary QPY format"
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Synthesize into u/cx gates; QASM always uses u3/cx gates",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(4),
        default=1,
        help="Transpiler optimization level used during synthesis (default: 1)",
    )
    args = parser.parse_args(argv)
    if args.format == "qpy" and args.output is None:
        parser.error("--format qpy requires --output")

    try:
        if args.output is not None:
            metadata_path = Path(str(args.output) + ".json")
            if args.matrix.resolve() in (args.output.resolve(), metadata_path.resolve()):
                raise ValueError("output or its metadata sidecar would overwrite the input matrix")
        matrix = json.loads(args.matrix.read_text(encoding="utf-8-sig"))
        result = block_encode(matrix, alpha=args.alpha)
        if args.format == "qasm":
            # u3 is in standard qelib1.inc, avoiding nonstandard parser extensions.
            circuit = result.transpile(
                basis_gates=("u3", "cx"), optimization_level=args.optimization_level
            )
        elif args.decompose:
            circuit = result.transpile(optimization_level=args.optimization_level)
        else:
            circuit = result.circuit

        metadata = {
            "alpha": result.alpha,
            "input_dimension": result.input_dimension,
            "padded_dimension": result.padded_dimension,
            "num_system_qubits": result.num_system_qubits,
            "ancilla_qubit": result.ancilla_qubit,
            "num_ancillas": 1,
            "global_phase": float(circuit.global_phase),
            "format": args.format,
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "gate_counts": dict(circuit.count_ops()),
        }
        circuit.metadata = metadata
        compact_metadata = json.dumps(metadata, ensure_ascii=True)
        if args.format == "qasm":
            # OpenQASM 2 has no whole-program global phase. Keep it in both a
            # machine-readable comment (also for stdout) and the JSON sidecar.
            content = f"// naive-block-encoding: {compact_metadata}\n" + qasm2.dumps(circuit)
        elif args.format == "text":
            content = str(circuit.draw(output="text", fold=120))
        else:
            content = None

        if args.output is None:
            print(content)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.format == "qpy":
                with args.output.open("wb") as stream:
                    qpy.dump(circuit, stream)
            else:
                args.output.write_text(content + "\n", encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(compact_metadata, file=sys.stderr)
    except (OSError, ValueError, TypeError, QiskitError) as exc:
        parser.error(str(exc))
    return 0
