"""CLI export tests that load and execute the exported circuits again."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from qiskit import qasm2, qpy
from qiskit.quantum_info import Operator

from naive_block_encoding import block_encode
from naive_block_encoding.cli import main


@pytest.fixture
def matrix_file(tmp_path: Path) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps([["1+1j", 2], [0, "-1j"]]), encoding="utf-8-sig")
    return path


def read_sidecar(output: Path) -> dict:
    return json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))


def test_json_complex_strings_and_default_stdout(matrix_file: Path, capsys) -> None:
    assert main(["--matrix", str(matrix_file), "--alpha", "4"]) == 0

    captured = capsys.readouterr()
    metadata = json.loads(captured.err)
    assert "U_A" in captured.out
    assert metadata["alpha"] == 4.0
    assert metadata["input_dimension"] == 2
    assert metadata["padded_dimension"] == 2
    assert metadata["num_system_qubits"] == 1
    assert metadata["ancilla_qubit"] == 1
    assert metadata["num_ancillas"] == 1
    assert metadata["format"] == "text"


def test_text_output_and_sidecar_in_new_directory(
    matrix_file: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "new-directory" / "circuit.txt"
    assert main(["--matrix", str(matrix_file), "--output", str(output), "--decompose"]) == 0

    captured = capsys.readouterr()
    metadata = read_sidecar(output)
    assert captured.out == ""
    assert json.loads(captured.err) == metadata
    assert output.read_text(encoding="utf-8").strip()
    assert set(metadata["gate_counts"]) <= {"u", "cx"}
    assert metadata["num_qubits"] == 2
    assert metadata["depth"] > 0


def test_qpy_roundtrip_preserves_operator_and_normalization(
    matrix_file: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "circuit.qpy"
    assert (
        main(
            [
                "--matrix",
                str(matrix_file),
                "--alpha",
                "4",
                "--format",
                "qpy",
                "--output",
                str(output),
                "--decompose",
            ]
        )
        == 0
    )

    with output.open("rb") as stream:
        circuits = qpy.load(stream)
    assert len(circuits) == 1
    restored = circuits[0]
    metadata = read_sidecar(output)
    assert restored.metadata == metadata
    assert metadata["alpha"] == 4.0
    assert metadata["format"] == "qpy"
    assert json.loads(capsys.readouterr().err) == metadata
    matrix = np.array([[1 + 1j, 2], [0, -1j]])
    expected = block_encode(matrix, alpha=4)
    assert_allclose(Operator(restored).data, Operator(expected.circuit).data, atol=1e-8)
    assert_allclose(Operator(restored).data[:2, :2], matrix / metadata["alpha"], atol=1e-8)


def test_standard_qasm_roundtrip_restores_phase_from_sidecar(
    matrix_file: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "circuit.qasm"
    assert (
        main(
            [
                "--matrix",
                str(matrix_file),
                "--alpha",
                "4",
                "--format",
                "qasm",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    # Deliberately use the standard parser, with no Qiskit legacy extensions.
    restored = qasm2.load(output)
    metadata = read_sidecar(output)
    assert metadata == json.loads(capsys.readouterr().err)
    assert metadata["format"] == "qasm"
    assert set(metadata["gate_counts"]) <= {"u3", "cx"}
    restored.global_phase += metadata["global_phase"]
    matrix = np.array([[1 + 1j, 2], [0, -1j]])
    expected = block_encode(matrix, alpha=4)
    assert_allclose(Operator(restored).data, Operator(expected.circuit).data, atol=1e-8)
    assert_allclose(Operator(restored).data[:2, :2], matrix / metadata["alpha"], atol=1e-8)


def test_qasm_stdout_contains_machine_readable_metadata(matrix_file: Path, capsys) -> None:
    assert main(["--matrix", str(matrix_file), "--format", "qasm"]) == 0

    captured = capsys.readouterr()
    prefix = "// naive-block-encoding: "
    first_line = captured.out.splitlines()[0]
    assert first_line.startswith(prefix)
    metadata = json.loads(first_line[len(prefix) :])
    assert metadata == json.loads(captured.err)
    restored = qasm2.loads(captured.out)
    restored.global_phase += metadata["global_phase"]
    matrix = np.array([[1 + 1j, 2], [0, -1j]])
    assert_allclose(Operator(restored).data[:2, :2], matrix / metadata["alpha"], atol=1e-8)


def test_qpy_requires_output_path(matrix_file: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(matrix_file), "--format", "qpy"])

    assert exc.value.code == 2
    assert "requires --output" in capsys.readouterr().err


def test_missing_matrix_argument_is_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2
    assert "--matrix" in capsys.readouterr().err


def test_missing_input_file_is_usage_error(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(tmp_path / "missing.json")])

    assert exc.value.code == 2
    assert "missing.json" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["not json", "[[1, 2, 3], [4, 5, 6]]", '[["bad"]]'])
def test_invalid_input_is_reported_without_writing_output(
    content: str, tmp_path: Path, capsys
) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(content, encoding="utf-8")
    output = tmp_path / "circuit.txt"

    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(source), "--output", str(output)])

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()
    assert not Path(str(output) + ".json").exists()


@pytest.mark.parametrize("collision", ["output", "sidecar"])
def test_output_and_sidecar_cannot_overwrite_input(collision: str, tmp_path: Path, capsys) -> None:
    source = tmp_path / "matrix.json"
    original = "[[1, 0], [0, 1]]\n"
    source.write_text(original, encoding="utf-8")
    output = source if collision == "output" else tmp_path / "matrix"

    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(source), "--output", str(output)])

    assert exc.value.code == 2
    assert "overwrite the input matrix" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8") == original
    if collision == "sidecar":
        assert not output.exists()
