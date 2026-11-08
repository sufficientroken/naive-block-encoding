"""Reproducible matrix generation and JSON-to-circuit integration tests."""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from qiskit import qpy
from qiskit.quantum_info import Operator

from naive_block_encoding import block_encode, random_matrix
from naive_block_encoding.cli import main


def test_bundled_strings_match_explicit_sources() -> None:
    data = files("naive_block_encoding").joinpath("data")
    sources = tuple(
        data.joinpath(name).read_text(encoding="utf-8")
        for name in ("random_strings_a.txt", "random_strings_b.txt")
    )

    assert all(source.strip() for source in sources)
    assert_array_equal(random_matrix(3), random_matrix(3, sources=sources))


def test_same_sources_and_index_reproduce_matrix() -> None:
    sources = ("first source", "second source")
    first = random_matrix(4, index=7, complex_entries=True, sources=sources)
    second = random_matrix(4, index=7, complex_entries=True, sources=sources)

    assert_array_equal(first, second)
    first[0, 0] = 100
    assert_array_equal(second, random_matrix(4, index=7, complex_entries=True, sources=sources))


def test_each_source_and_index_influence_matrix() -> None:
    reference = random_matrix(3, sources=("first", "second"))

    assert not np.array_equal(reference, random_matrix(3, sources=("changed", "second")))
    assert not np.array_equal(reference, random_matrix(3, sources=("first", "changed")))
    assert not np.array_equal(reference, random_matrix(3, index=1, sources=("first", "second")))
    assert not np.array_equal(reference, random_matrix(3, sources=("second", "first")))


def test_source_boundaries_are_unambiguous() -> None:
    assert not np.array_equal(
        random_matrix(3, sources=("ab", "c")), random_matrix(3, sources=("a", "bc"))
    )


def test_line_endings_are_normalized_without_stripping_other_whitespace() -> None:
    reference = random_matrix(3, sources=("first\nline\n", "second\nline\n"))

    assert_array_equal(reference, random_matrix(3, sources=("first\r\nline\r\n", "second\rline\r")))
    assert not np.array_equal(
        reference, random_matrix(3, sources=("first\nline", "second\nline\n"))
    )
    assert not np.array_equal(
        reference, random_matrix(3, sources=("first\nline\n ", "second\nline\n"))
    )


@pytest.mark.parametrize("dimension", [1, 3, 8])
@pytest.mark.parametrize("complex_entries", [False, True])
def test_shape_dtype_and_component_bounds(dimension: int, complex_entries: bool) -> None:
    matrix = random_matrix(dimension, complex_entries=complex_entries)

    assert matrix.shape == (dimension, dimension)
    assert matrix.dtype == (np.dtype("complex128") if complex_entries else np.dtype("float64"))
    assert np.isfinite(matrix).all()
    assert np.all((-1 <= matrix.real) & (matrix.real < 1))
    if complex_entries:
        assert np.all((-1 <= matrix.imag) & (matrix.imag < 1))
        assert not np.array_equal(matrix.real, matrix.imag)


def test_numpy_integer_arguments_are_supported() -> None:
    assert_array_equal(random_matrix(np.int64(3), index=np.int32(2)), random_matrix(3, index=2))


def test_global_numpy_random_state_is_unchanged() -> None:
    before = np.random.get_state()
    random_matrix(3, complex_entries=True)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("dimension", [0, -1, 1.5, "3", True, np.bool_(True), None])
def test_invalid_dimensions_are_rejected(dimension) -> None:
    with pytest.raises((TypeError, ValueError)):
        random_matrix(dimension)


@pytest.mark.parametrize("index", [-1, 0.5, "2", True, np.bool_(False), None])
def test_invalid_indices_are_rejected(index) -> None:
    with pytest.raises((TypeError, ValueError)):
        random_matrix(2, index=index)


@pytest.mark.parametrize("complex_entries", [0, 1, "true", None])
def test_invalid_complex_flags_are_rejected(complex_entries) -> None:
    with pytest.raises((TypeError, ValueError)):
        random_matrix(2, complex_entries=complex_entries)


@pytest.mark.parametrize(
    "sources",
    [
        (),
        ("one",),
        ("one", "two", "three"),
        ("", "two"),
        ("one", " \n\t"),
        (b"one", "two"),
        ("one", 2),
        "ab",
    ],
)
def test_invalid_sources_are_rejected(sources) -> None:
    with pytest.raises((TypeError, ValueError)):
        random_matrix(2, sources=sources)


def test_generated_complex_matrix_preserves_encoded_block_and_phase() -> None:
    matrix = random_matrix(3, index=4, complex_entries=True)
    encoding = block_encode(matrix)
    operator = Operator(encoding.circuit).data
    expected = np.zeros((4, 4), dtype=complex)
    expected[:3, :3] = matrix / encoding.alpha

    assert encoding.input_dimension == 3
    assert encoding.padded_dimension == 4
    assert encoding.ancilla_qubit == 2
    assert_allclose(operator[:4, :4], expected, atol=1e-12, rtol=0)
    assert_allclose(operator.conj().T @ operator, np.eye(8), atol=1e-12, rtol=0)


@pytest.mark.parametrize("complex_entries", [False, True])
def test_cli_json_output_can_be_encoded(complex_entries: bool, tmp_path: Path, capsys) -> None:
    matrix_path = tmp_path / "new" / "matrix.json"
    arguments = [
        "generate-matrix",
        "--dimension",
        "3",
        "--index",
        "2",
        "--output",
        str(matrix_path),
    ]
    if complex_entries:
        arguments.append("--complex")

    assert main(arguments) == 0
    assert capsys.readouterr().out == ""
    entries = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert isinstance(entries[0][0], str if complex_entries else float)
    matrix = np.asarray(entries, dtype=complex)
    assert_array_equal(matrix, random_matrix(3, index=2, complex_entries=complex_entries))

    circuit_path = tmp_path / "circuit.qpy"
    assert (
        main(["--matrix", str(matrix_path), "--format", "qpy", "--output", str(circuit_path)]) == 0
    )
    with circuit_path.open("rb") as stream:
        circuit = qpy.load(stream)[0]
    assert_allclose(
        Operator(circuit).data[:3, :3], matrix / circuit.metadata["alpha"], atol=1e-12, rtol=0
    )


def test_cli_stdout_is_a_json_matrix(capsys) -> None:
    assert main(["generate-matrix", "--dimension", "1"]) == 0

    assert_array_equal(json.loads(capsys.readouterr().out), random_matrix(1))


def test_cli_custom_source_files(tmp_path: Path, capsys) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_bytes(b"first\r\nline\r\n")
    source_b.write_text("second source", encoding="utf-8")

    assert (
        main(
            [
                "generate-matrix",
                "--dimension",
                "2",
                "--strings-a",
                str(source_a),
                "--strings-b",
                str(source_b),
            ]
        )
        == 0
    )
    matrix = json.loads(capsys.readouterr().out)

    assert_array_equal(matrix, random_matrix(2, sources=("first\nline\n", "second source")))


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--dimension", "0"],
        ["--dimension", "-2"],
        ["--dimension", "2.5"],
        ["--dimension", "2", "--index", "-1"],
        ["--dimension", "2", "--strings-a", "a.txt"],
        ["--dimension", "2", "--strings-b", "b.txt"],
    ],
)
def test_cli_invalid_arguments_are_usage_errors(
    arguments: list[str], tmp_path: Path, capsys
) -> None:
    output = tmp_path / "matrix.json"
    with pytest.raises(SystemExit) as exc:
        main(["generate-matrix", *arguments, "--output", str(output)])

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("invalid_source", ["missing", "empty", "invalid-utf8"])
def test_cli_unreadable_sources_do_not_write_output(
    invalid_source: str, tmp_path: Path, capsys
) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    if invalid_source == "empty":
        source_a.write_text(" \n", encoding="utf-8")
    elif invalid_source == "invalid-utf8":
        source_a.write_bytes(b"\xff")
    source_b.write_text("second source", encoding="utf-8")
    output = tmp_path / "matrix.json"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate-matrix",
                "--dimension",
                "2",
                "--strings-a",
                str(source_a),
                "--strings-b",
                str(source_b),
                "--output",
                str(output),
            ]
        )

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("collision", ["a", "b"])
def test_cli_output_cannot_overwrite_either_source(collision: str, tmp_path: Path, capsys) -> None:
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("first source", encoding="utf-8")
    source_b.write_text("second source", encoding="utf-8")
    output = source_a if collision == "a" else source_b

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "generate-matrix",
                "--dimension",
                "2",
                "--strings-a",
                str(source_a),
                "--strings-b",
                str(source_b),
                "--output",
                str(output),
            ]
        )

    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert source_a.read_text(encoding="utf-8") == "first source"
    assert source_b.read_text(encoding="utf-8") == "second source"
