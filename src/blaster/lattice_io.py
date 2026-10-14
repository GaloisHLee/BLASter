"""Read and write column-oriented bases in fplll's matrix format."""

from operator import index
from pathlib import Path
import re
import sys

import numpy as np

from ._matrix import as_int64_matrix, is_sage_matrix


_TOKEN = re.compile(r"\[|\]|[+-]?\d+")


def _tokens(text):
    tokens = []
    position = 0
    for match in _TOKEN.finditer(text):
        if text[position:match.start()].strip():
            raise ValueError("invalid token in lattice input")
        tokens.append(match.group())
        position = match.end()
    if text[position:].strip():
        raise ValueError("invalid token in lattice input")
    return tokens


def parse_lattice(text, *, qary_order=True, exact=False):
    """Parse an fplll matrix string and return a column-oriented basis.

    The default returns an int64 array and raises :class:`OverflowError` for
    entries outside the signed int64 range. Pass ``exact=True`` for an
    object-dtype array that preserves entries of any magnitude.
    """
    tokens = _tokens(text)
    cursor = 0

    def take(expected=None):
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("incomplete lattice input")
        token = tokens[cursor]
        if expected is not None and token != expected:
            raise ValueError(f"invalid lattice input: expected {expected!r}")
        cursor += 1
        return token

    take("[")
    rows = []
    while cursor < len(tokens) and tokens[cursor] == "[":
        take("[")
        row = []
        while cursor < len(tokens) and tokens[cursor] not in {"[", "]"}:
            row.append(int(take()))
        take("]")
        if not row:
            raise ValueError("lattice rows must not be empty")
        rows.append(row)
    take("]")
    if cursor != len(tokens) or not rows:
        raise ValueError("invalid lattice input")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("lattice rows must all have the same length")

    if qary_order and np.count_nonzero(rows[-1]) == 1:
        rows.reverse()

    row_matrix = (
        np.array(rows, dtype=object) if exact else as_int64_matrix(rows, name="lattice")
    )
    return np.ascontiguousarray(row_matrix.T)


def _read_text(source):
    if source is None:
        return sys.stdin.read()
    if hasattr(source, "read"):
        return source.read()
    return Path(source).read_text(encoding="utf-8")


def read_lattice(source=None, *, qary_order=True, exact=False):
    """Read a basis from a path, text stream, or stdin when *source* is None."""
    return parse_lattice(_read_text(source), qary_order=qary_order, exact=exact)


def read_qary_lattice(input_file=None, *, exact=False):
    """Backward-compatible q-ary lattice reader."""
    return read_lattice(input_file, qary_order=True, exact=exact)


def _as_object_matrix(value, *, name="matrix"):
    """Return *value* as a C-contiguous object-dtype matrix of Python ints.

    Mirrors :func:`blaster._matrix.as_int64_matrix` validation without the
    int64 range restriction, so entries of any magnitude are preserved.
    """
    if is_sage_matrix(value):
        array = np.asarray(value.list(), dtype=object).reshape(
            int(value.nrows()), int(value.ncols())
        )
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        array = np.asarray(value, dtype=object)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind in "fcUSVb":
        raise TypeError(f"{name} entries must be integers")

    converted = []
    for entry in array.flat:
        if isinstance(entry, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be integers")
        try:
            integer = index(entry)
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
        converted.append(integer)

    return np.ascontiguousarray(
        np.asarray(converted, dtype=object).reshape(array.shape)
    )


def _format_lattice(basis, *, exact=False):
    columns = _as_object_matrix(basis, name="basis") if exact else as_int64_matrix(basis, name="basis")
    rows = columns.T
    lines = ["[" + " ".join(map(str, row)) + "]" for row in rows]
    return "[" + "\n".join(lines) + "]\n"


def write_lattice(basis, output_file=None, *, exact=False):
    """Write a column-oriented basis to a path, stream, or stdout.

    The default rejects entries outside the signed int64 range; pass
    ``exact=True`` to write entries of any magnitude.
    """
    text = _format_lattice(basis, exact=exact)
    if output_file is None:
        sys.stdout.write(text)
    elif hasattr(output_file, "write"):
        output_file.write(text)
    else:
        Path(output_file).write_text(text, encoding="utf-8")
