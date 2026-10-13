"""Integer matrix conversion helpers shared by the public API and file I/O."""

from operator import index

import numpy as np


_INT64_INFO = np.iinfo(np.int64)


def as_int64_matrix(value, *, name="matrix"):
    """Return *value* as a C-contiguous int64 matrix without silent coercion."""
    try:
        if is_sage_matrix(value):
            array = np.asarray(value.list(), dtype=object).reshape(
                int(value.nrows()), int(value.ncols())
            )
        elif isinstance(value, np.ndarray):
            array = np.asarray(value)
        else:
            array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular two-dimensional matrix") from exc

    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    if array.dtype.kind in "fcUSVb":
        raise TypeError(f"{name} entries must be integers")

    if array.dtype.kind == "i" and array.dtype.itemsize <= np.dtype(np.int64).itemsize:
        return np.ascontiguousarray(array, dtype=np.int64)

    if array.dtype.kind == "u":
        if int(array.max()) > _INT64_INFO.max:
            raise OverflowError(f"{name} entries must fit in signed int64")
        return np.ascontiguousarray(array, dtype=np.int64)

    converted = []
    for entry in array.flat:
        if isinstance(entry, (bool, np.bool_)):
            raise TypeError(f"{name} entries must be integers")
        try:
            integer = index(entry)
        except TypeError as exc:
            raise TypeError(f"{name} entries must be integers") from exc
        if not _INT64_INFO.min <= integer <= _INT64_INFO.max:
            raise OverflowError(f"{name} entries must fit in signed int64")
        converted.append(integer)

    return np.ascontiguousarray(
        np.asarray(converted, dtype=np.int64).reshape(array.shape)
    )


def is_sage_matrix(value):
    """Return whether *value* behaves like a Sage matrix without importing Sage."""
    return (
        type(value).__module__.startswith("sage.")
        and hasattr(value, "base_ring")
        and hasattr(value, "nrows")
        and hasattr(value, "ncols")
    )


def to_sage_matrix(entries, base_ring=None):
    """Convert integer entries to a Sage matrix, importing Sage only when needed."""
    from sage.all import ZZ, matrix

    return matrix(base_ring if base_ring is not None else ZZ, entries.tolist())
