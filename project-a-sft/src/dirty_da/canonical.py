"""Defensive canonical/oracle table storage and integrity checks.

The guarantee is intentionally limited to the scalar and pandas dtypes validated
below. A deep copy plus a SHA-256 fingerprint is an application-level guard,
not a mathematical proof that arbitrary Python objects are immutable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import hashlib
import json
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


class CanonicalDataError(ValueError):
    """Raised when canonical input is unsupported or its snapshot is changed."""


_SCALAR_TYPES = (str, bytes, bool, Integral, Real, Decimal, date, datetime, time, timedelta)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (list, dict, set, tuple)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _validate_scalar(value: Any, location: str) -> None:
    if _is_missing(value) or isinstance(value, _SCALAR_TYPES):
        return
    raise CanonicalDataError(
        f"Unsupported mutable or opaque value at {location}: {type(value).__name__}"
    )


def _validate_label(value: Any, location: str) -> None:
    if isinstance(value, tuple):
        for part_number, part in enumerate(value):
            _validate_scalar(part, f"{location}[{part_number}]")
        return
    _validate_scalar(value, location)


def _dtype_descriptor(dtype: Any) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"dtype": str(dtype), "class": type(dtype).__name__}
    if isinstance(dtype, pd.CategoricalDtype):
        for position, category in enumerate(dtype.categories):
            _validate_scalar(category, f"category[{position}]")
        descriptor.update(
            ordered=dtype.ordered,
            categories=[repr(value) for value in dtype.categories],
            category_types=[type(value).__name__ for value in dtype.categories],
        )
    return descriptor


def validate_fingerprint_support(df: pd.DataFrame) -> None:
    """Reject values whose equality/fingerprint semantics are not supported."""

    if not isinstance(df, pd.DataFrame):
        raise CanonicalDataError("Expected a pandas DataFrame.")
    if not df.columns.is_unique:
        raise CanonicalDataError(
            "Duplicate column labels are unsupported because task fields would be ambiguous."
        )
    for position, label in enumerate(df.columns):
        _validate_label(label, f"column label {position}")
    if df.columns.name is not None:
        _validate_label(df.columns.name, "column index name")
    for level, name in enumerate(df.index.names):
        if name is not None:
            _validate_label(name, f"index name {level}")
    for position, label in enumerate(df.index):
        _validate_label(label, f"index label {position}")

    for column_position, column in enumerate(df.columns):
        series = df.iloc[:, column_position]
        dtype = series.dtype
        supported = (
            ptypes.is_bool_dtype(dtype)
            or ptypes.is_numeric_dtype(dtype)
            or ptypes.is_string_dtype(dtype)
            or ptypes.is_datetime64_any_dtype(dtype)
            or ptypes.is_timedelta64_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or ptypes.is_object_dtype(dtype)
        )
        if not supported:
            raise CanonicalDataError(f"Unsupported dtype for column {column!r}: {dtype}")
        if ptypes.is_object_dtype(dtype):
            for row_position, value in enumerate(series.array):
                _validate_scalar(value, f"row {row_position}, column {column!r}")


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a digest covering values, schema, order, dtypes, and index metadata."""

    validate_fingerprint_support(df)
    value_hashes = pd.util.hash_pandas_object(df, index=True, categorize=False)
    metadata = {
        "shape": list(df.shape),
        "columns": [repr(value) for value in df.columns],
        "column_label_types": [type(value).__name__ for value in df.columns],
        "column_index_name": repr(df.columns.name),
        "dtypes": [_dtype_descriptor(dtype) for dtype in df.dtypes],
        "index_type": type(df.index).__name__,
        "index_names": [repr(value) for value in df.index.names],
        "index_dtypes": [
            str(df.index.get_level_values(level).dtype) for level in range(df.index.nlevels)
        ],
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(value_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


class CanonicalTable:
    """Own a private snapshot and expose only independently copied DataFrames."""

    def __init__(self, df: pd.DataFrame, name: str = "canonical"):
        validate_fingerprint_support(df)
        self.name = name
        self._snapshot = df.copy(deep=True)
        self._initial_fingerprint = dataframe_fingerprint(self._snapshot)

    @property
    def fingerprint(self) -> str:
        self.assert_intact()
        return self._initial_fingerprint

    @property
    def shape(self) -> tuple[int, int]:
        self.assert_intact()
        return self._snapshot.shape

    @property
    def columns(self) -> tuple[Any, ...]:
        self.assert_intact()
        return tuple(self._snapshot.columns.tolist())

    def copy_frame(self) -> pd.DataFrame:
        self.assert_intact()
        return self._snapshot.copy(deep=True)

    def assert_intact(self) -> None:
        current = dataframe_fingerprint(self._snapshot)
        if current != self._initial_fingerprint:
            raise CanonicalDataError(f"Canonical snapshot '{self.name}' was modified.")
