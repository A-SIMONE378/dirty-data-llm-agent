"""Strict programmatic result comparison; no LLM judge is involved."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


@dataclass
class VerificationResult:
    passed: bool
    reason: str


def _mismatch(
    expected: Any,
    actual: Any,
    *,
    atol: float,
    rtol: float,
    path: str,
) -> str | None:
    if expected is None or actual is None:
        return None if expected is actual else f"{path}: expected {expected!r}, got {actual!r}"
    if isinstance(expected, bool) or isinstance(actual, bool):
        return None if type(expected) is type(actual) and expected == actual else (
            f"{path}: boolean/type mismatch"
        )
    if isinstance(expected, Integral):
        if not isinstance(actual, Integral) or isinstance(actual, bool):
            return f"{path}: expected integer, got {type(actual).__name__}"
        return None if int(expected) == int(actual) else f"{path}: {actual!r} != {expected!r}"
    if isinstance(expected, Real):
        if not isinstance(actual, Real) or isinstance(actual, (bool, Integral)):
            return f"{path}: expected float, got {type(actual).__name__}"
        if math.isnan(float(expected)) or math.isnan(float(actual)):
            return f"{path}: NaN is not a valid answer; use None for a missing result"
        return None if math.isclose(
            float(expected), float(actual), abs_tol=atol, rel_tol=rtol
        ) else f"{path}: {actual!r} is outside tolerance of {expected!r}"
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return f"{path}: list shape/type mismatch"
        for position, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatch = _mismatch(
                expected_item,
                actual_item,
                atol=atol,
                rtol=rtol,
                path=f"{path}[{position}]",
            )
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return f"{path}: object fields do not match"
        for key in expected:
            mismatch = _mismatch(
                expected[key],
                actual[key],
                atol=atol,
                rtol=rtol,
                path=f"{path}.{key}",
            )
            if mismatch:
                return mismatch
        return None
    if type(expected) is not type(actual):
        return f"{path}: expected {type(expected).__name__}, got {type(actual).__name__}"
    return None if expected == actual else f"{path}: {actual!r} != {expected!r}"


def verify_result(
    expected: Any,
    actual: Any,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> VerificationResult:
    if atol < 0 or rtol < 0:
        raise ValueError("Numeric tolerances must be non-negative.")
    mismatch = _mismatch(expected, actual, atol=atol, rtol=rtol, path="$")
    if mismatch is None:
        return VerificationResult(True, "exact structure/type match within numeric tolerance")
    return VerificationResult(False, mismatch)
