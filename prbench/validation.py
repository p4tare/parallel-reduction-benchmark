from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import DType, ReductionOperation


@dataclass(frozen=True)
class ValidationResult:
    is_correct: bool
    absolute_error: float
    relative_error: float | None
    tolerance: float


class ResultValidator:
    """Numerical validation separated from execution and configurable by data type."""

    def validate(
        self,
        actual: int | float,
        reference: int | float,
        sum_abs: float,
        dtype: DType,
        count: int,
        operation: ReductionOperation = ReductionOperation.sum,
    ) -> ValidationResult:
        if operation in {ReductionOperation.min, ReductionOperation.max}:
            if dtype in {DType.int32, DType.int64}:
                ok = int(actual) == int(reference)
                err = float(abs(int(actual) - int(reference)))
            else:
                actual_f = float(actual)
                ref_f = float(reference)
                ok = math.isfinite(actual_f) and actual_f == ref_f
                err = abs(actual_f - ref_f) if math.isfinite(actual_f) else math.inf
            return ValidationResult(bool(ok), float(err), 0.0 if ok else None, 0.0)

        if dtype in {DType.int32, DType.int64}:
            ok = int(actual) == int(reference)
            err = float(abs(int(actual) - int(reference)))
            return ValidationResult(ok, err, 0.0 if ok else None, 0.0)

        actual_f = float(actual)
        ref_f = float(reference)
        if not math.isfinite(actual_f):
            return ValidationResult(False, math.inf, math.inf, 0.0)

        eps = np.finfo(np.float32 if dtype == DType.float32 else np.float64).eps
        # Correctness validation is intended to catch implementation defects, not to declare
        # all floating-point summation orders bitwise equivalent.  Use a conventional relative
        # term plus a cancellation-sensitive absolute term that grows with sqrt(N), rather than
        # an O(N*eps*sum_abs) worst-case bound that becomes too permissive for large vectors.
        # Raw error is always preserved so numerical behavior can be analysed separately.
        rtol = 1.0e-5 if dtype == DType.float32 else 1.0e-12
        rms_scale = float(sum_abs) / max(1.0, math.sqrt(float(max(1, count))))
        cancellation_atol = 64.0 * eps * max(1.0, rms_scale)
        tolerance = max(cancellation_atol, rtol * abs(ref_f))
        abs_error = abs(actual_f - ref_f)
        rel_error = abs_error / abs(ref_f) if ref_f != 0.0 else None
        return ValidationResult(bool(abs_error <= tolerance), float(abs_error), rel_error, float(tolerance))
