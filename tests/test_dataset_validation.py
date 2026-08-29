from pathlib import Path

from prbench.datasets import DatasetFactory
from prbench.models import DType, DatasetSpec, Distribution, QuantizationConfig, QuantizationMode
from prbench.validation import ResultValidator


def test_dataset_is_deterministic(tmp_path: Path) -> None:
    spec = DatasetSpec(
        size=10001,
        dtype=DType.float32,
        distribution=Distribution.symmetric_uniform,
        seed=123,
        low=-1.0,
        high=1.0,
        quantization=QuantizationConfig(mode=QuantizationMode.binary_fraction, bits=8),
    )
    factory = DatasetFactory(tmp_path)
    first = factory.get_or_create(spec)
    second = factory.get_or_create(spec)
    assert first.metadata["sha256"] == second.metadata["sha256"]
    assert first.metadata["reference_sum"] == 0.0


def test_zero_reference_does_not_accept_wrong_result() -> None:
    result = ResultValidator().validate(123.0, 0.0, 1000.0, DType.float32, 1000)
    assert result.is_correct is False


def test_dataset_stores_min_max_references(tmp_path: Path) -> None:
    from prbench.models import ReductionOperation

    spec = DatasetSpec(
        size=1001,
        dtype=DType.float32,
        distribution=Distribution.uniform,
        seed=321,
        low=-3.0,
        high=7.0,
        quantization=QuantizationConfig(mode=QuantizationMode.binary_fraction, bits=8),
    )
    artifact = DatasetFactory(tmp_path).get_or_create(spec)
    assert artifact.metadata["reference_min"] <= artifact.metadata["reference_max"]
    assert artifact.reference_for(ReductionOperation.min) == artifact.metadata["reference_min"]
    assert artifact.reference_for(ReductionOperation.max) == artifact.metadata["reference_max"]


def test_min_max_validation_is_exact_for_finite_inputs() -> None:
    from prbench.models import ReductionOperation

    validator = ResultValidator()
    assert validator.validate(1.25, 1.25, 10.0, DType.float32, 100, ReductionOperation.min).is_correct
    assert not validator.validate(1.5, 1.25, 10.0, DType.float32, 100, ReductionOperation.min).is_correct
    assert validator.validate(9, 9, 10.0, DType.int32, 100, ReductionOperation.max).is_correct
