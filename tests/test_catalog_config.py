from pathlib import Path

from prbench.catalog import AlgorithmCatalog
from prbench.config import ConfigurationLoader


def test_catalog_contains_research_baselines() -> None:
    catalog = AlgorithmCatalog()
    ids = {a.id for a in catalog.all()}
    assert {"cpu_seq", "gpu_cub", "hybrid_static_profiled", "hybrid_dynamic_adaptive"} <= ids


def test_smoke_config_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(root / "configs" / "smoke_cpu.yaml")
    assert cfg.experiments[0].id == "cpu_smoke"


def test_invalid_algorithm_parameter_fails_fast(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
experiments:
  - id: x
    datasets:
      - {size: 1024, dtype: float32}
    algorithms:
      - id: gpu_cub
        params: {block_size: 123}
""",
        encoding="utf-8",
    )
    try:
        ConfigurationLoader(AlgorithmCatalog()).load(config)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported parameter must be rejected")


def test_nested_config_typo_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "bad_nested.yaml"
    config.write_text(
        """
measurement:
  warmup_runz: 3
experiments:
  - id: x
    datasets:
      - {size: 1024, dtype: float32}
    algorithms:
      - {id: cpu_seq}
""",
        encoding="utf-8",
    )
    try:
        ConfigurationLoader(AlgorithmCatalog()).load(config)
    except ValueError:
        pass
    else:
        raise AssertionError("nested configuration typos must be rejected")


def test_comprehensive_tuning_config_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(root / "configs" / "pilot_tuning_full.yaml")
    assert cfg.experiments[0].id == "comprehensive_tuning_each_gpu"


def test_dynamic_fixed_regression_config_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(root / "configs" / "regression_dynamic_fixed.yaml")
    request = cfg.experiments[0].algorithms[0]
    assert 262144 in request.params["chunk_size"]


def test_yaml_12_on_off_are_strings_for_enable_cuda(tmp_path: Path) -> None:
    config = tmp_path / "cuda_on.yaml"
    config.write_text(
        """
output_dir: results
build:
  enable_cuda: on
energy:
  enable_cpu: false
  enable_gpu: false
experiments:
  - id: yaml12
    operations: [sum]
    datasets:
      - {size: 1024, dtype: int32, distribution: integers, seed: 1, low: -1, high: 1, quantization: {mode: none}}
    algorithms:
      - {id: cpu_seq}
""",
        encoding="utf-8",
    )
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(config)
    assert cfg.build.enable_cuda == "on"
    assert cfg.energy.enable_cpu is False
    assert cfg.energy.enable_gpu is False


def test_yaml_12_off_is_not_coerced_to_false(tmp_path: Path) -> None:
    config = tmp_path / "cuda_off.yaml"
    config.write_text(
        """
output_dir: results
build:
  enable_cuda: off
experiments:
  - id: yaml12off
    operations: [sum]
    datasets:
      - {size: 1024, dtype: int32, distribution: integers, seed: 1, low: -1, high: 1, quantization: {mode: none}}
    algorithms:
      - {id: cpu_seq}
""",
        encoding="utf-8",
    )
    cfg = ConfigurationLoader(AlgorithmCatalog()).load(config)
    assert cfg.build.enable_cuda == "off"
