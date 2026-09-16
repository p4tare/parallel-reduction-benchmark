from __future__ import annotations

from pathlib import Path
import re

import yaml

from .catalog import AlgorithmCatalog
from .models import RootConfig


class PrbenchSafeLoader(yaml.SafeLoader):
    """Safe YAML loader with YAML 1.2-style boolean semantics."""


PrbenchSafeLoader.yaml_implicit_resolvers = {
    key: list(resolvers)
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for key, resolvers in list(PrbenchSafeLoader.yaml_implicit_resolvers.items()):
    PrbenchSafeLoader.yaml_implicit_resolvers[key] = [
        item for item in resolvers if item[0] != "tag:yaml.org,2002:bool"
    ]

PrbenchSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


class ConfigurationLoader:
    def __init__(self, catalog: AlgorithmCatalog) -> None:
        self.catalog = catalog

    def load(self, path: Path) -> RootConfig:
        if not path.exists():
            raise FileNotFoundError(path)
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=PrbenchSafeLoader)
        config = RootConfig.model_validate(raw)
        self._validate_algorithms(config)
        return config

    def _validate_algorithms(self, config: RootConfig) -> None:
        for group in config.experiments:
            for request in group.algorithms:
                definition = self.catalog.get(request.id)
                unknown = set(request.params) - set(definition.tunables)
                if unknown:
                    raise ValueError(
                        f"algorithm '{request.id}' has unsupported parameters: {sorted(unknown)}"
                    )
                for name, raw_value in request.params.items():
                    values = raw_value if isinstance(raw_value, list) else [raw_value]
                    for value in values:
                        self._validate_parameter_value(request.id, name, value)

    @staticmethod
    def _validate_parameter_value(algorithm_id: str, name: str, value: object) -> None:
        def as_int() -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{algorithm_id}.{name} must be an integer, got {value!r}")
            return value

        def as_number() -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{algorithm_id}.{name} must be numeric, got {value!r}")
            return float(value)

        if name == "block_size":
            number = as_int()
            if number < 32 or number > 1024 or (number & (number - 1)) != 0:
                raise ValueError(f"{algorithm_id}.block_size must be a power of two in [32, 1024]")
        elif name in {"chunk_size", "min_chunk_size", "max_chunk_size", "pipeline_chunk_elements"}:
            if as_int() <= 0:
                raise ValueError(f"{algorithm_id}.{name} must be positive")
        elif name in {"pipeline_streams", "pipeline_chunks", "reuse_count"}:
            if as_int() <= 0:
                raise ValueError(f"{algorithm_id}.{name} must be positive")
        elif name == "use_cuda_graphs":
            if not isinstance(value, bool):
                raise ValueError(f"{algorithm_id}.use_cuda_graphs must be boolean")
        elif name == "cpu_fraction":
            fraction = as_number()
            if not 0.0 <= fraction < 1.0:
                raise ValueError(f"{algorithm_id}.cpu_fraction must be in [0,1)")
        elif name == "guided_factor":
            if as_number() <= 0.0:
                raise ValueError(f"{algorithm_id}.guided_factor must be positive")
        elif name == "target_chunk_ms":
            if as_number() <= 0.0:
                raise ValueError(f"{algorithm_id}.target_chunk_ms must be positive")
        elif name == "ema_alpha":
            alpha = as_number()
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"{algorithm_id}.ema_alpha must be in (0, 1]")
