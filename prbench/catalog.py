from __future__ import annotations

from pathlib import Path

import yaml

from .models import AlgorithmDefinition


class AlgorithmCatalog:
    """Read-only registry of research algorithms and their capability requirements."""

    def __init__(self, catalog_path: Path | None = None) -> None:
        if catalog_path is None:
            catalog_path = Path(__file__).resolve().parent / "data" / "algorithm_catalog.yaml"
        raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
        defs = [AlgorithmDefinition.model_validate(item) for item in raw.get("algorithms", [])]
        self._definitions = {item.id: item for item in defs}
        if len(self._definitions) != len(defs):
            raise ValueError("duplicate algorithm ID in catalog")

    def get(self, algorithm_id: str) -> AlgorithmDefinition:
        try:
            return self._definitions[algorithm_id]
        except KeyError as exc:
            raise KeyError(f"unknown algorithm '{algorithm_id}'") from exc

    def all(self) -> list[AlgorithmDefinition]:
        return list(self._definitions.values())
