"""Storage configuration loading.

The project declares PyYAML as a dependency, but this module keeps a tiny
fallback parser for the simple MVP config shape so storage commands remain
usable before dependencies are installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_INTERPOLATION = re.compile(r"\$\{storage\.([a-zA-Z0-9_]+)\}")


@dataclass(frozen=True)
class StorageConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    raw_root: Path
    curated_root: Path
    current_curated_root: Path
    frozen_curated_root: Path
    reports_root: Path
    backtests_root: Path
    metadata_sqlite_path: Path
    schema_root: Path

    @property
    def required_directories(self) -> list[Path]:
        return [
            self.data_root,
            self.raw_root,
            self.curated_root,
            self.current_curated_root,
            self.frozen_curated_root,
            self.reports_root,
            self.backtests_root,
            self.schema_root,
        ]


def load_storage_config(config_path: Path) -> StorageConfig:
    resolved_config = config_path.resolve()
    project_root = resolved_config.parent.parent
    values = _load_storage_values(resolved_config)
    expanded = _expand_values(values)

    return StorageConfig(
        config_path=resolved_config,
        project_root=project_root,
        data_root=_resolve_path(project_root, expanded["data_root"]),
        raw_root=_resolve_path(project_root, expanded["raw_root"]),
        curated_root=_resolve_path(project_root, expanded["curated_root"]),
        current_curated_root=_resolve_path(project_root, expanded["current_curated_root"]),
        frozen_curated_root=_resolve_path(project_root, expanded["frozen_curated_root"]),
        reports_root=_resolve_path(project_root, expanded["reports_root"]),
        backtests_root=_resolve_path(project_root, expanded["backtests_root"]),
        metadata_sqlite_path=_resolve_path(project_root, expanded["metadata_sqlite_path"]),
        schema_root=_resolve_path(project_root, expanded["schema_root"]),
    )


def _load_storage_values(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        raise FileNotFoundError(f"storage config not found: {config_path}")

    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _load_simple_storage_yaml(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    storage = data.get("storage", {}) if isinstance(data, dict) else {}
    if not isinstance(storage, dict):
        raise ValueError("storage config must contain a mapping named 'storage'")
    return {key: str(value) for key, value in storage.items()}


def _load_simple_storage_yaml(config_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    in_storage = False

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "storage:":
            in_storage = True
            continue
        if not in_storage:
            continue
        if not raw_line.startswith("  "):
            break
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        values[key] = value.strip().strip("\"'")

    if not values:
        raise ValueError("storage config must contain storage path values")
    return values


def _expand_values(values: dict[str, str]) -> dict[str, str]:
    expanded: dict[str, str] = {}

    def expand(key: str, stack: set[str]) -> str:
        if key in expanded:
            return expanded[key]
        if key in stack:
            raise ValueError(f"circular storage config interpolation for {key}")
        if key not in values:
            raise KeyError(f"missing storage config value: {key}")

        def replace(match: re.Match[str]) -> str:
            return expand(match.group(1), stack | {key})

        expanded[key] = _INTERPOLATION.sub(replace, values[key])
        return expanded[key]

    for key in values:
        expand(key, set())
    return expanded


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
