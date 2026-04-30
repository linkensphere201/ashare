"""Storage initialization and validation."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stock_picker.config import StorageConfig, load_storage_config


@dataclass(frozen=True)
class StorageResult:
    ok: bool
    message: str


def init_storage(config_path: Path) -> StorageResult:
    config = load_storage_config(config_path)
    for directory in config.required_directories:
        directory.mkdir(parents=True, exist_ok=True)
    config.metadata_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    return StorageResult(True, f"storage initialized: {config.data_root}")


def validate_storage(config_path: Path) -> StorageResult:
    config = load_storage_config(config_path)
    missing = [path for path in config.required_directories if not path.exists()]
    if not config.metadata_sqlite_path.exists():
        missing.append(config.metadata_sqlite_path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        return StorageResult(False, f"storage validation failed; missing paths:\n{formatted}")
    return StorageResult(True, f"storage validation succeeded: {config.data_root}")


def register_schemas(config_path: Path) -> StorageResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    schema_files = sorted(config.schema_root.glob("*/*.yaml"))
    if not schema_files:
        return StorageResult(False, f"no schema files found under: {config.schema_root}")

    registered = 0
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        for schema_path in schema_files:
            schema = _load_schema_file(schema_path)
            _upsert_schema(connection, config, schema_path, schema)
            registered += 1

    return StorageResult(True, f"registered {registered} schema files into catalog")


def initialize_metadata_catalog(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(METADATA_SCHEMA_SQL)
        _ensure_metadata_migrations(connection)


def _ensure_metadata_migrations(connection: sqlite3.Connection) -> None:
    curated_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(curated_versions)").fetchall()
    }
    if "notes" not in curated_columns:
        connection.execute("ALTER TABLE curated_versions ADD COLUMN notes TEXT")

    task_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(provider_run_tasks)").fetchall()
    }
    if "symbol_start_offset" not in task_columns:
        connection.execute("ALTER TABLE provider_run_tasks ADD COLUMN symbol_start_offset INTEGER")
    if "symbol_end_offset" not in task_columns:
        connection.execute("ALTER TABLE provider_run_tasks ADD COLUMN symbol_end_offset INTEGER")


def _load_schema_file(schema_path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required to register schema files") from error

    data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"schema file must contain a mapping: {schema_path}")
    return data


def _upsert_schema(
    connection: sqlite3.Connection,
    config: StorageConfig,
    schema_path: Path,
    schema: dict[str, object],
) -> None:
    dataset_id = _required_text(schema, "dataset_id", "table_id", schema_path=schema_path)
    dataset_name = _required_text(schema, "dataset_name", "table_name", schema_path=schema_path)
    layer = _required_text(schema, "layer", schema_path=schema_path)
    version = _required_text(schema, "version", schema_path=schema_path)
    description = _optional_text(schema.get("description"))
    fields = _required_fields(schema, schema_path)
    schema_version_id = f"{dataset_id}_schema_{version}"
    relative_schema_path = schema_path.relative_to(config.project_root).as_posix()
    schema_hash = _sha256_file(schema_path)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    primary_symbol_field = _infer_primary_field(fields, ("symbol",))
    primary_time_field = _infer_primary_field(
        fields,
        ("trade_date", "event_date", "as_of_date", "business_date", "retrieved_at", "created_at"),
    )

    connection.execute(
        """
        INSERT INTO datasets (
          dataset_id,
          dataset_name,
          layer,
          description,
          primary_symbol_field,
          primary_time_field,
          status,
          created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 'active', ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
          dataset_name = excluded.dataset_name,
          layer = excluded.layer,
          description = excluded.description,
          primary_symbol_field = excluded.primary_symbol_field,
          primary_time_field = excluded.primary_time_field,
          status = excluded.status
        """,
        (
            dataset_id,
            dataset_name,
            layer,
            description,
            primary_symbol_field,
            primary_time_field,
            created_at,
        ),
    )
    connection.execute(
        """
        INSERT INTO schema_versions (
          schema_version_id,
          dataset_id,
          version,
          schema_file_path,
          schema_hash,
          status,
          created_at,
          change_summary
        )
        VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
        ON CONFLICT(schema_version_id) DO UPDATE SET
          dataset_id = excluded.dataset_id,
          version = excluded.version,
          schema_file_path = excluded.schema_file_path,
          schema_hash = excluded.schema_hash,
          status = excluded.status,
          change_summary = excluded.change_summary
        """,
        (
            schema_version_id,
            dataset_id,
            version,
            relative_schema_path,
            schema_hash,
            created_at,
            description,
        ),
    )
    connection.execute("DELETE FROM schema_fields WHERE schema_version_id = ?", (schema_version_id,))
    connection.executemany(
        """
        INSERT INTO schema_fields (
          schema_version_id,
          field_name,
          field_type,
          nullable,
          is_primary_key,
          is_partition_key,
          description
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                schema_version_id,
                _required_text(field, "name", schema_path=schema_path),
                _required_text(field, "type", schema_path=schema_path),
                int(bool(field.get("nullable", True))),
                int(bool(field.get("primary_key", False))),
                int(bool(field.get("partition_key", False))),
                _optional_text(field.get("description")),
            )
            for field in fields
        ],
    )


def _required_text(
    data: dict[str, object],
    *keys: str,
    schema_path: Path,
) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    joined = " or ".join(keys)
    raise ValueError(f"schema file missing required text field {joined}: {schema_path}")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _required_fields(schema: dict[str, object], schema_path: Path) -> list[dict[str, object]]:
    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"schema file must contain a non-empty fields list: {schema_path}")
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError(f"schema field must be a mapping: {schema_path}")
    return fields


def _infer_primary_field(fields: list[dict[str, object]], candidates: tuple[str, ...]) -> str | None:
    field_names = {str(field.get("name")) for field in fields}
    for candidate in candidates:
        if candidate in field_names:
            return candidate
    return None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


METADATA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS data_batches (
  batch_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  retrieved_at TEXT NOT NULL,
  business_date TEXT,
  raw_path TEXT NOT NULL,
  format TEXT NOT NULL,
  row_count INTEGER,
  schema_hash TEXT,
  content_checksum TEXT,
  status TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS provider_runs (
  run_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  start_date TEXT,
  end_date TEXT,
  as_of_date TEXT,
  status TEXT NOT NULL,
  total_symbols INTEGER NOT NULL,
  next_offset INTEGER NOT NULL,
  batch_size INTEGER NOT NULL,
  requested_symbols INTEGER NOT NULL,
  symbols_with_rows INTEGER NOT NULL,
  failed_symbols INTEGER NOT NULL,
  row_count INTEGER NOT NULL,
  raw_batch_ids TEXT NOT NULL,
  failure_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS provider_run_tasks (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  source TEXT NOT NULL,
  dataset_name TEXT NOT NULL,
  trade_date TEXT,
  symbol_start_offset INTEGER,
  symbol_end_offset INTEGER,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL,
  raw_batch_id TEXT,
  row_count INTEGER NOT NULL,
  error_message TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  notes TEXT,
  FOREIGN KEY (run_id) REFERENCES provider_runs(run_id)
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  dataset_name TEXT NOT NULL,
  layer TEXT NOT NULL,
  description TEXT,
  primary_symbol_field TEXT,
  primary_time_field TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_versions (
  schema_version_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  version TEXT NOT NULL,
  schema_file_path TEXT,
  schema_hash TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  change_summary TEXT,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
);

CREATE TABLE IF NOT EXISTS schema_fields (
  schema_version_id TEXT NOT NULL,
  field_name TEXT NOT NULL,
  field_type TEXT NOT NULL,
  nullable INTEGER NOT NULL,
  is_primary_key INTEGER NOT NULL,
  is_partition_key INTEGER NOT NULL,
  description TEXT,
  PRIMARY KEY (schema_version_id, field_name),
  FOREIGN KEY (schema_version_id) REFERENCES schema_versions(schema_version_id)
);

CREATE TABLE IF NOT EXISTS curated_versions (
  curated_version_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  schema_version_id TEXT NOT NULL,
  version_type TEXT NOT NULL,
  snapshot_id TEXT,
  path TEXT NOT NULL,
  as_of_date TEXT,
  created_at TEXT NOT NULL,
  source_batch_ids TEXT,
  row_count INTEGER,
  checksum TEXT,
  status TEXT NOT NULL,
  FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
  FOREIGN KEY (schema_version_id) REFERENCES schema_versions(schema_version_id)
);

CREATE TABLE IF NOT EXISTS snapshot_manifests (
  snapshot_id TEXT PRIMARY KEY,
  as_of_date TEXT NOT NULL,
  created_at TEXT NOT NULL,
  data_frequency TEXT NOT NULL,
  config_version TEXT,
  manifest_json TEXT NOT NULL,
  notes TEXT
);
"""
