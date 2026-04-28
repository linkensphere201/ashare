"""Storage initialization and validation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
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


def initialize_metadata_catalog(sqlite_path: Path) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        connection.executescript(METADATA_SCHEMA_SQL)


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
