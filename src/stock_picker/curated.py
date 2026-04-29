"""Curated data import helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from stock_picker.config import StorageConfig, load_storage_config
from stock_picker.storage import StorageResult, initialize_metadata_catalog


@dataclass(frozen=True)
class CuratedImportResult:
    ok: bool
    message: str
    output_path: Path | None = None
    row_count: int = 0


@dataclass(frozen=True)
class CuratedInspectionResult:
    ok: bool
    message: str


def import_curated_csv(
    config_path: Path,
    dataset_id: str,
    input_path: Path,
    source: str = "manual_csv",
    as_of_date: str | None = None,
) -> CuratedImportResult:
    config = load_storage_config(config_path)
    resolved_input = _resolve_input_path(config, input_path)
    if not resolved_input.exists():
        return CuratedImportResult(False, f"input CSV not found: {resolved_input}")

    initialize_metadata_catalog(config.metadata_sqlite_path)
    schema = _load_registered_schema(config, dataset_id)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    data_version = as_of_date or now[:10]
    source_batch_id = _source_batch_id(dataset_id, source, resolved_input, now)

    frame = pl.read_csv(resolved_input)
    frame = _enrich_frame(frame, schema, source, source_batch_id, data_version, now)
    frame = _fill_missing_nullable_columns(frame, schema)
    validation_error = _validate_required_columns(frame, schema)
    if validation_error:
        return CuratedImportResult(False, validation_error)

    output_dir = config.current_curated_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "part-000.parquet"
    frame.select([field["name"] for field in schema]).write_parquet(output_path)

    row_count = frame.height
    checksum = _sha256_file(output_path)
    curated_version_id = f"{dataset_id}_current_{data_version.replace('-', '')}"
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO data_batches (
              batch_id,
              source,
              dataset_name,
              retrieved_at,
              business_date,
              raw_path,
              format,
              row_count,
              schema_hash,
              content_checksum,
              status,
              notes
            )
            VALUES (?, ?, ?, ?, ?, ?, 'csv', ?, NULL, ?, 'success', ?)
            ON CONFLICT(batch_id) DO UPDATE SET
              retrieved_at = excluded.retrieved_at,
              business_date = excluded.business_date,
              raw_path = excluded.raw_path,
              row_count = excluded.row_count,
              content_checksum = excluded.content_checksum,
              status = excluded.status,
              notes = excluded.notes
            """,
            (
                source_batch_id,
                source,
                dataset_id,
                now,
                as_of_date,
                str(resolved_input),
                row_count,
                _sha256_file(resolved_input),
                "manual CSV import",
            ),
        )
        connection.execute(
            """
            INSERT INTO curated_versions (
              curated_version_id,
              dataset_id,
              schema_version_id,
              version_type,
              snapshot_id,
              path,
              as_of_date,
              created_at,
              source_batch_ids,
              row_count,
              checksum,
              status
            )
            VALUES (?, ?, ?, 'current', NULL, ?, ?, ?, ?, ?, ?, 'active')
            ON CONFLICT(curated_version_id) DO UPDATE SET
              schema_version_id = excluded.schema_version_id,
              path = excluded.path,
              as_of_date = excluded.as_of_date,
              created_at = excluded.created_at,
              source_batch_ids = excluded.source_batch_ids,
              row_count = excluded.row_count,
              checksum = excluded.checksum,
              status = excluded.status
            """,
            (
                curated_version_id,
                dataset_id,
                f"{dataset_id}_schema_v001",
                str(output_path),
                as_of_date,
                now,
                json.dumps([source_batch_id]),
                row_count,
                checksum,
            ),
        )

    return CuratedImportResult(
        True,
        f"imported {row_count} rows into curated current: {dataset_id}",
        output_path,
        row_count,
    )


def inspect_curated(config_path: Path, dataset_id: str) -> CuratedInspectionResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
              cv.curated_version_id,
              cv.schema_version_id,
              cv.path,
              cv.as_of_date,
              cv.created_at,
              cv.row_count,
              cv.checksum,
              cv.status,
              sv.schema_file_path,
              sv.schema_hash
            FROM curated_versions cv
            JOIN schema_versions sv ON sv.schema_version_id = cv.schema_version_id
            WHERE cv.dataset_id = ? AND cv.version_type = 'current'
            ORDER BY cv.created_at DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        fields = connection.execute(
            """
            SELECT field_name
            FROM schema_fields
            WHERE schema_version_id = (
              SELECT schema_version_id
              FROM schema_versions
              WHERE dataset_id = ? AND status = 'active'
              ORDER BY version DESC
              LIMIT 1
            )
            ORDER BY rowid
            """,
            (dataset_id,),
        ).fetchall()

    if row is None:
        return CuratedInspectionResult(False, f"no current curated version found for dataset: {dataset_id}")

    (
        curated_version_id,
        schema_version_id,
        output_path,
        as_of_date,
        created_at,
        metadata_row_count,
        checksum,
        status,
        schema_file_path,
        schema_hash,
    ) = row
    parquet_path = Path(str(output_path))
    if not parquet_path.exists():
        return CuratedInspectionResult(False, f"curated Parquet path is missing: {parquet_path}")

    frame = pl.read_parquet(parquet_path, n_rows=5)
    actual_row_count = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    field_names = [field[0] for field in fields]
    sample_columns = ", ".join(frame.columns)
    message = "\n".join(
        [
            f"dataset: {dataset_id}",
            f"curated_version_id: {curated_version_id}",
            f"schema_version_id: {schema_version_id}",
            f"schema_file_path: {schema_file_path}",
            f"schema_hash: {schema_hash}",
            f"path: {parquet_path}",
            f"as_of_date: {as_of_date}",
            f"created_at: {created_at}",
            f"status: {status}",
            f"metadata_row_count: {metadata_row_count}",
            f"actual_row_count: {actual_row_count}",
            f"checksum: {checksum}",
            f"registered_fields: {len(field_names)}",
            f"parquet_columns: {sample_columns}",
        ]
    )
    return CuratedInspectionResult(True, message)


def _load_registered_schema(config: StorageConfig, dataset_id: str) -> list[dict[str, object]]:
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT schema_version_id
            FROM schema_versions
            WHERE dataset_id = ? AND status = 'active'
            ORDER BY version DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"schema is not registered for dataset: {dataset_id}")
        fields = connection.execute(
            """
            SELECT field_name, field_type, nullable, is_primary_key, is_partition_key
            FROM schema_fields
            WHERE schema_version_id = ?
            ORDER BY rowid
            """,
            (row[0],),
        ).fetchall()

    return [
        {
            "name": field_name,
            "type": field_type,
            "nullable": bool(nullable),
            "primary_key": bool(is_primary_key),
            "partition_key": bool(is_partition_key),
        }
        for field_name, field_type, nullable, is_primary_key, is_partition_key in fields
    ]


def _enrich_frame(
    frame: pl.DataFrame,
    schema: list[dict[str, object]],
    source: str,
    source_batch_id: str,
    data_version: str,
    created_at: str,
) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "source": source,
        "source_batch_id": source_batch_id,
        "data_version": data_version,
        "created_at": created_at,
    }
    for field_name, value in defaults.items():
        if _schema_has_field(schema, field_name) and field_name not in frame.columns:
            frame = frame.with_columns(pl.lit(value).alias(field_name))

    if "trade_date" in frame.columns:
        frame = _derive_year_month(frame, "trade_date", "trade_year", "trade_month", schema)
    if "event_date" in frame.columns:
        frame = _derive_year_month(frame, "event_date", "event_year", "event_month", schema)

    return frame


def _derive_year_month(
    frame: pl.DataFrame,
    date_column: str,
    year_column: str,
    month_column: str,
    schema: list[dict[str, object]],
) -> pl.DataFrame:
    parsed_date = pl.col(date_column).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    expressions = []
    if _schema_has_field(schema, year_column) and year_column not in frame.columns:
        expressions.append(parsed_date.dt.year().alias(year_column))
    if _schema_has_field(schema, month_column) and month_column not in frame.columns:
        expressions.append(parsed_date.dt.month().alias(month_column))
    if expressions:
        frame = frame.with_columns(expressions)
    return frame


def _validate_required_columns(frame: pl.DataFrame, schema: list[dict[str, object]]) -> str | None:
    missing = [
        str(field["name"])
        for field in schema
        if field["name"] not in frame.columns and not bool(field["nullable"])
    ]
    if missing:
        return "curated import failed; missing columns: " + ", ".join(missing)
    non_nullable = [str(field["name"]) for field in schema if not bool(field["nullable"])]
    null_counts = frame.select(pl.col(non_nullable).null_count()).row(0, named=True)
    null_columns = [column for column, count in null_counts.items() if count]
    if null_columns:
        return "curated import failed; non-nullable columns contain nulls: " + ", ".join(null_columns)
    return None


def _fill_missing_nullable_columns(frame: pl.DataFrame, schema: list[dict[str, object]]) -> pl.DataFrame:
    expressions = [
        pl.lit(None).alias(str(field["name"]))
        for field in schema
        if bool(field["nullable"]) and field["name"] not in frame.columns
    ]
    if expressions:
        frame = frame.with_columns(expressions)
    return frame


def _schema_has_field(schema: list[dict[str, object]], field_name: str) -> bool:
    return any(field["name"] == field_name for field in schema)


def _resolve_input_path(config: StorageConfig, input_path: Path) -> Path:
    if input_path.is_absolute():
        return input_path.resolve()
    return (config.project_root / input_path).resolve()


def _source_batch_id(dataset_id: str, source: str, input_path: Path, created_at: str) -> str:
    digest = hashlib.sha256(f"{input_path}:{created_at}".encode("utf-8")).hexdigest()[:12]
    return f"{source}_{dataset_id}_{digest}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
