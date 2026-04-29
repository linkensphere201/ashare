"""Curated data quality checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    message: str


def check_curated_quality(config_path: Path, dataset_ids: list[str] | None = None) -> QualityResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    checks = _load_quality_targets(config.metadata_sqlite_path, dataset_ids)
    if not checks:
        return QualityResult(False, "quality check failed: no curated datasets found")

    issues: list[str] = []
    checked: list[str] = []
    for check in checks:
        dataset_id = str(check["dataset_id"])
        dataset_issues = _check_one_dataset(check)
        if dataset_issues:
            issues.extend(f"{dataset_id}: {issue}" for issue in dataset_issues)
        checked.append(dataset_id)

    if issues:
        return QualityResult(
            False,
            "quality check failed:\n" + "\n".join(f"- {issue}" for issue in issues),
        )
    return QualityResult(True, f"quality check succeeded: {len(checked)} datasets checked")


def _load_quality_targets(
    sqlite_path: Path,
    dataset_ids: list[str] | None,
) -> list[dict[str, object]]:
    with sqlite3.connect(sqlite_path) as connection:
        if dataset_ids:
            requested = [(dataset_id,) for dataset_id in dataset_ids]
            connection.execute("CREATE TEMP TABLE requested_datasets(dataset_id TEXT PRIMARY KEY)")
            connection.executemany("INSERT INTO requested_datasets(dataset_id) VALUES (?)", requested)
            rows = connection.execute(
                """
                SELECT
                  d.dataset_id,
                  sv.schema_version_id,
                  cv.path,
                  cv.row_count,
                  cv.checksum
                FROM requested_datasets rd
                LEFT JOIN datasets d ON d.dataset_id = rd.dataset_id
                LEFT JOIN schema_versions sv ON sv.dataset_id = rd.dataset_id AND sv.status = 'active'
                LEFT JOIN curated_versions cv
                  ON cv.dataset_id = rd.dataset_id
                 AND cv.version_type = 'current'
                 AND cv.status = 'active'
                ORDER BY rd.dataset_id
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT
                  d.dataset_id,
                  sv.schema_version_id,
                  cv.path,
                  cv.row_count,
                  cv.checksum
                FROM datasets d
                LEFT JOIN schema_versions sv ON sv.dataset_id = d.dataset_id AND sv.status = 'active'
                LEFT JOIN curated_versions cv
                  ON cv.dataset_id = d.dataset_id
                 AND cv.version_type = 'current'
                 AND cv.status = 'active'
                WHERE d.layer = 'curated' AND d.status = 'active'
                ORDER BY d.dataset_id
                """
            ).fetchall()

        targets: list[dict[str, object]] = []
        for dataset_id, schema_version_id, path, row_count, checksum in rows:
            fields = []
            if schema_version_id:
                fields = connection.execute(
                    """
                    SELECT field_name, nullable, is_primary_key
                    FROM schema_fields
                    WHERE schema_version_id = ?
                    ORDER BY rowid
                    """,
                    (schema_version_id,),
                ).fetchall()
            targets.append(
                {
                    "dataset_id": dataset_id,
                    "schema_version_id": schema_version_id,
                    "path": path,
                    "row_count": row_count,
                    "checksum": checksum,
                    "fields": fields,
                }
            )
    return targets


def _check_one_dataset(check: dict[str, object]) -> list[str]:
    issues: list[str] = []
    if check["dataset_id"] is None:
        return ["dataset is not registered in catalog"]
    if check["schema_version_id"] is None:
        issues.append("schema version is not registered")
    if check["path"] is None:
        issues.append("current curated version is missing")
        return issues

    parquet_path = Path(str(check["path"]))
    if not parquet_path.exists():
        issues.append(f"Parquet path is missing: {parquet_path}")
        return issues

    try:
        frame = pl.read_parquet(parquet_path)
    except Exception as error:  # pragma: no cover - defensive for corrupted files.
        return [f"Parquet read failed: {error}"]

    actual_row_count = frame.height
    metadata_row_count = check["row_count"]
    if actual_row_count == 0:
        issues.append("Parquet row count is 0")
    if metadata_row_count is not None and int(metadata_row_count) != actual_row_count:
        issues.append(
            f"metadata row_count {metadata_row_count} does not match actual row count {actual_row_count}"
        )

    fields = [
        {
            "name": field_name,
            "nullable": bool(nullable),
            "primary_key": bool(is_primary_key),
        }
        for field_name, nullable, is_primary_key in check["fields"]
    ]
    expected_columns = [str(field["name"]) for field in fields]
    actual_columns = frame.columns
    missing_columns = [column for column in expected_columns if column not in actual_columns]
    extra_columns = [column for column in actual_columns if column not in expected_columns]
    if missing_columns:
        issues.append("missing schema columns: " + ", ".join(missing_columns))
    if extra_columns:
        issues.append("extra Parquet columns: " + ", ".join(extra_columns))

    present_non_nullable = [
        str(field["name"])
        for field in fields
        if not bool(field["nullable"]) and field["name"] in actual_columns
    ]
    if present_non_nullable:
        null_counts = frame.select(pl.col(present_non_nullable).null_count()).row(0, named=True)
        for column, null_count in null_counts.items():
            if null_count:
                issues.append(f"non-nullable field {column} contains {null_count} nulls")

    primary_key_columns = [
        str(field["name"])
        for field in fields
        if bool(field["primary_key"]) and field["name"] in actual_columns
    ]
    if primary_key_columns:
        duplicate_count = (
            frame.group_by(primary_key_columns)
            .len()
            .filter(pl.col("len") > 1)
            .height
        )
        if duplicate_count:
            issues.append(
                "primary key duplicates found for "
                + ", ".join(primary_key_columns)
                + f": {duplicate_count} duplicate keys"
            )

    return issues
