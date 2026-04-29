"""Data snapshot manifest helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from stock_picker.config import load_storage_config
from stock_picker.quality import check_curated_quality
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class SnapshotResult:
    ok: bool
    message: str
    snapshot_id: str | None = None


def create_snapshot(
    config_path: Path,
    as_of_date: str,
    config_version: str = "manual_config_v001",
) -> SnapshotResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)

    quality = check_curated_quality(config_path)
    if not quality.ok:
        return SnapshotResult(False, "snapshot creation failed; " + quality.message)

    curated_versions = _load_current_curated_versions(config.metadata_sqlite_path)
    if not curated_versions:
        return SnapshotResult(False, "snapshot creation failed: no current curated versions found")

    snapshot_id = _next_snapshot_id(config.metadata_sqlite_path, as_of_date)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "curated_versions": {
            row["dataset_id"]: {
                "curated_version_id": row["curated_version_id"],
                "schema_version_id": row["schema_version_id"],
                "path": row["path"],
                "row_count": row["row_count"],
                "checksum": row["checksum"],
                "as_of_date": row["as_of_date"],
            }
            for row in curated_versions
        },
        "quality": {
            "ok": quality.ok,
            "message": quality.message,
        },
        "coverage": {
            "dataset_count": len(curated_versions),
            "total_row_count": sum(int(row["row_count"] or 0) for row in curated_versions),
        },
    }

    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO snapshot_manifests (
              snapshot_id,
              as_of_date,
              created_at,
              data_frequency,
              config_version,
              manifest_json,
              notes
            )
            VALUES (?, ?, ?, 'daily', ?, ?, ?)
            """,
            (
                snapshot_id,
                as_of_date,
                created_at,
                config_version,
                json.dumps(manifest, sort_keys=True),
                "logical current curated snapshot",
            ),
        )

    return SnapshotResult(True, f"created snapshot: {snapshot_id}", snapshot_id)


def inspect_snapshot(config_path: Path, snapshot_id: str) -> SnapshotResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT as_of_date, created_at, data_frequency, config_version, manifest_json, notes
            FROM snapshot_manifests
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()

    if row is None:
        return SnapshotResult(False, f"snapshot not found: {snapshot_id}")

    as_of_date, created_at, data_frequency, config_version, manifest_json, notes = row
    manifest = json.loads(manifest_json)
    curated_versions = manifest.get("curated_versions", {})
    coverage = manifest.get("coverage", {})
    quality = manifest.get("quality", {})
    dataset_lines = [
        f"- {dataset_id}: {details.get('curated_version_id')} rows={details.get('row_count')}"
        for dataset_id, details in sorted(curated_versions.items())
    ]
    message = "\n".join(
        [
            f"snapshot_id: {snapshot_id}",
            f"as_of_date: {as_of_date}",
            f"created_at: {created_at}",
            f"data_frequency: {data_frequency}",
            f"config_version: {config_version}",
            f"notes: {notes}",
            f"quality_ok: {quality.get('ok')}",
            f"quality_message: {quality.get('message')}",
            f"dataset_count: {coverage.get('dataset_count')}",
            f"total_row_count: {coverage.get('total_row_count')}",
            "curated_versions:",
            *dataset_lines,
        ]
    )
    return SnapshotResult(True, message, snapshot_id)


def _load_current_curated_versions(sqlite_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            SELECT
              dataset_id,
              curated_version_id,
              schema_version_id,
              path,
              as_of_date,
              row_count,
              checksum
            FROM curated_versions
            WHERE version_type = 'current' AND status = 'active'
            ORDER BY dataset_id
            """
        ).fetchall()
    return [
        {
            "dataset_id": dataset_id,
            "curated_version_id": curated_version_id,
            "schema_version_id": schema_version_id,
            "path": path,
            "as_of_date": row_as_of_date,
            "row_count": row_count,
            "checksum": checksum,
        }
        for dataset_id, curated_version_id, schema_version_id, path, row_as_of_date, row_count, checksum in rows
    ]


def _next_snapshot_id(sqlite_path: Path, as_of_date: str) -> str:
    prefix = f"snapshot_{as_of_date.replace('-', '')}_"
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            "SELECT snapshot_id FROM snapshot_manifests WHERE snapshot_id LIKE ?",
            (prefix + "%",),
        ).fetchall()
    max_suffix = 0
    for (snapshot_id,) in rows:
        suffix = str(snapshot_id).removeprefix(prefix)
        if suffix.isdigit():
            max_suffix = max(max_suffix, int(suffix))
    return f"{prefix}{max_suffix + 1:03d}"
