"""Operator-facing display helpers for curated data and run lineage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from prettytable import PrettyTable

from stock_picker.config import load_storage_config
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class DisplayResult:
    ok: bool
    message: str


def preview_curated(
    config_path: Path,
    dataset_id: str,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    columns: str | None = None,
    limit: int = 20,
) -> DisplayResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    version = _load_current_curated_version(config.metadata_sqlite_path, dataset_id)
    if version is None:
        return DisplayResult(False, f"no current curated version found for dataset: {dataset_id}")
    path = Path(str(version["path"]))
    if not path.exists():
        return DisplayResult(False, f"curated Parquet path is missing: {path}")
    if limit <= 0:
        return DisplayResult(False, "preview failed; limit must be positive")

    frame = pl.read_parquet(path)
    if symbol and "symbol" in frame.columns:
        frame = frame.filter(pl.col("symbol") == symbol)
    date_column = _first_existing_column(frame, ["trade_date", "as_of_date", "event_date", "business_date"])
    if date_column:
        if start_date:
            frame = frame.filter(pl.col(date_column).cast(pl.Utf8) >= start_date)
        if end_date:
            frame = frame.filter(pl.col(date_column).cast(pl.Utf8) <= end_date)

    requested_columns = _parse_columns(columns) if columns else _default_preview_columns(dataset_id, frame)
    frame = _join_security_name(config, frame, requested_columns)
    missing_columns = [column for column in requested_columns if column not in frame.columns]
    if missing_columns:
        return DisplayResult(False, "preview failed; missing columns: " + ", ".join(missing_columns))

    sort_columns = [column for column in [date_column, "symbol"] if column in frame.columns]
    if sort_columns:
        frame = frame.sort(sort_columns, descending=[True] + [False] * (len(sort_columns) - 1))
    preview = frame.select(requested_columns).head(limit)
    if preview.is_empty():
        return DisplayResult(True, f"no rows matched curated preview for dataset: {dataset_id}")

    return DisplayResult(
        True,
        "\n".join(
            [
                f"curated_preview: {dataset_id}",
                f"curated_version_id: {version['curated_version_id']}",
                f"as_of_date: {version['as_of_date']}",
                f"row_count: {version['row_count']}",
                "rows:",
                _frame_to_pretty_table(preview),
            ]
        ),
    )


def list_runs(config_path: Path, limit: int = 20) -> DisplayResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    if limit <= 0:
        return DisplayResult(False, "run listing failed; limit must be positive")

    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM data_batches").fetchone()[0]
        rows = connection.execute(
            """
            SELECT batch_id, source, dataset_name, business_date, row_count, status, schema_hash
            FROM data_batches
            ORDER BY retrieved_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        return DisplayResult(True, "no data import runs found")

    frame = pl.DataFrame(
        rows,
        schema=[
            "batch_id",
            "source",
            "dataset",
            "business_date",
            "row_count",
            "status",
            "schema_hash",
        ],
        orient="row",
    ).with_columns(
        pl.col("schema_hash").map_elements(_short_hash, return_dtype=pl.Utf8).alias("schema_hash")
    )
    return DisplayResult(
        True,
        "\n".join(
            [
                "data_import_runs:",
                f"total_import_runs: {total}",
                f"displayed_runs: {frame.height}",
                "runs:",
                _frame_to_pretty_table(frame),
            ]
        ),
    )


def inspect_run(config_path: Path, batch_id: str) -> DisplayResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
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
            FROM data_batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()
        curated_rows = connection.execute(
            """
            SELECT curated_version_id, dataset_id, schema_version_id, as_of_date, row_count, checksum, status
            FROM curated_versions
            WHERE source_batch_ids LIKE ?
            ORDER BY created_at DESC
            """,
            (f"%{batch_id}%",),
        ).fetchall()
        snapshot_rows = connection.execute(
            """
            SELECT snapshot_id, as_of_date, created_at, data_frequency, config_version, manifest_json
            FROM snapshot_manifests
            ORDER BY created_at DESC
            """
        ).fetchall()
    if row is None:
        return DisplayResult(False, f"data import run not found: {batch_id}")

    snapshot_matches = _snapshot_matches(curated_rows, snapshot_rows)
    curated_lines = [
        [curated_version_id, dataset_id, schema_version_id, as_of_date, row_count, status, checksum]
        for curated_version_id, dataset_id, schema_version_id, as_of_date, row_count, checksum, status in curated_rows
    ]
    snapshot_lines = [
        [snapshot_id, as_of_date, created_at, data_frequency, config_version]
        for snapshot_id, as_of_date, created_at, data_frequency, config_version in snapshot_matches
    ]
    (
        batch_id_value,
        source,
        dataset_name,
        retrieved_at,
        business_date,
        raw_path,
        file_format,
        row_count,
        schema_hash,
        content_checksum,
        status,
        notes,
    ) = row
    return DisplayResult(
        True,
        "\n".join(
            [
                f"batch_id: {batch_id_value}",
                f"source: {source}",
                f"dataset: {dataset_name}",
                f"retrieved_at: {retrieved_at}",
                f"business_date: {business_date}",
                f"raw_path: {raw_path}",
                f"format: {file_format}",
                f"row_count: {row_count}",
                f"schema_hash: {schema_hash}",
                f"content_checksum: {content_checksum}",
                f"status: {status}",
                f"notes: {notes}",
                "linked_curated_versions:",
                (
                    _rows_to_pretty_table(
                        ["curated_version_id", "dataset", "schema_version_id", "as_of_date", "row_count", "status", "checksum"],
                        curated_lines,
                    )
                    if curated_lines
                    else "none"
                ),
                "linked_snapshots:",
                (
                    _rows_to_pretty_table(
                        ["snapshot_id", "as_of_date", "created_at", "frequency", "config_version"],
                        snapshot_lines,
                    )
                    if snapshot_lines
                    else "none"
                ),
            ]
        ),
    )


def _load_current_curated_version(sqlite_path: Path, dataset_id: str) -> dict[str, object] | None:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT curated_version_id, path, as_of_date, row_count
            FROM curated_versions
            WHERE dataset_id = ? AND version_type = 'current' AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
    if row is None:
        return None
    curated_version_id, path, as_of_date, row_count = row
    return {
        "curated_version_id": curated_version_id,
        "path": path,
        "as_of_date": as_of_date,
        "row_count": row_count,
    }


def _parse_columns(columns: str) -> list[str]:
    return [column.strip() for column in columns.split(",") if column.strip()]


def _default_preview_columns(dataset_id: str, frame: pl.DataFrame) -> list[str]:
    if dataset_id == "daily_prices":
        return [column for column in ["symbol", "name", "trade_date", "pct_change", "close"] if column == "name" or column in frame.columns]
    if dataset_id == "capital_flow_or_chip":
        return [
            column
            for column in ["symbol", "name", "trade_date", "main_net_inflow_rate", "close_profit_ratio"]
            if column == "name" or column in frame.columns
        ]
    return frame.columns[: min(8, len(frame.columns))]


def _join_security_name(config, frame: pl.DataFrame, requested_columns: list[str]) -> pl.DataFrame:
    if "name" not in requested_columns or "name" in frame.columns or "symbol" not in frame.columns:
        return frame
    security_path = config.current_curated_root / "security_master" / "part-000.parquet"
    if not security_path.exists():
        return frame
    security = pl.read_parquet(security_path, columns=["symbol", "name"])
    return frame.join(security, on="symbol", how="left")


def _first_existing_column(frame: pl.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _snapshot_matches(
    curated_rows: list[tuple[object, ...]],
    snapshot_rows: list[tuple[object, ...]],
) -> list[tuple[object, ...]]:
    curated_version_ids = {str(row[0]) for row in curated_rows}
    matches = []
    for snapshot_id, as_of_date, created_at, data_frequency, config_version, manifest_json in snapshot_rows:
        manifest = json.loads(str(manifest_json))
        snapshot_curated_ids = {
            str(details.get("curated_version_id"))
            for details in manifest.get("curated_versions", {}).values()
        }
        if curated_version_ids.intersection(snapshot_curated_ids):
            matches.append((snapshot_id, as_of_date, created_at, data_frequency, config_version))
    return matches


def _frame_to_pretty_table(frame: pl.DataFrame) -> str:
    rows = [list(row) for row in frame.iter_rows()]
    return _rows_to_pretty_table(frame.columns, rows)


def _rows_to_pretty_table(field_names: list[str], rows: list[list[object]]) -> str:
    table = PrettyTable()
    table.field_names = field_names
    table.align = "l"
    table.max_width = 48
    for row in rows:
        table.add_row(["" if value is None else value for value in row])
    return table.get_string()


def _short_hash(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= 16:
        return text
    return text[:12] + "..."
