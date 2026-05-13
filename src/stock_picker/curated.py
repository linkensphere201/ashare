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


RAW_TO_CURATED_DATASET = {
    "security_master": "security_master",
    "trading_calendar": "trading_calendar",
    "daily_prices": "daily_prices",
    "adj_factor": "daily_prices",
    "index_daily": "daily_prices",
    "daily_basic": "daily_prices",
    "stk_limit": "daily_prices",
    "suspend_d": "daily_prices",
    "moneyflow_dc": "capital_flow_or_chip",
    "cyq_perf": "capital_flow_or_chip",
    "index_classify": "industry_classification",
    "sw_daily": "industry_daily",
}
SUPPLEMENTAL_DAILY_PRICE_DATASETS = {"adj_factor", "daily_basic", "stk_limit"}


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
        _deactivate_previous_current_versions(connection, dataset_id, curated_version_id)
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


def promote_raw_batch(
    config_path: Path,
    source: str,
    dataset: str,
    as_of_date: str | None = None,
    batch_id: str | None = None,
) -> CuratedImportResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    batch = _load_raw_batch(config.metadata_sqlite_path, source, dataset, as_of_date, batch_id)
    if batch is None:
        selector = batch_id or f"{source}/{dataset}/{as_of_date or 'latest'}"
        return CuratedImportResult(False, f"raw batch not found: {selector}")

    curated_dataset = RAW_TO_CURATED_DATASET.get(batch["dataset_name"])
    if curated_dataset is None:
        return CuratedImportResult(False, f"no curated mapping for raw dataset: {batch['dataset_name']}")

    raw_path = Path(str(batch["raw_path"]))
    if not raw_path.exists():
        return CuratedImportResult(False, f"raw batch file not found: {raw_path}")

    if batch["dataset_name"] == "suspend_d":
        return _promote_suspend_batch(config, batch)

    schema = _load_registered_schema(config, curated_dataset)
    raw_frame = pl.read_csv(raw_path)
    mapped = _map_tushare_raw_to_curated(raw_frame, str(batch["dataset_name"]))
    if batch["dataset_name"] in SUPPLEMENTAL_DAILY_PRICE_DATASETS:
        mapped = _filter_to_existing_current_keys(config, curated_dataset, mapped, ["symbol", "trade_date"])
    now = datetime.now(UTC).isoformat(timespec="seconds")
    data_version = str(batch["business_date"] or now[:10])
    mapped = _enrich_frame(mapped, schema, source, str(batch["batch_id"]), data_version, now)
    primary_keys = _primary_key_columns(schema)
    overlap_summary = _current_overlap_summary(config, curated_dataset, mapped, primary_keys)
    mapped = _merge_with_existing_current(config, curated_dataset, mapped, schema)
    mapped = _fill_missing_nullable_columns(mapped, schema)
    validation_error = _validate_required_columns(mapped, schema)
    if validation_error:
        return CuratedImportResult(False, validation_error)
    existing_source_batch_ids = _load_current_source_batch_ids(config.metadata_sqlite_path, curated_dataset)
    source_batch_ids = _merge_source_batch_ids(existing_source_batch_ids, [str(batch["batch_id"])])
    return _write_curated_current(
        config=config,
        dataset_id=curated_dataset,
        frame=mapped,
        schema=schema,
        as_of_date=data_version,
        source_batch_ids=source_batch_ids,
        created_at=now,
        notes=_curated_notes(
            source_batch_id=str(batch["batch_id"]),
            raw_dataset=str(batch["dataset_name"]),
            overlap_summary=overlap_summary,
        ),
    )


def promote_raw_run(
    config_path: Path,
    run_id: str,
    dataset: str | None = None,
) -> CuratedImportResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    batches = _load_raw_batches_for_run(config.metadata_sqlite_path, run_id, dataset)
    if not batches:
        selector = f"{run_id}/{dataset or 'all'}"
        return CuratedImportResult(False, f"raw run batches not found: {selector}")

    grouped: dict[str, list[dict[str, object]]] = {}
    for batch in batches:
        grouped.setdefault(str(batch["dataset_name"]), []).append(batch)

    results: list[CuratedImportResult] = []
    for raw_dataset, raw_batches in grouped.items():
        curated_dataset = RAW_TO_CURATED_DATASET.get(raw_dataset)
        if curated_dataset is None:
            return CuratedImportResult(False, f"no curated mapping for raw dataset: {raw_dataset}")
        if raw_dataset == "suspend_d":
            for batch in raw_batches:
                result = _promote_suspend_batch(config, batch)
                if not result.ok:
                    return result
                results.append(result)
            continue
        schema = _load_registered_schema(config, curated_dataset)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        data_version = str(raw_batches[-1]["business_date"] or now[:10])
        mapped_frames: list[pl.DataFrame] = []
        incoming_batch_ids: list[str] = []
        for batch in raw_batches:
            raw_path = Path(str(batch["raw_path"]))
            if not raw_path.exists():
                return CuratedImportResult(False, f"raw batch file not found: {raw_path}")
            incoming_batch_ids.append(str(batch["batch_id"]))
            raw_frame = pl.read_csv(raw_path)
            mapped = _map_tushare_raw_to_curated(raw_frame, raw_dataset)
            mapped = _enrich_frame(mapped, schema, str(batch["source"]), str(batch["batch_id"]), data_version, now)
            mapped_frames.append(mapped)

        combined = pl.concat(mapped_frames, how="diagonal_relaxed")
        primary_keys = _primary_key_columns(schema)
        overlap_summary = _current_overlap_summary(config, curated_dataset, combined, primary_keys)
        combined = _merge_with_existing_current(config, curated_dataset, combined, schema)
        combined = _fill_missing_nullable_columns(combined, schema)
        validation_error = _validate_required_columns(combined, schema)
        if validation_error:
            return CuratedImportResult(False, validation_error)
        existing_source_batch_ids = _load_current_source_batch_ids(config.metadata_sqlite_path, curated_dataset)
        source_batch_ids = _merge_source_batch_ids(existing_source_batch_ids, incoming_batch_ids)
        results.append(
            _write_curated_current(
                config=config,
                dataset_id=curated_dataset,
                frame=combined,
                schema=schema,
                as_of_date=data_version,
                source_batch_ids=source_batch_ids,
                created_at=now,
                notes=_curated_bulk_notes(
                    run_id=run_id,
                    raw_dataset=raw_dataset,
                    source_batch_ids=incoming_batch_ids,
                    overlap_summary=overlap_summary,
                ),
            )
        )

    if len(results) == 1:
        return results[0]
    total_rows = sum(result.row_count for result in results)
    message = "promoted raw run: " + "; ".join(result.message for result in results)
    return CuratedImportResult(True, message, results[-1].output_path, total_rows)


def _promote_suspend_batch(config: StorageConfig, batch: dict[str, object]) -> CuratedImportResult:
    raw_path = Path(str(batch["raw_path"]))
    if not raw_path.exists():
        return CuratedImportResult(False, f"raw batch file not found: {raw_path}")

    raw_frame = pl.read_csv(raw_path)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    data_version = str(batch["business_date"] or now[:10])
    daily_schema = _load_registered_schema(config, "daily_prices")
    risk_schema = _load_registered_schema(config, "risk_events")
    daily_frame = _map_tushare_raw_to_curated(raw_frame, "suspend_d")
    daily_frame = _filter_to_existing_current_keys(config, "daily_prices", daily_frame, ["symbol", "trade_date"])
    risk_frame = _map_tushare_suspend_to_risk_events(raw_frame)

    daily_frame = _enrich_frame(daily_frame, daily_schema, str(batch["source"]), str(batch["batch_id"]), data_version, now)
    risk_frame = _enrich_frame(risk_frame, risk_schema, str(batch["source"]), str(batch["batch_id"]), data_version, now)
    risk_frame = _fill_missing_nullable_columns(risk_frame, risk_schema)

    daily_frame = _merge_with_existing_current(config, "daily_prices", daily_frame, daily_schema)
    daily_frame = _fill_missing_nullable_columns(daily_frame, daily_schema)
    daily_error = _validate_required_columns(daily_frame, daily_schema)
    if daily_error:
        return CuratedImportResult(False, daily_error)
    risk_error = _validate_required_columns(risk_frame, risk_schema)
    if risk_error:
        return CuratedImportResult(False, risk_error)

    daily_batches = _merge_source_batch_ids(
        _load_current_source_batch_ids(config.metadata_sqlite_path, "daily_prices"),
        [str(batch["batch_id"])],
    )
    risk_batches = _merge_source_batch_ids(
        _load_current_source_batch_ids(config.metadata_sqlite_path, "risk_events"),
        [str(batch["batch_id"])],
    )
    daily_result = _write_curated_current(
        config=config,
        dataset_id="daily_prices",
        frame=daily_frame,
        schema=daily_schema,
        as_of_date=data_version,
        source_batch_ids=daily_batches,
        created_at=now,
        notes=_curated_notes(str(batch["batch_id"]), "suspend_d", {}),
    )
    if not daily_result.ok:
        return daily_result
    risk_result = _write_curated_current(
        config=config,
        dataset_id="risk_events",
        frame=_merge_with_existing_current(config, "risk_events", risk_frame, risk_schema),
        schema=risk_schema,
        as_of_date=data_version,
        source_batch_ids=risk_batches,
        created_at=now,
        notes=_curated_notes(str(batch["batch_id"]), "suspend_d", {}),
    )
    if not risk_result.ok:
        return risk_result
    return CuratedImportResult(
        True,
        f"promoted suspend_d into curated current: daily_prices rows={daily_result.row_count}; risk_events rows={risk_result.row_count}",
        daily_result.output_path,
        daily_result.row_count + risk_result.row_count,
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
              cv.notes,
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
        notes,
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
            f"notes: {notes}",
            f"registered_fields: {len(field_names)}",
            f"parquet_columns: {sample_columns}",
        ]
    )
    return CuratedInspectionResult(True, message)


def _write_curated_current(
    config: StorageConfig,
    dataset_id: str,
    frame: pl.DataFrame,
    schema: list[dict[str, object]],
    as_of_date: str | None,
    source_batch_ids: list[str],
    created_at: str,
    notes: str | None = None,
) -> CuratedImportResult:
    output_dir = config.current_curated_root / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "part-000.parquet"
    frame.select([field["name"] for field in schema]).write_parquet(output_path)
    row_count = frame.height
    checksum = _sha256_file(output_path)
    data_version = as_of_date or created_at[:10]
    curated_version_id = f"{dataset_id}_current_{data_version.replace('-', '')}"
    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        _deactivate_previous_current_versions(connection, dataset_id, curated_version_id)
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
              status,
              notes
            )
            VALUES (?, ?, ?, 'current', NULL, ?, ?, ?, ?, ?, ?, 'active', ?)
            ON CONFLICT(curated_version_id) DO UPDATE SET
              schema_version_id = excluded.schema_version_id,
              path = excluded.path,
              as_of_date = excluded.as_of_date,
              created_at = excluded.created_at,
              source_batch_ids = excluded.source_batch_ids,
              row_count = excluded.row_count,
              checksum = excluded.checksum,
              status = excluded.status,
              notes = excluded.notes
            """,
            (
                curated_version_id,
                dataset_id,
                f"{dataset_id}_schema_v001",
                str(output_path),
                as_of_date,
                created_at,
                json.dumps(source_batch_ids),
                row_count,
                checksum,
                notes,
            ),
        )

    return CuratedImportResult(
        True,
        f"promoted {row_count} rows into curated current: {dataset_id}",
        output_path,
        row_count,
    )


def _deactivate_previous_current_versions(
    connection: sqlite3.Connection,
    dataset_id: str,
    curated_version_id: str,
) -> None:
    connection.execute(
        """
        UPDATE curated_versions
        SET status = 'inactive'
        WHERE dataset_id = ?
          AND version_type = 'current'
          AND status = 'active'
          AND curated_version_id <> ?
        """,
        (dataset_id, curated_version_id),
    )


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


def _load_raw_batch(
    sqlite_path: Path,
    source: str,
    dataset: str,
    as_of_date: str | None,
    batch_id: str | None,
) -> dict[str, object] | None:
    with sqlite3.connect(sqlite_path) as connection:
        if batch_id:
            row = connection.execute(
                """
                SELECT batch_id, source, dataset_name, business_date, raw_path
                FROM data_batches
                WHERE batch_id = ? AND status = 'success'
                """,
                (batch_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT batch_id, source, dataset_name, business_date, raw_path
                FROM data_batches
                WHERE source = ?
                  AND dataset_name = ?
                  AND (? IS NULL OR business_date = ?)
                  AND status = 'success'
                ORDER BY retrieved_at DESC
                LIMIT 1
                """,
                (source, dataset, as_of_date, as_of_date),
            ).fetchone()
    if row is None:
        return None
    batch_id_value, source_value, dataset_name, business_date, raw_path = row
    return {
        "batch_id": batch_id_value,
        "source": source_value,
        "dataset_name": dataset_name,
        "business_date": business_date,
        "raw_path": raw_path,
    }


def _load_raw_batches_for_run(
    sqlite_path: Path,
    run_id: str,
    dataset: str | None,
) -> list[dict[str, object]]:
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            SELECT
              db.batch_id,
              db.source,
              db.dataset_name,
              db.business_date,
              db.raw_path,
              prt.trade_date,
              prt.symbol_start_offset
            FROM provider_run_tasks prt
            JOIN data_batches db ON db.batch_id = prt.raw_batch_id
            WHERE prt.run_id = ?
              AND prt.status = 'success'
              AND prt.raw_batch_id IS NOT NULL
              AND db.status = 'success'
              AND (? IS NULL OR db.dataset_name = ?)
            ORDER BY db.dataset_name, COALESCE(prt.trade_date, ''), COALESCE(prt.symbol_start_offset, -1), db.batch_id
            """,
            (run_id, dataset, dataset),
        ).fetchall()
    return [
        {
            "batch_id": batch_id,
            "source": source,
            "dataset_name": dataset_name,
            "business_date": business_date,
            "raw_path": raw_path,
        }
        for batch_id, source, dataset_name, business_date, raw_path, _trade_date, _symbol_start_offset in rows
    ]


def _map_tushare_raw_to_curated(frame: pl.DataFrame, dataset: str) -> pl.DataFrame:
    if dataset == "security_master":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                pl.col("symbol").alias("raw_symbol"),
                pl.col("ts_code").str.split(".").list.get(1).alias("exchange"),
                pl.lit("stock").alias("asset_type"),
                _market_segment_expr().alias("market_segment"),
                _market_segment_name_expr().alias("market_segment_name"),
                _parse_yyyymmdd("list_date").alias("list_date"),
                _parse_yyyymmdd("delist_date").alias("delist_date"),
                pl.lit("active").alias("status"),
            ]
        ).select(
            [
                "symbol",
                "raw_symbol",
                "exchange",
                "asset_type",
                "name",
                "market",
                "market_segment",
                "market_segment_name",
                "area",
                "industry",
                "list_date",
                "delist_date",
                "status",
            ]
        )
    if dataset == "trading_calendar":
        return frame.with_columns(
            [
                pl.lit("cn_a_share").alias("calendar_id"),
                _parse_yyyymmdd("cal_date").alias("trade_date"),
                (pl.col("is_open").cast(pl.Int64, strict=False) == 1).alias("is_trading_day"),
                _parse_yyyymmdd("pretrade_date").alias("previous_trade_date"),
                pl.lit(None).alias("next_trade_date"),
            ]
        ).select(["calendar_id", "trade_date", "is_trading_day", "previous_trade_date", "next_trade_date"])
    if dataset == "daily_prices":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.lit("stock").alias("asset_type"),
                pl.col("vol").alias("volume"),
                pl.col("pct_chg").alias("pct_change"),
            ]
        ).select(["symbol", "trade_date", "asset_type", "open", "high", "low", "close", "pre_close", "volume", "amount", "pct_change"])
    if dataset == "index_daily":
        volume_expr = pl.col("vol").alias("volume") if "vol" in frame.columns else pl.lit(None).alias("volume")
        amount_expr = pl.col("amount").alias("amount") if "amount" in frame.columns else pl.lit(None).alias("amount")
        pct_change_expr = pl.col("pct_chg").alias("pct_change") if "pct_chg" in frame.columns else pl.lit(None).alias("pct_change")
        pre_close_expr = pl.col("pre_close").alias("pre_close") if "pre_close" in frame.columns else pl.lit(None).alias("pre_close")
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.lit("index").alias("asset_type"),
                volume_expr,
                amount_expr,
                pct_change_expr,
                pre_close_expr,
            ]
        ).select(["symbol", "trade_date", "asset_type", "open", "high", "low", "close", "pre_close", "volume", "amount", "pct_change"])
    if dataset == "adj_factor":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("adj_factor").cast(pl.Float64, strict=False).alias("adj_factor"),
            ]
        ).select(["symbol", "trade_date", "adj_factor"])
    if dataset == "daily_basic":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("turnover_rate").cast(pl.Float64, strict=False).alias("turnover_rate"),
            ]
        ).select(["symbol", "trade_date", "turnover_rate"])
    if dataset == "stk_limit":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("up_limit").cast(pl.Float64, strict=False).alias("limit_up"),
                pl.col("down_limit").cast(pl.Float64, strict=False).alias("limit_down"),
            ]
        ).select(["symbol", "trade_date", "limit_up", "limit_down"])
    if dataset == "suspend_d":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                (pl.col("suspend_type") == "S").alias("is_suspended"),
            ]
        ).select(["symbol", "trade_date", "is_suspended"])
    if dataset == "moneyflow_dc":
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("net_amount").alias("main_net_inflow"),
                pl.col("net_amount_rate").alias("main_net_inflow_rate"),
                pl.lit("tushare_moneyflow_dc").alias("data_method"),
            ]
        ).select(["symbol", "trade_date", "main_net_inflow", "main_net_inflow_rate", "data_method"])
    if dataset == "cyq_perf":
        data_method = (
            pl.when(pl.col("provider_status") == "not_found")
            .then(pl.lit("tushare_cyq_perf:not_found"))
            .otherwise(pl.lit("tushare_cyq_perf"))
            if "provider_status" in frame.columns
            else pl.lit("tushare_cyq_perf")
        )
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("winner_rate").alias("close_profit_ratio"),
                data_method.alias("data_method"),
            ]
        ).select(["symbol", "trade_date", "close_profit_ratio", "data_method"])
    if dataset == "index_classify":
        parent_expr = pl.col("parent_code") if "parent_code" in frame.columns else pl.lit(None)
        src_expr = pl.col("src") if "src" in frame.columns else pl.lit("SW2021")
        return frame.with_columns(
            [
                pl.col("index_code").alias("index_code"),
                pl.col("industry_name").alias("industry_name"),
                pl.col("level").alias("level"),
                src_expr.alias("source_system"),
                parent_expr.alias("parent_code"),
            ]
        ).select(["index_code", "industry_name", "level", "source_system", "parent_code"])
    if dataset == "sw_daily":
        name_expr = pl.col("name").alias("industry_name") if "name" in frame.columns else pl.lit(None).alias("industry_name")
        open_expr = pl.col("open").alias("open") if "open" in frame.columns else pl.lit(None).alias("open")
        high_expr = pl.col("high").alias("high") if "high" in frame.columns else pl.lit(None).alias("high")
        low_expr = pl.col("low").alias("low") if "low" in frame.columns else pl.lit(None).alias("low")
        volume_expr = pl.col("vol").alias("volume") if "vol" in frame.columns else pl.lit(None).alias("volume")
        amount_expr = pl.col("amount").alias("amount") if "amount" in frame.columns else pl.lit(None).alias("amount")
        pct_change_expr = pl.col("pct_change").alias("pct_change") if "pct_change" in frame.columns else pl.col("pct_chg").alias("pct_change")
        pre_close_expr = pl.col("pre_close").alias("pre_close") if "pre_close" in frame.columns else pl.lit(None).alias("pre_close")
        return frame.with_columns(
            [
                pl.col("ts_code").alias("index_code"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                name_expr,
                open_expr,
                high_expr,
                low_expr,
                pct_change_expr,
                pre_close_expr,
                volume_expr,
                amount_expr,
            ]
        ).select(["index_code", "trade_date", "industry_name", "open", "high", "low", "close", "pre_close", "pct_change", "volume", "amount"])
    raise ValueError(f"unsupported raw dataset mapping: {dataset}")


def _map_tushare_suspend_to_risk_events(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            pl.col("ts_code").alias("symbol"),
            _parse_yyyymmdd("trade_date").alias("event_date"),
            pl.lit("suspension").alias("event_type"),
            pl.when(pl.col("suspend_type") == "S").then(pl.lit("blocker")).otherwise(pl.lit("info")).alias("severity"),
            (pl.col("suspend_type") == "S").alias("is_active"),
            pl.concat_str(
                [
                    pl.lit("Tushare suspend_d suspend_type="),
                    pl.col("suspend_type").cast(pl.Utf8),
                ]
            ).alias("description"),
        ]
    ).select(["symbol", "event_date", "event_type", "severity", "is_active", "description"])


def _parse_yyyymmdd(column: str) -> pl.Expr:
    if column in ("delist_date", "pretrade_date"):
        return (
            pl.when(pl.col(column).is_null() | (pl.col(column).cast(pl.Utf8) == ""))
            .then(None)
            .otherwise(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False))
        )
    return pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)


def _market_segment_expr() -> pl.Expr:
    exchange = pl.col("ts_code").str.split(".").list.get(1)
    return (
        pl.when((exchange == "SH") & (pl.col("market") == "主板"))
        .then(pl.lit("sh_main"))
        .when((exchange == "SH") & (pl.col("market") == "科创板"))
        .then(pl.lit("star"))
        .when((exchange == "SZ") & (pl.col("market") == "主板"))
        .then(pl.lit("sz_main"))
        .when((exchange == "SZ") & (pl.col("market") == "创业板"))
        .then(pl.lit("chinext"))
        .when((exchange == "BJ") | (pl.col("market") == "北交所"))
        .then(pl.lit("bj"))
        .otherwise(pl.lit("unknown"))
    )


def _market_segment_name_expr() -> pl.Expr:
    exchange = pl.col("ts_code").str.split(".").list.get(1)
    return (
        pl.when((exchange == "SH") & (pl.col("market") == "主板"))
        .then(pl.lit("上证主板"))
        .when((exchange == "SH") & (pl.col("market") == "科创板"))
        .then(pl.lit("科创板"))
        .when((exchange == "SZ") & (pl.col("market") == "主板"))
        .then(pl.lit("深市主板"))
        .when((exchange == "SZ") & (pl.col("market") == "创业板"))
        .then(pl.lit("创业板"))
        .when((exchange == "BJ") | (pl.col("market") == "北交所"))
        .then(pl.lit("北交所"))
        .otherwise(pl.lit("未知"))
    )


def _merge_with_existing_current(
    config: StorageConfig,
    dataset_id: str,
    frame: pl.DataFrame,
    schema: list[dict[str, object]],
) -> pl.DataFrame:
    output_path = config.current_curated_root / dataset_id / "part-000.parquet"
    if not output_path.exists():
        return frame
    existing = pl.read_parquet(output_path)
    combined = pl.concat([existing, frame], how="diagonal_relaxed")
    primary_keys = _primary_key_columns(schema)
    if not primary_keys:
        return combined
    aggregations = [
        pl.col(column).drop_nulls().last().alias(column)
        for column in combined.columns
        if column not in primary_keys
    ]
    return combined.group_by(primary_keys, maintain_order=True).agg(aggregations)


def _filter_to_existing_current_keys(
    config: StorageConfig,
    dataset_id: str,
    frame: pl.DataFrame,
    key_columns: list[str],
) -> pl.DataFrame:
    output_path = config.current_curated_root / dataset_id / "part-000.parquet"
    if frame.is_empty() or not output_path.exists():
        return frame.head(0)
    join_keys = [f"__join_{column}" for column in key_columns]
    left = frame.with_columns(
        [pl.col(column).cast(pl.Utf8).alias(join_key) for column, join_key in zip(key_columns, join_keys, strict=True)]
    )
    existing_keys = (
        pl.read_parquet(output_path)
        .select([pl.col(column).cast(pl.Utf8).alias(join_key) for column, join_key in zip(key_columns, join_keys, strict=True)])
        .unique()
    )
    return left.join(existing_keys, on=join_keys, how="inner").drop(join_keys)


def _current_overlap_summary(
    config: StorageConfig,
    dataset_id: str,
    frame: pl.DataFrame,
    primary_keys: list[str],
) -> dict[str, object]:
    if not primary_keys:
        return {"primary_keys": [], "incoming_rows": frame.height, "incoming_duplicate_keys": 0, "overlap_keys": 0}
    incoming_duplicate_keys = _duplicate_key_count(frame, primary_keys)
    output_path = config.current_curated_root / dataset_id / "part-000.parquet"
    if not output_path.exists():
        return {
            "primary_keys": primary_keys,
            "incoming_rows": frame.height,
            "incoming_duplicate_keys": incoming_duplicate_keys,
            "overlap_keys": 0,
        }
    existing = pl.read_parquet(output_path)
    incoming_keys = frame.select([pl.col(column).cast(pl.Utf8).alias(column) for column in primary_keys]).unique()
    existing_keys = existing.select([pl.col(column).cast(pl.Utf8).alias(column) for column in primary_keys]).unique()
    overlap_keys = (
        incoming_keys
        .join(existing_keys, on=primary_keys, how="inner")
        .height
    )
    return {
        "primary_keys": primary_keys,
        "incoming_rows": frame.height,
        "incoming_duplicate_keys": incoming_duplicate_keys,
        "overlap_keys": overlap_keys,
    }


def _duplicate_key_count(frame: pl.DataFrame, primary_keys: list[str]) -> int:
    if not primary_keys:
        return 0
    return frame.group_by(primary_keys).len().filter(pl.col("len") > 1).height


def _primary_key_columns(schema: list[dict[str, object]]) -> list[str]:
    return [str(field["name"]) for field in schema if bool(field["primary_key"])]


def _load_current_source_batch_ids(sqlite_path: Path, dataset_id: str) -> list[str]:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT source_batch_ids
            FROM curated_versions
            WHERE dataset_id = ? AND version_type = 'current' AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
    if row is None or not row[0]:
        return []
    try:
        values = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _merge_source_batch_ids(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    for batch_id in existing + incoming:
        if batch_id not in merged:
            merged.append(batch_id)
    return merged


def _curated_notes(
    source_batch_id: str,
    raw_dataset: str,
    overlap_summary: dict[str, object],
) -> str:
    return json.dumps(
        {
            "last_promoted_batch_id": source_batch_id,
            "last_promoted_raw_dataset": raw_dataset,
            "last_promote_overlap": overlap_summary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _curated_bulk_notes(
    run_id: str,
    raw_dataset: str,
    source_batch_ids: list[str],
    overlap_summary: dict[str, object],
) -> str:
    return json.dumps(
        {
            "last_promoted_run_id": run_id,
            "last_promoted_raw_dataset": raw_dataset,
            "last_promoted_batch_count": len(source_batch_ids),
            "last_promoted_first_batch_id": source_batch_ids[0] if source_batch_ids else None,
            "last_promoted_last_batch_id": source_batch_ids[-1] if source_batch_ids else None,
            "last_promote_overlap": overlap_summary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


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
