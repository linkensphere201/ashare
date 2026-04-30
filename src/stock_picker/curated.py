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
    "moneyflow_dc": "capital_flow_or_chip",
    "cyq_perf": "capital_flow_or_chip",
}


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

    schema = _load_registered_schema(config, curated_dataset)
    raw_frame = pl.read_csv(raw_path)
    mapped = _map_tushare_raw_to_curated(raw_frame, str(batch["dataset_name"]))
    now = datetime.now(UTC).isoformat(timespec="seconds")
    data_version = str(batch["business_date"] or now[:10])
    mapped = _enrich_frame(mapped, schema, source, str(batch["batch_id"]), data_version, now)
    mapped = _fill_missing_nullable_columns(mapped, schema)
    mapped = _merge_with_existing_current(config, curated_dataset, mapped, schema)
    validation_error = _validate_required_columns(mapped, schema)
    if validation_error:
        return CuratedImportResult(False, validation_error)
    return _write_curated_current(
        config=config,
        dataset_id=curated_dataset,
        frame=mapped,
        schema=schema,
        as_of_date=data_version,
        source_batch_ids=[str(batch["batch_id"])],
        created_at=now,
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


def _write_curated_current(
    config: StorageConfig,
    dataset_id: str,
    frame: pl.DataFrame,
    schema: list[dict[str, object]],
    as_of_date: str | None,
    source_batch_ids: list[str],
    created_at: str,
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
                created_at,
                json.dumps(source_batch_ids),
                row_count,
                checksum,
            ),
        )

    return CuratedImportResult(
        True,
        f"promoted {row_count} rows into curated current: {dataset_id}",
        output_path,
        row_count,
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
        return frame.with_columns(
            [
                pl.col("ts_code").alias("symbol"),
                _parse_yyyymmdd("trade_date").alias("trade_date"),
                pl.col("winner_rate").alias("close_profit_ratio"),
                pl.lit("tushare_cyq_perf").alias("data_method"),
            ]
        ).select(["symbol", "trade_date", "close_profit_ratio", "data_method"])
    raise ValueError(f"unsupported raw dataset mapping: {dataset}")


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
    primary_keys = [str(field["name"]) for field in schema if bool(field["primary_key"])]
    if not primary_keys:
        return combined
    aggregations = [
        pl.col(column).drop_nulls().last().alias(column)
        for column in combined.columns
        if column not in primary_keys
    ]
    return combined.group_by(primary_keys, maintain_order=True).agg(aggregations)


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
