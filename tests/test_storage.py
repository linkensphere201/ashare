from pathlib import Path
from datetime import date, timedelta
import json
import sqlite3

import polars as pl

from stock_picker.curated import import_curated_csv, inspect_curated, promote_raw_batch, promote_raw_run
from stock_picker.display import inspect_run, list_runs, preview_curated
from stock_picker.factor_research import research_candidate_001
import stock_picker.provider as provider_module
from stock_picker.provider import fetch_cyq_perf_batch, fetch_provider_raw, probe_provider_api, run_cyq_perf_batches, run_market_daily, write_raw_batch
from stock_picker.quality import check_curated_quality
from stock_picker.reports import show_report
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage
from stock_picker.strategy import backtest_candidate_001, rank_candidate_001


def test_storage_init_and_validate(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)

    init_result = init_storage(config_path)
    assert init_result.ok

    validate_result = validate_storage(config_path)
    assert validate_result.ok

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "data_batches",
        "provider_runs",
        "provider_run_tasks",
        "datasets",
        "schema_versions",
        "schema_fields",
        "curated_versions",
        "snapshot_manifests",
    }.issubset(table_names)


def test_register_schemas_is_idempotent(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(tmp_path)

    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    assert register_schemas(config_path).ok

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        dataset_count = connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        schema_version_count = connection.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0]
        schema_field_count = connection.execute("SELECT COUNT(*) FROM schema_fields").fetchone()[0]
        schema_version = connection.execute(
            "SELECT dataset_id, schema_file_path, LENGTH(schema_hash) FROM schema_versions"
        ).fetchone()

    assert dataset_count == 1
    assert schema_version_count == 1
    assert schema_field_count == 2
    assert schema_version == ("daily_prices", "schemas/curated/daily_prices.yaml", 64)


def test_import_curated_csv_writes_parquet_and_metadata(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(
        tmp_path,
        extra_fields=[
            ("adj_factor", "double", True),
            ("source", "string", False),
            ("source_batch_id", "string", False),
            ("data_version", "string", False),
            ("created_at", "datetime", False),
            ("trade_year", "integer", False),
            ("trade_month", "integer", False),
        ],
    )
    input_dir = tmp_path / "data" / "manual_input"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "daily_prices.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,trade_date",
                "600519.SH,2026-04-28",
                "000001.SZ,2026-04-28",
            ]
        ),
        encoding="utf-8",
    )

    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    result = import_curated_csv(
        config_path=config_path,
        dataset_id="daily_prices",
        input_path=input_path,
        as_of_date="2026-04-28",
    )

    assert result.ok
    assert result.row_count == 2
    assert result.output_path is not None
    assert result.output_path.exists()

    frame = pl.read_parquet(result.output_path)
    assert frame.select("trade_year").to_series().to_list() == [2026, 2026]
    assert frame.select("trade_month").to_series().to_list() == [4, 4]
    assert frame.select("adj_factor").to_series().null_count() == 2
    assert "source_batch_id" in frame.columns

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        batch_count = connection.execute("SELECT COUNT(*) FROM data_batches").fetchone()[0]
        curated_version = connection.execute(
            """
            SELECT dataset_id, version_type, as_of_date, row_count, status
            FROM curated_versions
            """
        ).fetchone()

    assert batch_count == 1
    assert curated_version == ("daily_prices", "current", "2026-04-28", 2, "active")

    inspection = inspect_curated(config_path, "daily_prices")
    assert inspection.ok
    assert "dataset: daily_prices" in inspection.message
    assert "metadata_row_count: 2" in inspection.message
    assert "actual_row_count: 2" in inspection.message

    quality = check_curated_quality(config_path, ["daily_prices"])
    assert quality.ok
    assert quality.message == "quality check succeeded: 1 datasets checked"

    snapshot = create_snapshot(config_path, "2026-04-28")
    assert snapshot.ok
    assert snapshot.snapshot_id == "snapshot_20260428_001"
    next_snapshot = create_snapshot(config_path, "2026-04-28")
    assert next_snapshot.snapshot_id == "snapshot_20260428_002"
    inspection = inspect_snapshot(config_path, "snapshot_20260428_001")
    assert inspection.ok
    assert "dataset_count: 1" in inspection.message
    assert "- daily_prices: daily_prices_current_20260428 rows=2" in inspection.message


def test_check_quality_fails_on_duplicate_primary_key(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(
        tmp_path,
        extra_fields=[
            ("source", "string", False),
            ("source_batch_id", "string", False),
            ("data_version", "string", False),
            ("created_at", "datetime", False),
            ("trade_year", "integer", False),
            ("trade_month", "integer", False),
        ],
    )
    input_dir = tmp_path / "data" / "manual_input"
    input_dir.mkdir(parents=True)
    input_path = input_dir / "daily_prices.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,trade_date",
                "600519.SH,2026-04-28",
                "600519.SH,2026-04-28",
            ]
        ),
        encoding="utf-8",
    )

    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    assert import_curated_csv(
        config_path=config_path,
        dataset_id="daily_prices",
        input_path=input_path,
        as_of_date="2026-04-28",
    ).ok

    quality = check_curated_quality(config_path, ["daily_prices"])
    assert not quality.ok
    assert "primary key duplicates found" in quality.message


def test_write_raw_batch_records_metadata(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    frame = pl.DataFrame(
        {
            "ts_code": ["600519.SH", "000001.SZ"],
            "trade_date": ["20260428", "20260428"],
            "close": [1690.0, 10.3],
        }
    )

    result = write_raw_batch(
        config_path=config_path,
        source="tushare",
        dataset="daily_prices",
        frame=frame,
        as_of_date="2026-04-28",
    )

    assert result.ok
    assert result.batch_id == "tushare_daily_prices_20260428_001"
    assert result.raw_path is not None
    assert result.raw_path.exists()

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT source, dataset_name, business_date, raw_path, format, row_count, status
            FROM data_batches
            """
        ).fetchone()

    assert row[0:3] == ("tushare", "daily_prices", "2026-04-28")
    assert row[4:] == ("csv", 2, "success")
    assert str(result.raw_path) == row[3]


def test_fetch_provider_requires_tushare_token(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    result = fetch_provider_raw(
        config_path=config_path,
        source="tushare",
        dataset="daily_prices",
        start_date="2026-04-28",
        end_date="2026-04-28",
    )

    assert not result.ok
    assert result.message == "missing required environment variable: TUSHARE_TOKEN"


def test_preview_curated_daily_prices_shows_default_stock_fields(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    data_root = tmp_path / "data" / "curated" / "current"
    security_path = data_root / "security_master" / "part-000.parquet"
    daily_path = data_root / "daily_prices" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "name": ["Kweichow Moutai"],
        }
    ).write_parquet(security_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [date(2026, 4, 28)],
            "pct_change": [0.72],
            "close": [1690.0],
            "amount": [3549000.0],
        }
    ).write_parquet(daily_path)
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('security_master_current_20260428', 'security_master', 'security_master_schema_v001', 'current', NULL, ?, '2026-04-28', '2026-04-28T16:00:00+00:00', '[]', 1, 'test', 'active')
            """,
            (str(security_path),),
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
            VALUES ('daily_prices_current_20260428', 'daily_prices', 'daily_prices_schema_v001', 'current', NULL, ?, '2026-04-28', '2026-04-28T16:00:00+00:00', '[]', 1, 'test', 'active')
            """,
            (str(daily_path),),
        )

    result = preview_curated(config_path, "daily_prices", symbol="600519.SH", limit=5)

    assert result.ok
    assert "curated_preview: daily_prices" in result.message
    assert "| symbol    | name            | trade_date | pct_change | close  |" in result.message
    assert "| 600519.SH | Kweichow Moutai | 2026-04-28 | 0.72       | 1690.0 |" in result.message


def test_list_and_inspect_runs_show_lineage(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    raw = pl.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20260428"], "close": [1690.0]})
    batch = write_raw_batch(config_path, "tushare", "daily_prices", raw, "2026-04-28")
    assert batch.ok
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('daily_prices_current_20260428', 'daily_prices', 'daily_prices_schema_v001', 'current', NULL, 'data/curated/current/daily_prices/part-000.parquet', '2026-04-28', '2026-04-28T16:00:00+00:00', ?, 1, 'curated-checksum', 'active')
            """,
            (json.dumps([batch.batch_id]),),
        )
        manifest = {
            "curated_versions": {
                "daily_prices": {
                    "curated_version_id": "daily_prices_current_20260428",
                    "row_count": 1,
                }
            }
        }
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
            VALUES ('snapshot_20260428_001', '2026-04-28', '2026-04-28T16:00:00+00:00', 'daily', 'test', ?, NULL)
            """,
            (json.dumps(manifest),),
        )

    runs = list_runs(config_path, limit=10)
    detail = inspect_run(config_path, str(batch.batch_id))

    assert runs.ok
    assert "total_import_runs: 1" in runs.message
    assert str(batch.batch_id) in runs.message
    assert detail.ok
    assert f"batch_id: {batch.batch_id}" in detail.message
    assert "linked_curated_versions:" in detail.message
    assert "| daily_prices_current_20260428 | daily_prices | daily_prices_schema_v001 | 2026-04-28 | 1         | active | curated-checksum |" in detail.message
    assert "linked_snapshots:" in detail.message
    assert "| snapshot_20260428_001 | 2026-04-28 | 2026-04-28T16:00:00+00:00 | daily     | test           |" in detail.message


def test_promote_tushare_daily_prices_raw(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(
        tmp_path,
        extra_fields=[
            ("asset_type", "string", False),
            ("open", "double", True),
            ("high", "double", True),
            ("low", "double", True),
            ("close", "double", True),
            ("pre_close", "double", True),
            ("volume", "double", True),
            ("amount", "double", True),
            ("pct_change", "double", True),
            ("source", "string", False),
            ("source_batch_id", "string", False),
            ("data_version", "string", False),
            ("created_at", "datetime", False),
            ("trade_year", "integer", False),
            ("trade_month", "integer", False),
        ],
    )
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    raw = pl.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20260428"],
            "open": [1680.0],
            "high": [1698.0],
            "low": [1672.0],
            "close": [1690.0],
            "pre_close": [1678.0],
            "vol": [2100000.0],
            "amount": [3549000.0],
            "pct_chg": [0.72],
        }
    )
    write_raw_batch(config_path, "tushare", "daily_prices", raw, "2026-04-28")

    result = promote_raw_batch(config_path, "tushare", "daily_prices", "2026-04-28")

    assert result.ok
    assert result.output_path is not None
    frame = pl.read_parquet(result.output_path)
    assert frame.select("symbol").to_series().to_list() == ["600519.SH"]
    assert frame.select("volume").to_series().to_list() == [2100000.0]
    assert frame.select("trade_year").to_series().to_list() == [2026]


def test_promote_raw_records_overlap_notes_and_quality_warning(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(
        tmp_path,
        extra_fields=[
            ("asset_type", "string", False),
            ("open", "double", True),
            ("high", "double", True),
            ("low", "double", True),
            ("close", "double", True),
            ("pre_close", "double", True),
            ("volume", "double", True),
            ("amount", "double", True),
            ("pct_change", "double", True),
            ("source", "string", False),
            ("source_batch_id", "string", False),
            ("data_version", "string", False),
            ("created_at", "datetime", False),
            ("trade_year", "integer", False),
            ("trade_month", "integer", False),
        ],
    )
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    first = pl.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20260428"],
            "open": [1680.0],
            "high": [1698.0],
            "low": [1672.0],
            "close": [1690.0],
            "pre_close": [1678.0],
            "vol": [2100000.0],
            "amount": [3549000.0],
            "pct_chg": [0.72],
        }
    )
    second = first.with_columns(pl.lit(1700.0).alias("close"))
    write_raw_batch(config_path, "tushare", "daily_prices", first, "2026-04-28")
    write_raw_batch(config_path, "tushare", "daily_prices", second, "2026-04-28")

    first_promote = promote_raw_batch(
        config_path,
        "tushare",
        "daily_prices",
        batch_id="tushare_daily_prices_20260428_001",
    )
    second_promote = promote_raw_batch(
        config_path,
        "tushare",
        "daily_prices",
        batch_id="tushare_daily_prices_20260428_002",
    )

    assert first_promote.ok
    assert second_promote.ok
    assert second_promote.output_path is not None
    frame = pl.read_parquet(second_promote.output_path)
    assert frame.height == 1
    assert frame.select("close").to_series().to_list() == [1700.0]

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        source_batch_ids, notes = connection.execute(
            """
            SELECT source_batch_ids, notes
            FROM curated_versions
            WHERE dataset_id = 'daily_prices' AND version_type = 'current'
            """
        ).fetchone()

    assert json.loads(source_batch_ids) == [
        "tushare_daily_prices_20260428_001",
        "tushare_daily_prices_20260428_002",
    ]
    parsed_notes = json.loads(notes)
    assert parsed_notes["last_promoted_batch_id"] == "tushare_daily_prices_20260428_002"
    assert parsed_notes["last_promote_overlap"]["overlap_keys"] == 1

    quality = check_curated_quality(config_path, ["daily_prices"])
    assert quality.ok
    assert "quality warnings:" in quality.message
    assert "overlapped 1 existing primary keys" in quality.message


def test_promote_raw_run_merges_all_successful_task_batches(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_daily_prices_schema(
        tmp_path,
        extra_fields=[
            ("asset_type", "string", False),
            ("open", "double", True),
            ("high", "double", True),
            ("low", "double", True),
            ("close", "double", True),
            ("pre_close", "double", True),
            ("volume", "double", True),
            ("amount", "double", True),
            ("pct_change", "double", True),
            ("trade_year", "integer", False),
            ("trade_month", "integer", False),
            ("source", "string", False),
            ("source_batch_id", "string", False),
            ("data_version", "string", False),
            ("created_at", "datetime", False),
        ],
    )
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok

    first = write_raw_batch(
        config_path,
        "tushare",
        "daily_prices",
        pl.DataFrame(
            {
                "ts_code": ["600519.SH"],
                "trade_date": ["20260427"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.0],
                "close": [10.5],
                "pre_close": [10.0],
                "vol": [100.0],
                "amount": [1000.0],
                "pct_chg": [5.0],
            }
        ),
        "2026-04-28",
    )
    second = write_raw_batch(
        config_path,
        "tushare",
        "daily_prices",
        pl.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260428"],
                "open": [20.0],
                "high": [21.0],
                "low": [19.0],
                "close": [20.5],
                "pre_close": [20.0],
                "vol": [200.0],
                "amount": [2000.0],
                "pct_chg": [2.5],
            }
        ),
        "2026-04-28",
    )
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO provider_runs (
              run_id, source, dataset_name, start_date, end_date, as_of_date, status,
              total_symbols, next_offset, batch_size, requested_symbols, symbols_with_rows,
              failed_symbols, row_count, raw_batch_ids, failure_json, created_at, updated_at, notes
            )
            VALUES ('bulk_promote_test', 'tushare', 'market_daily', '2026-04-27', '2026-04-28', '2026-04-28', 'completed',
              2, 2, 0, 2, 2, 0, 2, '[]', '[]', '2026-04-28T00:00:00+00:00', '2026-04-28T00:00:00+00:00', NULL)
            """
        )
        connection.executemany(
            """
            INSERT INTO provider_run_tasks (
              task_id, run_id, source, dataset_name, task_type, trade_date,
              symbol_start_offset, symbol_end_offset, start_date, end_date, payload_json,
              status, attempts, raw_batch_id, row_count, error_reason, error_message,
              provider_message, retryable, created_at, updated_at, started_at, finished_at, notes
            )
            VALUES (?, 'bulk_promote_test', 'tushare', 'daily_prices', 'market_daily_date', ?,
              NULL, NULL, NULL, NULL, '{}', 'success', 1, ?, 1, NULL, NULL, NULL, NULL,
              '2026-04-28T00:00:00+00:00', '2026-04-28T00:00:00+00:00', NULL, NULL, NULL)
            """,
            [
                ("bulk_promote_test:daily_prices:20260427", "2026-04-27", first.batch_id),
                ("bulk_promote_test:daily_prices:20260428", "2026-04-28", second.batch_id),
            ],
        )

    result = promote_raw_run(config_path, "bulk_promote_test", "daily_prices")

    assert result.ok
    assert result.row_count == 2
    assert result.output_path is not None
    frame = pl.read_parquet(result.output_path).sort(["trade_date", "symbol"])
    assert frame.select("symbol", "trade_date", "close", "source_batch_id").to_dicts() == [
        {"symbol": "600519.SH", "trade_date": date(2026, 4, 27), "close": 10.5, "source_batch_id": first.batch_id},
        {"symbol": "000001.SZ", "trade_date": date(2026, 4, 28), "close": 20.5, "source_batch_id": second.batch_id},
    ]


def test_promote_tushare_security_master_adds_market_fields(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_security_master_schema(tmp_path)
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    raw = pl.DataFrame(
        {
            "ts_code": ["600519.SH", "688526.SH", "300750.SZ", "920964.BJ"],
            "symbol": ["600519", "688526", "300750", "920964"],
            "name": ["Kweichow Moutai", "Bioleader", "CATL", "Runong"],
            "area": ["Guizhou", "Shanghai", "Fujian", "Beijing"],
            "industry": ["Liquor", "Biotech", "Battery", "Agriculture"],
            "market": ["主板", "科创板", "创业板", "北交所"],
            "list_date": ["20010827", "20200922", "20180611", "20200727"],
            "delist_date": ["", "", "", ""],
        }
    )
    write_raw_batch(config_path, "tushare", "security_master", raw, "2026-04-28")

    result = promote_raw_batch(config_path, "tushare", "security_master", "2026-04-28")

    assert result.ok
    assert result.output_path is not None
    frame = pl.read_parquet(result.output_path).sort("symbol")
    rows = frame.select(["symbol", "exchange", "market", "market_segment", "market_segment_name", "area", "industry"]).to_dicts()
    assert rows == [
        {
            "symbol": "300750.SZ",
            "exchange": "SZ",
            "market": "创业板",
            "market_segment": "chinext",
            "market_segment_name": "创业板",
            "area": "Fujian",
            "industry": "Battery",
        },
        {
            "symbol": "600519.SH",
            "exchange": "SH",
            "market": "主板",
            "market_segment": "sh_main",
            "market_segment_name": "上证主板",
            "area": "Guizhou",
            "industry": "Liquor",
        },
        {
            "symbol": "688526.SH",
            "exchange": "SH",
            "market": "科创板",
            "market_segment": "star",
            "market_segment_name": "科创板",
            "area": "Shanghai",
            "industry": "Biotech",
        },
        {
            "symbol": "920964.BJ",
            "exchange": "BJ",
            "market": "北交所",
            "market_segment": "bj",
            "market_segment_name": "北交所",
            "area": "Beijing",
            "industry": "Agriculture",
        },
    ]


def test_promote_tushare_moneyflow_and_cyq_merge(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_capital_flow_schema(tmp_path)
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    moneyflow = pl.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20260428"],
            "net_amount": [1200.5],
            "net_amount_rate": [3.2],
        }
    )
    cyq = pl.DataFrame(
        {
            "ts_code": ["600519.SH"],
            "trade_date": ["20260428"],
            "winner_rate": [85.0],
        }
    )
    write_raw_batch(config_path, "tushare", "moneyflow_dc", moneyflow, "2026-04-28")
    write_raw_batch(config_path, "tushare", "cyq_perf", cyq, "2026-04-28")

    moneyflow_result = promote_raw_batch(config_path, "tushare", "moneyflow_dc", "2026-04-28")
    cyq_result = promote_raw_batch(config_path, "tushare", "cyq_perf", "2026-04-28")

    assert moneyflow_result.ok
    assert cyq_result.ok
    assert cyq_result.output_path is not None
    frame = pl.read_parquet(cyq_result.output_path)
    row = frame.row(0, named=True)
    assert row["symbol"] == "600519.SH"
    assert row["main_net_inflow"] == 1200.5
    assert row["main_net_inflow_rate"] == 3.2
    assert row["close_profit_ratio"] == 85.0


def test_rank_candidate_001_outputs_matching_candidate(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    data_root = tmp_path / "data" / "curated" / "current"
    security_path = data_root / "security_master" / "part-000.parquet"
    daily_path = data_root / "daily_prices" / "part-000.parquet"
    capital_path = data_root / "capital_flow_or_chip" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    capital_path.parent.mkdir(parents=True)

    latest = date(2026, 4, 28)
    dates = [latest - timedelta(days=39 - index) for index in range(40)]
    closes = [10.0] * 39 + [20.0]
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "name": ["Kweichow Moutai"],
            "status": ["active"],
        }
    ).write_parquet(security_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * 40,
            "trade_date": dates,
            "close": closes,
            "amount": [1000.0] * 40,
        }
    ).write_parquet(daily_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [latest],
            "main_net_inflow_rate": [3.2],
            "close_profit_ratio": [85.0],
        }
    ).write_parquet(capital_path)
    manifest = {
        "curated_versions": {
            "security_master": {"path": str(security_path)},
            "daily_prices": {"path": str(daily_path)},
            "capital_flow_or_chip": {"path": str(capital_path)},
        }
    }
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('snapshot_test', '2026-04-28', '2026-04-28T16:00:00+00:00', 'daily', 'test', ?, NULL)
            """,
            (json.dumps(manifest),),
        )

    result = rank_candidate_001(config_path, "snapshot_test", top=10)

    assert result.ok
    assert "600519.SH" in result.message
    assert "net_amount_rate>0" in result.message


def test_rank_candidate_001_excludes_st_names(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    data_root = tmp_path / "data" / "curated" / "current"
    security_path = data_root / "security_master" / "part-000.parquet"
    daily_path = data_root / "daily_prices" / "part-000.parquet"
    capital_path = data_root / "capital_flow_or_chip" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    capital_path.parent.mkdir(parents=True)

    latest = date(2026, 4, 28)
    dates = [latest - timedelta(days=39 - index) for index in range(40)]
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ", "000002.SZ"],
            "name": ["Kweichow Moutai", "*ST Test", "ST Test"],
            "status": ["active", "active", "active"],
        }
    ).write_parquet(security_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * 40 + ["000001.SZ"] * 40 + ["000002.SZ"] * 40,
            "trade_date": dates * 3,
            "close": ([10.0] * 39 + [20.0]) * 3,
            "amount": [1000.0] * 120,
        }
    ).write_parquet(daily_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ", "000002.SZ"],
            "trade_date": [latest, latest, latest],
            "main_net_inflow_rate": [3.2, 99.0, 98.0],
            "close_profit_ratio": [85.0, 99.0, 99.0],
        }
    ).write_parquet(capital_path)
    manifest = {
        "curated_versions": {
            "security_master": {"path": str(security_path)},
            "daily_prices": {"path": str(daily_path)},
            "capital_flow_or_chip": {"path": str(capital_path)},
        }
    }
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('snapshot_st_filter', '2026-04-28', '2026-04-28T16:00:00+00:00', 'daily', 'test', ?, NULL)
            """,
            (json.dumps(manifest),),
        )

    result = rank_candidate_001(config_path, "snapshot_st_filter", top=10)

    assert result.ok
    assert "600519.SH" in result.message
    assert "000001.SZ" not in result.message
    assert "000002.SZ" not in result.message


def test_backtest_candidate_001_outputs_forward_return(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    data_root = tmp_path / "data" / "curated" / "current"
    security_path = data_root / "security_master" / "part-000.parquet"
    daily_path = data_root / "daily_prices" / "part-000.parquet"
    capital_path = data_root / "capital_flow_or_chip" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    capital_path.parent.mkdir(parents=True)

    latest = date(2026, 4, 28)
    dates = [latest - timedelta(days=45 - index) for index in range(46)]
    signal_date = dates[39]
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "name": ["Kweichow Moutai"],
            "status": ["active"],
        }
    ).write_parquet(security_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * 46,
            "trade_date": dates,
            "close": [10.0] * 39 + [20.0] + [22.0] * 5 + [24.2],
            "amount": [1000.0] * 46,
        }
    ).write_parquet(daily_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"],
            "trade_date": [signal_date],
            "main_net_inflow_rate": [3.2],
            "close_profit_ratio": [85.0],
        }
    ).write_parquet(capital_path)
    manifest = {
        "curated_versions": {
            "security_master": {"path": str(security_path)},
            "daily_prices": {"path": str(daily_path)},
            "capital_flow_or_chip": {"path": str(capital_path)},
        }
    }
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('snapshot_backtest', '2026-04-28', '2026-04-28T16:00:00+00:00', 'daily', 'test', ?, NULL)
            """,
            (json.dumps(manifest),),
        )

    result = backtest_candidate_001(config_path, "snapshot_backtest", holding_days=5, top=10)

    assert result.ok
    assert "Strategy Candidate 001 v2 backtest" in result.message
    assert "signal timing: T close signal, T+1 entry" in result.message
    assert "trade_count: 1" in result.message
    assert "avg_forward_return: 0.100000" in result.message


def test_factor_research_candidate_001_writes_report_artifacts(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    data_root = tmp_path / "data" / "curated" / "current"
    security_path = data_root / "security_master" / "part-000.parquet"
    daily_path = data_root / "daily_prices" / "part-000.parquet"
    capital_path = data_root / "capital_flow_or_chip" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    capital_path.parent.mkdir(parents=True)

    latest = date(2026, 4, 28)
    dates = [latest - timedelta(days=45 - index) for index in range(46)]
    signal_date = dates[39]
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "name": ["Kweichow Moutai", "*ST Test"],
            "status": ["active", "active"],
        }
    ).write_parquet(security_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * 46 + ["000001.SZ"] * 46,
            "trade_date": dates * 2,
            "close": ([10.0] * 39 + [20.0] + [22.0] * 5 + [24.2]) * 2,
            "amount": [1000.0] * 92,
        }
    ).write_parquet(daily_path)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ"],
            "trade_date": [signal_date, signal_date],
            "main_net_inflow_rate": [3.2, 99.0],
            "close_profit_ratio": [85.0, 99.0],
        }
    ).write_parquet(capital_path)
    manifest = {
        "curated_versions": {
            "security_master": {"path": str(security_path)},
            "daily_prices": {"path": str(daily_path)},
            "capital_flow_or_chip": {"path": str(capital_path)},
        }
    }
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
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
            VALUES ('snapshot_factor_research', '2026-04-28', '2026-04-28T16:00:00+00:00', 'daily', 'test', ?, NULL)
            """,
            (json.dumps(manifest),),
        )

    result = research_candidate_001(
        config_path,
        "snapshot_factor_research",
        holding_days=5,
        top=10,
        report_id="candidate_001_test_report",
    )

    assert result.ok
    assert result.report_dir is not None
    summary_path = result.report_dir / "summary.md"
    factor_summary_path = result.report_dir / "factor_summary.json"
    factor_results_path = result.report_dir / "factor_results.csv"
    ranking_path = result.report_dir / "ranking.csv"
    backtest_path = result.report_dir / "backtest_trades.csv"
    metrics_path = result.report_dir / "metrics.json"
    assert summary_path.exists()
    assert factor_summary_path.exists()
    assert factor_results_path.exists()
    assert ranking_path.exists()
    assert backtest_path.exists()
    assert metrics_path.exists()
    summary = summary_path.read_text(encoding="utf-8")
    assert "Factor Summary" in summary
    assert "Explanation" in summary
    assert "Ranking Result" in summary
    assert "Backtest Result" in summary
    assert "ST` or `*ST` are excluded" in summary
    summary_json = json.loads(factor_summary_path.read_text(encoding="utf-8"))
    assert "strategy_description" in summary_json
    assert "factor_value_description" in summary_json
    assert "ranking_description" in summary_json
    assert "backtest_description" in summary_json
    assert summary_json["entry_price_rule"] == "T+1 open if available, otherwise T+1 close"
    factor_results = pl.read_csv(factor_results_path)
    assert "factor_pass" in factor_results.columns
    assert "factor_value" in factor_results.columns
    ranking = pl.read_csv(ranking_path)
    assert "symbol" in ranking.columns
    assert "factor_value" in ranking.columns
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["trade_count"] == 1
    shown = show_report(config_path, "candidate_001_test_report", limit=5)
    assert shown.ok
    assert "factor_description:" in shown.message
    assert "factor_results:" not in shown.message
    assert "ranking_results:" in shown.message
    assert "backtest_result:" in shown.message
    assert "排序逻辑" in shown.message
    assert "factor_rank" in shown.message
    assert "symbol" in shown.message
    assert "factor_value" in shown.message
    assert "factor_pass" not in shown.message
    assert "+-" in shown.message


def test_fetch_cyq_perf_requires_ts_code(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok

    result = fetch_provider_raw(
        config_path=config_path,
        source="tushare",
        dataset="cyq_perf",
        start_date="2026-04-28",
        end_date="2026-04-28",
    )

    assert not result.ok
    assert result.message == "cyq_perf fetch requires --ts-code"


def test_fetch_cyq_perf_batch_requires_tushare_token(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    result = fetch_cyq_perf_batch(config_path=config_path)

    assert not result.ok
    assert result.message == "missing required environment variable: TUSHARE_TOKEN"


def test_fetch_cyq_perf_batch_writes_combined_raw_batch(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ", "300750.SZ"],
            "status": ["active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [85.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = fetch_cyq_perf_batch(
        config_path=config_path,
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        limit=2,
    )

    assert result.ok
    assert result.batch_id == "tushare_cyq_perf_20260428_001"
    assert result.raw_path is not None
    frame = pl.read_csv(result.raw_path)
    assert frame.height == 2
    assert sorted(frame.select("ts_code").to_series().to_list()) == ["000001.SZ", "300750.SZ"]
    assert "symbols_requested=2" in result.message
    assert "symbols_with_rows=2" in result.message


def test_run_cyq_perf_batches_resumes_from_saved_offset(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000004.SZ", "000006.SZ"],
            "status": ["active", "active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    first = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_test_run",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=1,
    )
    second = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_test_run",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=1,
    )

    assert first.ok
    assert "tasks_success=1/2" in first.message
    assert second.ok
    assert "status=completed" in second.message
    assert "tasks_success=2/2" in second.message

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        run = connection.execute(
            """
            SELECT status, next_offset, requested_symbols, symbols_with_rows, row_count, raw_batch_ids
            FROM provider_runs
            WHERE run_id = 'cyq_test_run'
            """
        ).fetchone()
        batch_rows = connection.execute(
            "SELECT batch_id, row_count FROM data_batches WHERE dataset_name = 'cyq_perf' ORDER BY batch_id"
        ).fetchall()
        tasks = connection.execute(
            """
            SELECT dataset_name, task_type, start_date, end_date, symbol_start_offset,
                   symbol_end_offset, status, row_count, raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = 'cyq_test_run'
            ORDER BY symbol_start_offset
            """
        ).fetchall()

    assert run[0] == "completed"
    assert run[4] == 4
    assert json.loads(run[5]) == ["tushare_cyq_perf_20260428_001", "tushare_cyq_perf_20260428_002"]
    assert batch_rows == [("tushare_cyq_perf_20260428_001", 2), ("tushare_cyq_perf_20260428_002", 2)]
    assert tasks == [
        ("cyq_perf", "cyq_perf_symbol_batch", "2026-04-26", "2026-04-28", 0, 2, "success", 2, "tushare_cyq_perf_20260428_001"),
        ("cyq_perf", "cyq_perf_symbol_batch", "2026-04-26", "2026-04-28", 2, 4, "success", 2, "tushare_cyq_perf_20260428_002"),
    ]


def test_run_cyq_perf_batches_reports_progress(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000004.SZ", "000006.SZ"],
            "status": ["active", "active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    progress_messages: list[str] = []

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_progress_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=2,
        progress_every_batches=1,
        progress_callback=progress_messages.append,
    )

    assert result.ok
    assert len(progress_messages) == 2
    assert "this_invocation=1/2" in progress_messages[0]
    assert "total_success=1/2" in progress_messages[0]
    assert "last=cyq_perf/2026-04-26-2026-04-28 symbols=0-2" in progress_messages[0]
    assert "this_invocation=2/2" in progress_messages[1]
    assert "total_success=2/2" in progress_messages[1]
    assert "status=completed" in result.message
    assert "recent_raw_batches=tushare_cyq_perf_20260428_001, tushare_cyq_perf_20260428_002" in result.message


def test_run_cyq_perf_batches_retries_rate_limit_error(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "status": ["active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(provider_module.time, "sleep", lambda seconds: None)
    calls: dict[str, int] = {}

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            calls[ts_code] = calls.get(ts_code, 0) + 1
            if ts_code == "000001.SZ" and calls[ts_code] == 1:
                raise Exception("抱歉，您访问接口(cyq_perf)频率超限(200次/分钟)")
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_retry_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=1,
        retry=1,
        retry_wait_seconds=0,
    )

    assert result.ok
    assert calls == {"000001.SZ": 2, "000002.SZ": 1}
    assert "tasks_success=1/1" in result.message
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        run = connection.execute(
            """
            SELECT status, requested_symbols, symbols_with_rows, failed_symbols, row_count
            FROM provider_runs
            WHERE run_id = 'cyq_retry_test'
            """
        ).fetchone()
        task = connection.execute(
            """
            SELECT status, attempts, error_reason, row_count, raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = 'cyq_retry_test'
            """
        ).fetchone()
    assert run == ("completed", 1, 1, 0, 2)
    assert task == ("success", 2, None, 2, "tushare_cyq_perf_20260428_001")


def test_run_cyq_perf_batches_writes_not_found_placeholder_for_empty_result(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    _write_capital_flow_schema(tmp_path)
    assert init_storage(config_path).ok
    assert register_schemas(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000004.SZ"],
            "status": ["active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            if ts_code == "000002.SZ":
                return pl.DataFrame({"ts_code": [], "trade_date": [], "winner_rate": []}).to_pandas()
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_not_found_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=3,
        max_batches=1,
    )

    assert result.ok
    assert "tasks_success=1/1" in result.message
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        raw_path = connection.execute(
            "SELECT raw_path FROM data_batches WHERE batch_id = ?",
            (result.batch_id,),
        ).fetchone()[0]
    raw = pl.read_csv(raw_path).sort("ts_code")
    assert raw.select("ts_code", "trade_date", "winner_rate", "provider_status", "provider_message").to_dicts() == [
        {"ts_code": "000001.SZ", "trade_date": 20260428, "winner_rate": 80.0, "provider_status": None, "provider_message": None},
        {"ts_code": "000002.SZ", "trade_date": 20260428, "winner_rate": None, "provider_status": "not_found", "provider_message": "empty_result"},
        {"ts_code": "000004.SZ", "trade_date": 20260428, "winner_rate": 80.0, "provider_status": None, "provider_message": None},
    ]

    promote = promote_raw_batch(config_path, "tushare", "cyq_perf", "2026-04-28")
    assert promote.ok
    curated = pl.read_parquet(tmp_path / "data" / "curated" / "current" / "capital_flow_or_chip" / "part-000.parquet")
    missing = curated.filter(pl.col("symbol") == "000002.SZ").to_dicts()[0]
    assert missing["close_profit_ratio"] is None
    assert missing["data_method"] == "tushare_cyq_perf:not_found"


def test_provider_run_resume_marks_failed_run_running_before_continuing(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000004.SZ"],
            "status": ["active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    fail_first = True

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            nonlocal fail_first
            if fail_first and ts_code == "000001.SZ":
                raise Exception("provider internal error")
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    first = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_resume_status_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=1,
    )
    assert not first.ok
    fail_first = False
    second = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_resume_status_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=1,
    )

    assert second.ok
    assert "status=running" in second.message
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        run_status = connection.execute(
            "SELECT status FROM provider_runs WHERE run_id = 'cyq_resume_status_test'"
        ).fetchone()
        tasks = connection.execute(
            """
            SELECT symbol_start_offset, symbol_end_offset, status, attempts, error_reason
            FROM provider_run_tasks
            WHERE run_id = 'cyq_resume_status_test'
            ORDER BY symbol_start_offset
            """
        ).fetchall()
    assert run_status == ("running",)
    assert tasks == [(0, 2, "success", 1, None), (2, 3, "pending", 0, None)]


def test_run_cyq_perf_batches_stops_non_retryable_failure_without_partial_raw(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ", "000004.SZ"],
            "status": ["active", "active", "active"],
        }
    ).write_parquet(security_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def cyq_perf(self, ts_code, start_date=None, end_date=None):
            if ts_code == "000002.SZ":
                raise Exception("provider internal error")
            return pl.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": [end_date],
                    "winner_rate": [80.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_cyq_perf_batches(
        config_path=config_path,
        run_id="cyq_strict_failure_test",
        start_date="2026-04-26",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        batch_size=2,
        max_batches=2,
        retry=1,
        retry_wait_seconds=0,
    )

    assert not result.ok
    assert "reason=unknown" in result.message
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        batches = connection.execute(
            "SELECT batch_id FROM data_batches WHERE dataset_name = 'cyq_perf'"
        ).fetchall()
        tasks = connection.execute(
            """
            SELECT symbol_start_offset, symbol_end_offset, status, attempts, error_reason, retryable, row_count, raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = 'cyq_strict_failure_test'
            ORDER BY symbol_start_offset
            """
        ).fetchall()
        run = connection.execute(
            """
            SELECT status, requested_symbols, symbols_with_rows, failed_symbols, row_count, raw_batch_ids
            FROM provider_runs
            WHERE run_id = 'cyq_strict_failure_test'
            """
        ).fetchone()
    assert batches == []
    assert tasks == [
        (0, 2, "failed", 1, "unknown", 0, 0, None),
        (2, 3, "pending", 0, None, None, 0, None),
    ]
    assert run == ("failed", 0, 0, 1, 0, "[]")


def test_run_market_daily_resumes_by_trade_date_tasks(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_trading_calendar_current(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def daily(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
            return pl.DataFrame(
                {
                    "ts_code": ["600519.SH"],
                    "trade_date": [trade_date],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.0],
                    "close": [10.5],
                    "pre_close": [10.0],
                    "vol": [100.0],
                    "amount": [1000.0],
                    "pct_chg": [5.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    first = run_market_daily(
        config_path=config_path,
        run_id="market_daily_test",
        datasets=["daily_prices"],
        start_date="2026-04-27",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        max_tasks=1,
        requests_per_minute=0,
    )
    second = run_market_daily(
        config_path=config_path,
        run_id="market_daily_test",
        datasets=["daily_prices"],
        start_date="2026-04-27",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        max_tasks=1,
        requests_per_minute=0,
    )

    assert first.ok
    assert "tasks_success=1/2" in first.message
    assert second.ok
    assert "status=completed" in second.message
    assert "tasks_success=2/2" in second.message

    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        tasks = connection.execute(
            """
            SELECT dataset_name, trade_date, status, attempts, row_count, raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = 'market_daily_test'
            ORDER BY trade_date
            """
        ).fetchall()
        run = connection.execute(
            "SELECT status, requested_symbols, row_count, raw_batch_ids FROM provider_runs WHERE run_id = 'market_daily_test'"
        ).fetchone()

    assert [task[:5] for task in tasks] == [
        ("daily_prices", "2026-04-27", "success", 1, 1),
        ("daily_prices", "2026-04-28", "success", 1, 1),
    ]
    assert [task[5] for task in tasks] == ["tushare_daily_prices_20260428_001", "tushare_daily_prices_20260428_002"]
    assert run[:3] == ("completed", 2, 2)
    assert json.loads(run[3]) == ["tushare_daily_prices_20260428_001", "tushare_daily_prices_20260428_002"]


def test_run_market_daily_retries_rate_limit_error(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_trading_calendar_current(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setattr(provider_module.time, "sleep", lambda seconds: None)
    calls = {"daily": 0}

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def daily(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
            calls["daily"] += 1
            if calls["daily"] == 1:
                raise Exception("每分钟最多访问该接口80次")
            return pl.DataFrame(
                {
                    "ts_code": ["600519.SH"],
                    "trade_date": [trade_date],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.0],
                    "close": [10.5],
                    "pre_close": [10.0],
                    "vol": [100.0],
                    "amount": [1000.0],
                    "pct_chg": [5.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_market_daily(
        config_path=config_path,
        run_id="market_daily_retry_test",
        datasets=["daily_prices"],
        start_date="2026-04-27",
        end_date="2026-04-27",
        as_of_date="2026-04-28",
        max_tasks=1,
        requests_per_minute=0,
        retry=1,
        retry_wait_seconds=0,
    )

    assert result.ok
    assert calls["daily"] == 2
    assert "tasks_success=1/1" in result.message
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        task = connection.execute(
            "SELECT status, attempts, error_message FROM provider_run_tasks WHERE run_id = 'market_daily_retry_test'"
        ).fetchone()
    assert task == ("success", 2, None)


def test_run_market_daily_splits_moneyflow_by_symbol_batches(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_trading_calendar_current(tmp_path)
    _write_security_master_current(tmp_path, ["000001.SZ", "000002.SZ", "600519.SH"])
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    requested_codes: list[str] = []

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def moneyflow_dc(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
            requested_codes.append(ts_code)
            codes = ts_code.split(",")
            return pl.DataFrame(
                {
                    "ts_code": codes,
                    "trade_date": [trade_date] * len(codes),
                    "net_amount": [100.0] * len(codes),
                    "net_amount_rate": [1.0] * len(codes),
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_market_daily(
        config_path=config_path,
        run_id="market_daily_moneyflow_split_test",
        datasets=["moneyflow_dc"],
        start_date="2026-04-27",
        end_date="2026-04-27",
        as_of_date="2026-04-28",
        max_tasks=2,
        requests_per_minute=0,
        symbol_batch_size=2,
    )

    assert result.ok
    assert "tasks_success=2/2" in result.message
    assert requested_codes == ["000001.SZ,000002.SZ", "600519.SH"]
    sqlite_path = tmp_path / "data" / "metadata.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        tasks = connection.execute(
            """
            SELECT dataset_name, trade_date, symbol_start_offset, symbol_end_offset, status, row_count
            FROM provider_run_tasks
            WHERE run_id = 'market_daily_moneyflow_split_test'
            ORDER BY symbol_start_offset
            """
        ).fetchall()
    assert tasks == [
        ("moneyflow_dc", "2026-04-27", 0, 2, "success", 2),
        ("moneyflow_dc", "2026-04-27", 2, 3, "success", 1),
    ]


def test_run_market_daily_reports_progress(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_trading_calendar_current(tmp_path)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    progress_messages: list[str] = []

    class FakeTushare:
        @staticmethod
        def pro_api(token):
            assert token == "test-token"
            return FakePro()

    class FakePro:
        def daily(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
            return pl.DataFrame(
                {
                    "ts_code": ["600519.SH"],
                    "trade_date": [trade_date],
                    "open": [10.0],
                    "high": [11.0],
                    "low": [9.0],
                    "close": [10.5],
                    "pre_close": [10.0],
                    "vol": [100.0],
                    "amount": [1000.0],
                    "pct_chg": [5.0],
                }
            ).to_pandas()

    import sys

    monkeypatch.setitem(sys.modules, "tushare", FakeTushare)

    result = run_market_daily(
        config_path=config_path,
        run_id="market_daily_progress_test",
        datasets=["daily_prices"],
        start_date="2026-04-27",
        end_date="2026-04-28",
        as_of_date="2026-04-28",
        max_tasks=2,
        requests_per_minute=0,
        progress_every_tasks=1,
        progress_callback=progress_messages.append,
    )

    assert result.ok
    assert len(progress_messages) == 2
    assert "this_invocation=1/2" in progress_messages[0]
    assert "total_success=1/2" in progress_messages[0]
    assert "last=daily_prices/2026-04-27" in progress_messages[0]
    assert "this_invocation=2/2" in progress_messages[1]
    assert "status=completed" in result.message
    assert "tasks_this_run=2" in result.message
    assert "recent_raw_batches=tushare_daily_prices_20260428_001, tushare_daily_prices_20260428_002" in result.message


def test_probe_provider_requires_tushare_token(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    result = probe_provider_api(
        source="tushare",
        api="cyq_perf",
        ts_code="600519.SH",
        trade_date="20260428",
    )

    assert not result.ok
    assert result.message == "missing required environment variable: TUSHARE_TOKEN"


def test_probe_provider_rejects_unsupported_api() -> None:
    result = probe_provider_api(source="tushare", api="unknown_api")

    assert not result.ok
    assert result.message == "unsupported tushare api: unknown_api"


def _write_daily_prices_schema(
    tmp_path: Path,
    extra_fields: list[tuple[str, str, bool]] | None = None,
) -> None:
    schema_dir = tmp_path / "schemas" / "curated"
    schema_dir.mkdir(parents=True)
    lines = [
        "dataset_id: daily_prices",
        "dataset_name: Daily Prices",
        "layer: curated",
        "version: v001",
        "description: Daily price schema.",
        "fields:",
        "  - name: symbol",
        "    type: string",
        "    nullable: false",
        "    primary_key: true",
        "    partition_key: false",
        "    description: Canonical symbol.",
        "  - name: trade_date",
        "    type: date",
        "    nullable: false",
        "    primary_key: true",
        "    partition_key: false",
        "    description: Trading date.",
    ]
    for name, field_type, nullable in extra_fields or []:
        lines.extend(
            [
                f"  - name: {name}",
                f"    type: {field_type}",
                f"    nullable: {str(nullable).lower()}",
                "    primary_key: false",
                f"    partition_key: {str(name in {'trade_year', 'trade_month'}).lower()}",
                f"    description: {name}.",
            ]
        )
    (schema_dir / "daily_prices.yaml").write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def _write_capital_flow_schema(tmp_path: Path) -> None:
    schema_dir = tmp_path / "schemas" / "curated"
    schema_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("symbol", "string", False, True, False),
        ("trade_date", "date", False, True, False),
        ("main_force_holding_ratio", "double", True, False, False),
        ("close_profit_ratio", "double", True, False, False),
        ("main_net_inflow", "double", True, False, False),
        ("main_net_inflow_rate", "double", True, False, False),
        ("retail_net_inflow", "double", True, False, False),
        ("chip_concentration", "double", True, False, False),
        ("data_method", "string", True, False, False),
        ("trade_year", "integer", False, False, True),
        ("trade_month", "integer", False, False, True),
        ("source", "string", False, False, False),
        ("source_batch_id", "string", False, False, False),
        ("data_version", "string", False, False, False),
        ("created_at", "datetime", False, False, False),
    ]
    lines = [
        "dataset_id: capital_flow_or_chip",
        "dataset_name: Capital Flow Or Chip",
        "layer: curated",
        "version: v001",
        "description: Capital flow schema.",
        "fields:",
    ]
    for name, field_type, nullable, primary_key, partition_key in fields:
        lines.extend(
            [
                f"  - name: {name}",
                f"    type: {field_type}",
                f"    nullable: {str(nullable).lower()}",
                f"    primary_key: {str(primary_key).lower()}",
                f"    partition_key: {str(partition_key).lower()}",
                f"    description: {name}.",
            ]
        )
    (schema_dir / "capital_flow_or_chip.yaml").write_text("\n".join(lines), encoding="utf-8")


def _write_security_master_schema(tmp_path: Path) -> None:
    schema_dir = tmp_path / "schemas" / "curated"
    schema_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        ("symbol", "string", False, True, False),
        ("raw_symbol", "string", True, False, False),
        ("exchange", "string", True, False, False),
        ("asset_type", "string", False, False, False),
        ("name", "string", True, False, False),
        ("market", "string", True, False, False),
        ("market_segment", "string", True, False, False),
        ("market_segment_name", "string", True, False, False),
        ("area", "string", True, False, False),
        ("industry", "string", True, False, False),
        ("list_date", "date", True, False, False),
        ("delist_date", "date", True, False, False),
        ("status", "string", False, False, False),
        ("source", "string", False, False, False),
        ("source_batch_id", "string", False, False, False),
        ("data_version", "string", False, False, False),
        ("created_at", "datetime", False, False, False),
    ]
    lines = [
        "dataset_id: security_master",
        "dataset_name: Security Master",
        "layer: curated",
        "version: v001",
        "description: Security master schema.",
        "fields:",
    ]
    for name, field_type, nullable, primary_key, partition_key in fields:
        lines.extend(
            [
                f"  - name: {name}",
                f"    type: {field_type}",
                f"    nullable: {str(nullable).lower()}",
                f"    primary_key: {str(primary_key).lower()}",
                f"    partition_key: {str(partition_key).lower()}",
                f"    description: {name}.",
            ]
        )
    (schema_dir / "security_master.yaml").write_text("\n".join(lines), encoding="utf-8")


def _write_trading_calendar_current(tmp_path: Path) -> None:
    calendar_path = tmp_path / "data" / "curated" / "current" / "trading_calendar" / "part-000.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "calendar_id": ["cn_a_share", "cn_a_share"],
            "trade_date": [date(2026, 4, 27), date(2026, 4, 28)],
            "is_trading_day": [True, True],
        }
    ).write_parquet(calendar_path)


def _write_security_master_current(tmp_path: Path, symbols: list[str]) -> None:
    security_path = tmp_path / "data" / "curated" / "current" / "security_master" / "part-000.parquet"
    security_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": symbols,
            "status": ["active"] * len(symbols),
        }
    ).write_parquet(security_path)


def _write_storage_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "storage.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                "  data_root: ./data",
                "  raw_root: ${storage.data_root}/raw",
                "  curated_root: ${storage.data_root}/curated",
                "  current_curated_root: ${storage.curated_root}/current",
                "  frozen_curated_root: ${storage.curated_root}/frozen",
                "  reports_root: ${storage.data_root}/reports",
                "  backtests_root: ${storage.data_root}/backtests",
                "  metadata_sqlite_path: ${storage.data_root}/metadata.sqlite",
                "  schema_root: ./schemas",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
