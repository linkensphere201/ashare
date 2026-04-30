from pathlib import Path
from datetime import date, timedelta
import json
import sqlite3

import polars as pl

from stock_picker.curated import import_curated_csv, inspect_curated, promote_raw_batch
from stock_picker.display import inspect_run, list_runs, preview_curated
from stock_picker.provider import fetch_cyq_perf_batch, fetch_provider_raw, probe_provider_api, write_raw_batch
from stock_picker.quality import check_curated_quality
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
    dates = [latest - timedelta(days=44 - index) for index in range(45)]
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
            "symbol": ["600519.SH"] * 45,
            "trade_date": dates,
            "close": [10.0] * 39 + [20.0] + [22.0] * 5,
            "amount": [1000.0] * 45,
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
    assert "trade_count: 1" in result.message
    assert "avg_forward_return: 0.100000" in result.message


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
