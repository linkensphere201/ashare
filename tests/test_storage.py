from pathlib import Path
import sqlite3

import polars as pl

from stock_picker.curated import import_curated_csv, inspect_curated
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage


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
