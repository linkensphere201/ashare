from pathlib import Path
import sqlite3

from stock_picker.storage import init_storage, validate_storage


def test_storage_init_and_validate(tmp_path: Path) -> None:
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
