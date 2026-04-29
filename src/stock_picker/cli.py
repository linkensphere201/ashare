"""Command-line entry point for the stock picker project."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_picker.curated import import_curated_csv, inspect_curated
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-picker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser("storage", help="Manage local storage paths and metadata")
    storage_subparsers = storage.add_subparsers(dest="storage_command", required=True)

    init_cmd = storage_subparsers.add_parser("init", help="Create configured storage directories and metadata tables")
    init_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    validate_cmd = storage_subparsers.add_parser("validate", help="Validate configured storage directories and metadata")
    validate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    register_schemas_cmd = storage_subparsers.add_parser(
        "register-schemas",
        help="Register schema YAML files into the SQLite metadata catalog",
    )
    register_schemas_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    import_curated_cmd = storage_subparsers.add_parser(
        "import-curated-csv",
        help="Import a local CSV into the curated current Parquet store",
    )
    import_curated_cmd.add_argument("--dataset", required=True, help="Registered dataset id")
    import_curated_cmd.add_argument("--input", required=True, help="Input CSV path")
    import_curated_cmd.add_argument("--source", default="manual_csv", help="Source name for lineage metadata")
    import_curated_cmd.add_argument("--as-of-date", help="Business as-of date, such as 2026-04-28")
    import_curated_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    inspect_curated_cmd = storage_subparsers.add_parser(
        "inspect-curated",
        help="Inspect the current curated Parquet store and metadata for one dataset",
    )
    inspect_curated_cmd.add_argument("--dataset", required=True, help="Registered dataset id")
    inspect_curated_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    check_quality_cmd = storage_subparsers.add_parser(
        "check-quality",
        help="Run MVP quality checks against curated current datasets",
    )
    check_quality_cmd.add_argument(
        "--dataset",
        action="append",
        help="Dataset id to check; repeat for multiple datasets. Defaults to all curated datasets.",
    )
    check_quality_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    create_snapshot_cmd = storage_subparsers.add_parser(
        "create-snapshot",
        help="Create a logical data snapshot manifest from current curated versions",
    )
    create_snapshot_cmd.add_argument("--as-of-date", required=True, help="Snapshot as-of date, such as 2026-04-28")
    create_snapshot_cmd.add_argument("--config-version", default="manual_config_v001", help="Config version label")
    create_snapshot_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    inspect_snapshot_cmd = storage_subparsers.add_parser(
        "inspect-snapshot",
        help="Inspect a stored logical data snapshot manifest",
    )
    inspect_snapshot_cmd.add_argument("--snapshot-id", required=True, help="Snapshot id")
    inspect_snapshot_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "storage" and args.storage_command == "init":
        result = init_storage(Path(args.config))
        print(result.message)
        return 0

    if args.command == "storage" and args.storage_command == "validate":
        result = validate_storage(Path(args.config))
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "register-schemas":
        result = register_schemas(Path(args.config))
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "import-curated-csv":
        result = import_curated_csv(
            config_path=Path(args.config),
            dataset_id=args.dataset,
            input_path=Path(args.input),
            source=args.source,
            as_of_date=args.as_of_date,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "inspect-curated":
        result = inspect_curated(Path(args.config), args.dataset)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "check-quality":
        result = check_curated_quality(Path(args.config), args.dataset)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "create-snapshot":
        result = create_snapshot(
            Path(args.config),
            as_of_date=args.as_of_date,
            config_version=args.config_version,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "inspect-snapshot":
        result = inspect_snapshot(Path(args.config), args.snapshot_id)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
