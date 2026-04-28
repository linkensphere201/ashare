"""Command-line entry point for the stock picker project."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_picker.storage import init_storage, validate_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-picker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser("storage", help="Manage local storage paths and metadata")
    storage_subparsers = storage.add_subparsers(dest="storage_command", required=True)

    init_cmd = storage_subparsers.add_parser("init", help="Create configured storage directories and metadata tables")
    init_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    validate_cmd = storage_subparsers.add_parser("validate", help="Validate configured storage directories and metadata")
    validate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
