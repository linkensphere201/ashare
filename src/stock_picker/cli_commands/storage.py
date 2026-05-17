from __future__ import annotations

from pathlib import Path

from stock_picker.curated import import_curated_csv, inspect_curated, promote_raw_batch, promote_raw_run
from stock_picker.display import inspect_run, list_runs, preview_curated
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage


def execute(args, context):
    if args.storage_command == "init":
        return init_storage(context.config_path)
    if args.storage_command == "validate":
        return validate_storage(context.config_path)
    if args.storage_command == "register-schemas":
        return register_schemas(context.config_path)
    if args.storage_command == "import-curated-csv":
        return import_curated_csv(context.config_path, args.dataset, Path(args.input), args.source, args.as_of_date)
    if args.storage_command == "promote-raw":
        return promote_raw_batch(context.config_path, args.source, args.dataset, args.as_of_date, args.batch_id)
    if args.storage_command == "promote-raw-run":
        return promote_raw_run(context.config_path, args.run_id, args.dataset)
    if args.storage_command == "inspect-curated":
        return inspect_curated(context.config_path, args.dataset)
    if args.storage_command == "preview-curated":
        return preview_curated(context.config_path, args.dataset, args.symbol, args.start_date, args.end_date, args.columns, args.limit)
    if args.storage_command == "list-runs":
        return list_runs(context.config_path, args.limit)
    if args.storage_command == "inspect-run":
        return inspect_run(context.config_path, args.batch_id)
    if args.storage_command == "check-quality":
        return check_curated_quality(context.config_path, args.dataset)
    if args.storage_command == "create-snapshot":
        return create_snapshot(context.config_path, args.as_of_date, args.config_version)
    if args.storage_command == "inspect-snapshot":
        return inspect_snapshot(context.config_path, args.snapshot_id)
    return None
