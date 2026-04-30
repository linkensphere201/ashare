"""Command-line entry point for the stock picker project."""

from __future__ import annotations

import argparse
from pathlib import Path

from stock_picker.curated import import_curated_csv, inspect_curated, promote_raw_batch
from stock_picker.display import inspect_run, list_runs, preview_curated
from stock_picker.provider import fetch_cyq_perf_batch, fetch_provider_raw, probe_provider_api, run_cyq_perf_batches, run_market_daily
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage
from stock_picker.strategy import backtest_candidate_001, rank_candidate_001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-picker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser("storage", help="Manage local storage paths and metadata")
    storage_subparsers = storage.add_subparsers(dest="storage_command", required=True)

    provider = subparsers.add_parser("provider", help="Fetch raw data from external providers")
    provider_subparsers = provider.add_subparsers(dest="provider_command", required=True)

    strategy = subparsers.add_parser("strategy", help="Run strategy ranking and diagnostics")
    strategy_subparsers = strategy.add_subparsers(dest="strategy_command", required=True)

    rank_candidate_cmd = strategy_subparsers.add_parser(
        "rank-candidate-001",
        help="Rank Strategy Candidate 001 v2 candidates from a snapshot",
    )
    rank_candidate_cmd.add_argument("--snapshot-id", required=True, help="Snapshot id")
    rank_candidate_cmd.add_argument("--top", type=int, default=10, help="Candidate count")
    rank_candidate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    backtest_candidate_cmd = strategy_subparsers.add_parser(
        "backtest-candidate-001",
        help="Backtest Strategy Candidate 001 v2 forward returns from a snapshot",
    )
    backtest_candidate_cmd.add_argument("--snapshot-id", required=True, help="Snapshot id")
    backtest_candidate_cmd.add_argument("--holding-days", type=int, default=20, help="Holding window in trading rows")
    backtest_candidate_cmd.add_argument("--top", type=int, default=10, help="Candidate count per signal date")
    backtest_candidate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    fetch_cmd = provider_subparsers.add_parser("fetch", help="Fetch a raw provider dataset into the raw store")
    fetch_cmd.add_argument("--source", required=True, help="Provider source, such as tushare")
    fetch_cmd.add_argument("--dataset", required=True, help="Dataset id, such as daily_prices")
    fetch_cmd.add_argument("--ts-code", help="Provider stock code, such as 600519.SH")
    fetch_cmd.add_argument("--trade-date", help="Single trade date, such as 20260428 or 2026-04-28")
    fetch_cmd.add_argument("--start-date", help="Start date, such as 2026-01-01")
    fetch_cmd.add_argument("--end-date", help="End date, such as 2026-04-28")
    fetch_cmd.add_argument("--as-of-date", help="Business as-of date for the raw batch")
    fetch_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")
    fetch_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    fetch_cyq_batch_cmd = provider_subparsers.add_parser(
        "fetch-cyq-perf-batch",
        help="Fetch Tushare cyq_perf for multiple symbols into one raw batch",
    )
    fetch_cyq_batch_cmd.add_argument("--source", default="tushare", help="Provider source, currently tushare")
    fetch_cyq_batch_cmd.add_argument(
        "--symbol",
        action="append",
        help="Optional symbol to fetch; repeat for multiple symbols. Defaults to active security_master symbols.",
    )
    fetch_cyq_batch_cmd.add_argument("--start-date", help="Start date, such as 2026-01-01")
    fetch_cyq_batch_cmd.add_argument("--end-date", help="End date, such as 2026-04-28")
    fetch_cyq_batch_cmd.add_argument("--as-of-date", help="Business as-of date for the raw batch")
    fetch_cyq_batch_cmd.add_argument("--limit", type=int, help="Maximum symbols to fetch")
    fetch_cyq_batch_cmd.add_argument("--offset", type=int, default=0, help="Number of symbols to skip")
    fetch_cyq_batch_cmd.add_argument("--delay-seconds", type=float, default=0.0, help="Sleep between symbol requests")
    fetch_cyq_batch_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")
    fetch_cyq_batch_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    run_cyq_cmd = provider_subparsers.add_parser(
        "run-cyq-perf-batches",
        help="Run resumable multi-batch Tushare cyq_perf fetching",
    )
    run_cyq_cmd.add_argument("--source", default="tushare", help="Provider source, currently tushare")
    run_cyq_cmd.add_argument("--run-id", help="Optional run id. Defaults to source/dataset/as-of-date run id.")
    run_cyq_cmd.add_argument("--start-date", help="Start date, such as 2026-01-01")
    run_cyq_cmd.add_argument("--end-date", help="End date, such as 2026-04-28")
    run_cyq_cmd.add_argument("--as-of-date", help="Business as-of date for raw batches")
    run_cyq_cmd.add_argument("--batch-size", type=int, default=100, help="Symbols per raw batch")
    run_cyq_cmd.add_argument("--max-batches", type=int, default=1, help="Maximum batches to run in this invocation")
    run_cyq_cmd.add_argument("--delay-seconds", type=float, default=0.0, help="Sleep between symbol requests")
    run_cyq_cmd.add_argument("--retry", type=int, default=3, help="Retries per symbol after the first attempt")
    run_cyq_cmd.add_argument("--retry-wait-seconds", type=float, default=60.0, help="Initial retry wait for symbol fetch failures")
    run_cyq_cmd.add_argument("--backoff-multiplier", type=float, default=2.0, help="Retry wait multiplier")
    run_cyq_cmd.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Print progress after every N completed batches; use 0 to disable",
    )
    run_cyq_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")
    run_cyq_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    run_market_daily_cmd = provider_subparsers.add_parser(
        "run-market-daily",
        help="Run resumable date-task Tushare daily market fetching",
    )
    run_market_daily_cmd.add_argument("--source", default="tushare", help="Provider source, currently tushare")
    run_market_daily_cmd.add_argument("--run-id", help="Optional run id. Defaults to source/window run id.")
    run_market_daily_cmd.add_argument(
        "--dataset",
        action="append",
        help="Dataset to fetch; repeat for multiple datasets. Defaults to daily_prices and moneyflow_dc.",
    )
    run_market_daily_cmd.add_argument("--start-date", required=True, help="Start date, such as 2025-04-28")
    run_market_daily_cmd.add_argument("--end-date", required=True, help="End date, such as 2026-04-28")
    run_market_daily_cmd.add_argument("--as-of-date", help="Business as-of date for raw batches")
    run_market_daily_cmd.add_argument("--max-tasks", type=int, default=1, help="Maximum date tasks to run in this invocation")
    run_market_daily_cmd.add_argument("--requests-per-minute", type=float, default=40.0, help="Provider request rate limit")
    run_market_daily_cmd.add_argument("--retry", type=int, default=3, help="Retries per task after the first attempt")
    run_market_daily_cmd.add_argument("--retry-wait-seconds", type=float, default=60.0, help="Initial retry wait")
    run_market_daily_cmd.add_argument("--backoff-multiplier", type=float, default=2.0, help="Retry wait multiplier")
    run_market_daily_cmd.add_argument("--symbol-batch-size", type=int, default=1000, help="Symbols per moneyflow_dc task")
    run_market_daily_cmd.add_argument(
        "--progress-every-tasks",
        type=int,
        default=50,
        help="Print progress after every N completed tasks; use 0 to disable",
    )
    run_market_daily_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")
    run_market_daily_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    probe_cmd = provider_subparsers.add_parser("probe", help="Probe provider API access and expected fields")
    probe_cmd.add_argument("--source", required=True, help="Provider source, such as tushare")
    probe_cmd.add_argument("--api", required=True, help="Provider API name, such as cyq_perf")
    probe_cmd.add_argument("--ts-code", help="Provider stock code, such as 600519.SH")
    probe_cmd.add_argument("--trade-date", help="Single trade date, such as 20260428 or 2026-04-28")
    probe_cmd.add_argument("--start-date", help="Start date, such as 2026-01-01")
    probe_cmd.add_argument("--end-date", help="End date, such as 2026-04-28")
    probe_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")

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

    promote_raw_cmd = storage_subparsers.add_parser(
        "promote-raw",
        help="Promote a raw provider batch into the curated current Parquet store",
    )
    promote_raw_cmd.add_argument("--source", required=True, help="Raw source, such as tushare")
    promote_raw_cmd.add_argument("--dataset", required=True, help="Raw dataset id, such as daily_prices")
    promote_raw_cmd.add_argument("--as-of-date", help="Business as-of date for selecting the raw batch")
    promote_raw_cmd.add_argument("--batch-id", help="Exact raw batch id to promote")
    promote_raw_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    inspect_curated_cmd = storage_subparsers.add_parser(
        "inspect-curated",
        help="Inspect the current curated Parquet store and metadata for one dataset",
    )
    inspect_curated_cmd.add_argument("--dataset", required=True, help="Registered dataset id")
    inspect_curated_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    preview_curated_cmd = storage_subparsers.add_parser(
        "preview-curated",
        help="Preview curated current rows for human inspection",
    )
    preview_curated_cmd.add_argument("--dataset", required=True, help="Registered dataset id")
    preview_curated_cmd.add_argument("--symbol", help="Optional symbol filter, such as 600519.SH")
    preview_curated_cmd.add_argument("--start-date", help="Optional inclusive start date, such as 2026-04-26")
    preview_curated_cmd.add_argument("--end-date", help="Optional inclusive end date, such as 2026-04-28")
    preview_curated_cmd.add_argument("--columns", help="Optional comma-separated columns to display")
    preview_curated_cmd.add_argument("--limit", type=int, default=20, help="Maximum rows to display")
    preview_curated_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    list_runs_cmd = storage_subparsers.add_parser(
        "list-runs",
        help="List raw data import runs from the metadata catalog",
    )
    list_runs_cmd.add_argument("--limit", type=int, default=20, help="Maximum runs to display")
    list_runs_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    inspect_run_cmd = storage_subparsers.add_parser(
        "inspect-run",
        help="Inspect one raw data import run and linked curated/snapshot outputs",
    )
    inspect_run_cmd.add_argument("--batch-id", required=True, help="Raw data batch id")
    inspect_run_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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

    if args.command == "storage" and args.storage_command == "promote-raw":
        result = promote_raw_batch(
            config_path=Path(args.config),
            source=args.source,
            dataset=args.dataset,
            as_of_date=args.as_of_date,
            batch_id=args.batch_id,
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

    if args.command == "storage" and args.storage_command == "preview-curated":
        result = preview_curated(
            config_path=Path(args.config),
            dataset_id=args.dataset,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            columns=args.columns,
            limit=args.limit,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "list-runs":
        result = list_runs(Path(args.config), args.limit)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "storage" and args.storage_command == "inspect-run":
        result = inspect_run(Path(args.config), args.batch_id)
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

    if args.command == "provider" and args.provider_command == "fetch":
        result = fetch_provider_raw(
            config_path=Path(args.config),
            source=args.source,
            dataset=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            ts_code=args.ts_code,
            trade_date=args.trade_date,
            token_env=args.token_env,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "provider" and args.provider_command == "fetch-cyq-perf-batch":
        result = fetch_cyq_perf_batch(
            config_path=Path(args.config),
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            symbols=args.symbol,
            limit=args.limit,
            offset=args.offset,
            delay_seconds=args.delay_seconds,
            token_env=args.token_env,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "provider" and args.provider_command == "run-cyq-perf-batches":
        result = run_cyq_perf_batches(
            config_path=Path(args.config),
            source=args.source,
            run_id=args.run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            delay_seconds=args.delay_seconds,
            token_env=args.token_env,
            progress_every_batches=args.progress_every_batches,
            progress_callback=print,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "provider" and args.provider_command == "run-market-daily":
        result = run_market_daily(
            config_path=Path(args.config),
            source=args.source,
            run_id=args.run_id,
            datasets=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            max_tasks=args.max_tasks,
            requests_per_minute=args.requests_per_minute,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
            symbol_batch_size=args.symbol_batch_size,
            token_env=args.token_env,
            progress_every_tasks=args.progress_every_tasks,
            progress_callback=print,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "provider" and args.provider_command == "probe":
        result = probe_provider_api(
            source=args.source,
            api=args.api,
            ts_code=args.ts_code,
            trade_date=args.trade_date,
            start_date=args.start_date,
            end_date=args.end_date,
            token_env=args.token_env,
        )
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "strategy" and args.strategy_command == "rank-candidate-001":
        result = rank_candidate_001(Path(args.config), args.snapshot_id, args.top)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    if args.command == "strategy" and args.strategy_command == "backtest-candidate-001":
        result = backtest_candidate_001(Path(args.config), args.snapshot_id, args.holding_days, args.top)
        if result.ok:
            print(result.message)
            return 0
        print(result.message)
        return 1

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
