"""Command-line entry point for the stock picker project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys

from stock_picker.config import StorageConfig, load_storage_config
from stock_picker.analysis import analyze_stock
from stock_picker.app_worker import run_daily_check, run_holding_price_loop, run_holding_price_refresh, run_worker_loop, run_worker_once
from stock_picker.curated import import_curated_csv, inspect_curated, promote_raw_batch, promote_raw_run
from stock_picker.display import inspect_run, list_runs, preview_curated
from stock_picker.factor_exploration import (
    backtest_candidate_002,
    compute_daily_factors,
    evaluate_factor_run,
    rank_candidate_002,
)
from stock_picker.factor_research import research_candidate_001
from stock_picker.provider import fetch_cyq_perf_batch, fetch_provider_raw, probe_provider_api, run_cyq_perf_batches, run_market_daily
from stock_picker.publish import build_candidate_pool, build_daily_bundle, build_market_status
from stock_picker.quality import check_curated_quality
from stock_picker.reports import show_report
from stock_picker.snapshot import create_snapshot, inspect_snapshot
from stock_picker.storage import init_storage, register_schemas, validate_storage
from stock_picker.strategy import backtest_candidate_001, rank_candidate_001
from stock_picker.sync import sync_latest
from stock_picker.workflow import pause_workflow, stock_analysis_workflow, sync_report_workflow, workflow_status


LOGGER_NAME = "stock_picker"


@dataclass(frozen=True)
class CliContext:
    config_path: Path
    storage_config: StorageConfig


@dataclass(frozen=True)
class CliCommandResult:
    ok: bool
    message: str
    exit_code: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-picker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    storage = subparsers.add_parser("storage", help="Manage local storage paths and metadata")
    storage_subparsers = storage.add_subparsers(dest="storage_command", required=True)

    provider = subparsers.add_parser("provider", help="Fetch raw data from external providers")
    provider_subparsers = provider.add_subparsers(dest="provider_command", required=True)

    strategy = subparsers.add_parser("strategy", help="Run strategy ranking and diagnostics")
    strategy_subparsers = strategy.add_subparsers(dest="strategy_command", required=True)

    factor = subparsers.add_parser("factor", help="Run factor research reports")
    factor_subparsers = factor.add_subparsers(dest="factor_command", required=True)

    reports = subparsers.add_parser("reports", help="Display generated reports")
    reports_subparsers = reports.add_subparsers(dest="reports_command", required=True)

    publish = subparsers.add_parser("publish", help="Build commercial app daily bundle payloads")
    publish_subparsers = publish.add_subparsers(dest="publish_command", required=True)

    analysis = subparsers.add_parser("analysis", help="Build commercial app analysis outputs")
    analysis_subparsers = analysis.add_subparsers(dest="analysis_command", required=True)

    workflow = subparsers.add_parser("workflow", help="Run desktop workflow orchestration")
    workflow_subparsers = workflow.add_subparsers(dest="workflow_command", required=True)

    app_worker = subparsers.add_parser("app-worker", help="Run local app analysis worker")
    app_worker_subparsers = app_worker.add_subparsers(dest="app_worker_command", required=True)

    show_report_cmd = reports_subparsers.add_parser(
        "show-report",
        help="Show a generated factor research report as PrettyTables",
    )
    show_report_cmd.add_argument("--report-id", required=True, help="Report id")
    show_report_cmd.add_argument("--limit", type=int, default=10, help="Maximum rows per displayed table")
    show_report_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    market_status_cmd = publish_subparsers.add_parser(
        "build-market-status",
        help="Build the customer-facing market_status_v001 payload",
    )
    market_status_cmd.add_argument("--trade-date", help="Trade date; defaults to latest available date")
    market_status_cmd.add_argument("--output", help="Output JSON path")
    market_status_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    candidate_pool_cmd = publish_subparsers.add_parser(
        "build-candidate-pool",
        help="Build the customer-facing candidate_pool_v001 payload",
    )
    candidate_pool_cmd.add_argument("--factor-run-id", required=True, help="Candidate 002 factor run id")
    candidate_pool_cmd.add_argument("--trade-date", help="Trade date; defaults to latest factor date")
    candidate_pool_cmd.add_argument("--previous-candidate-pool", help="Previous candidate_pool_v001 JSON path")
    candidate_pool_cmd.add_argument("--previous-bundle", help="Previous daily_publish_bundle_v001 JSON path")
    candidate_pool_cmd.add_argument("--output", help="Output JSON path")
    candidate_pool_cmd.add_argument("--top", type=int, default=10, help="Candidate count")
    candidate_pool_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    daily_bundle_cmd = publish_subparsers.add_parser(
        "build-daily-bundle",
        help="Build the daily_publish_bundle_v001 payload",
    )
    daily_bundle_cmd.add_argument("--factor-run-id", required=True, help="Candidate 002 factor run id")
    daily_bundle_cmd.add_argument("--trade-date", help="Trade date; defaults to latest available date")
    daily_bundle_cmd.add_argument("--previous-candidate-pool", help="Previous candidate_pool_v001 JSON path")
    daily_bundle_cmd.add_argument("--previous-bundle", help="Previous daily_publish_bundle_v001 JSON path")
    daily_bundle_cmd.add_argument("--output", help="Output JSON path")
    daily_bundle_cmd.add_argument("--top", type=int, default=10, help="Candidate count")
    daily_bundle_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    analysis_stock_cmd = analysis_subparsers.add_parser(
        "stock",
        help="Build a structured specific-stock analysis artifact",
    )
    analysis_stock_cmd.add_argument("--symbol", required=True, help="Stock symbol, such as 600519.SH")
    analysis_stock_cmd.add_argument("--factor-run-id", required=True, help="Candidate 002 factor run id")
    analysis_stock_cmd.add_argument("--trade-date", help="Analysis date; defaults to latest factor date")
    analysis_stock_cmd.add_argument("--output", help="Output JSON path")
    analysis_stock_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    workflow_sync_cmd = workflow_subparsers.add_parser(
        "sync-report",
        help="Run the desktop sync-latest -> daily bundle workflow",
    )
    workflow_sync_cmd.add_argument("--workflow-id", help="Stable workflow id for resume/status")
    workflow_sync_cmd.add_argument("--dry-run", action="store_true", help="Only inspect missing data")
    workflow_sync_cmd.add_argument("--confirm", action="store_true", help="Allow real execution after preflight")
    workflow_sync_cmd.add_argument("--factor-run-id", help="Optional Candidate 002 factor run id")
    workflow_sync_cmd.add_argument("--start-date", help="Optional start date")
    workflow_sync_cmd.add_argument("--end-date", help="Optional end date")
    workflow_sync_cmd.add_argument("--top", type=int, default=20, help="Candidate count")
    workflow_sync_cmd.add_argument("--json-events", action="store_true", help="Print workflow events as JSON lines")
    workflow_sync_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    workflow_stock_cmd = workflow_subparsers.add_parser(
        "stock-analysis",
        help="Run stock analysis through the desktop workflow runtime",
    )
    workflow_stock_cmd.add_argument("--symbol", required=True)
    workflow_stock_cmd.add_argument("--factor-run-id", required=True)
    workflow_stock_cmd.add_argument("--workflow-id")
    workflow_stock_cmd.add_argument("--trade-date")
    workflow_stock_cmd.add_argument("--json-events", action="store_true")
    workflow_stock_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    workflow_status_cmd = workflow_subparsers.add_parser("status", help="Show workflow status")
    workflow_status_cmd.add_argument("--workflow-id")
    workflow_status_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    workflow_pause_cmd = workflow_subparsers.add_parser("pause", help="Pause a workflow at the next step boundary")
    workflow_pause_cmd.add_argument("--workflow-id", required=True)
    workflow_pause_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    app_worker_once_cmd = app_worker_subparsers.add_parser("run-once", help="Claim and process at most one app stock_analysis task")
    app_worker_once_cmd.add_argument("--worker-config", default="config/app-worker.yaml", help="Path to local worker config")
    app_worker_once_cmd.add_argument("--mock-task", help="Optional mock task JSON path")
    app_worker_once_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    app_worker_run_cmd = app_worker_subparsers.add_parser("run", help="Run the polling app stock_analysis worker loop")
    app_worker_run_cmd.add_argument("--worker-config", default="config/app-worker.yaml", help="Path to local worker config")
    app_worker_run_cmd.add_argument("--max-iterations", type=int, help="Optional max poll iterations")
    app_worker_run_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    app_worker_daily_cmd = app_worker_subparsers.add_parser("daily-check", help="Build and upload the daily bundle when ready")
    app_worker_daily_cmd.add_argument("--worker-config", default="config/app-worker.yaml", help="Path to local worker config")
    app_worker_daily_cmd.add_argument("--factor-run-id", help="Candidate 002 factor run id; defaults to worker config")
    app_worker_daily_cmd.add_argument("--trade-date", help="Trade date; defaults to worker config or latest available data")
    app_worker_daily_cmd.add_argument("--previous-candidate-pool", help="Previous candidate_pool_v001 JSON path")
    app_worker_daily_cmd.add_argument("--previous-bundle", help="Previous daily_publish_bundle_v001 JSON path")
    app_worker_daily_cmd.add_argument("--top", type=int, default=10, help="Candidate count")
    app_worker_daily_cmd.add_argument("--force", action="store_true", help="Upload even if the same trade_date/hash was already uploaded")
    app_worker_daily_cmd.add_argument("--mock-upload", action="store_true", help="Write the raw daily bundle locally instead of calling APP backend")
    app_worker_daily_cmd.add_argument("--mock-upload-path", help="Optional local JSON path for --mock-upload")
    app_worker_daily_cmd.add_argument("--auto-pipeline", action="store_true", help="Run the daily sync/factor workflow if no local Candidate 002 factor run exists")
    app_worker_daily_cmd.add_argument("--json-events", action="store_true", help="Print worker progress events as JSON lines")
    app_worker_daily_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    app_worker_holding_cmd = app_worker_subparsers.add_parser("refresh-holding-prices", help="Refresh APP holding prices from local Tushare quotes")
    app_worker_holding_cmd.add_argument("--worker-config", default="config/app-worker.yaml", help="Path to local worker config")
    app_worker_holding_cmd.add_argument("--trade-date", help="Optional Tushare trade date; defaults to latest available quote")
    app_worker_holding_cmd.add_argument("--token-env", help="Environment variable containing Tushare token; defaults to worker config")
    app_worker_holding_cmd.add_argument("--mock-watchlist", help="Optional mock APP watchlist JSON path")
    app_worker_holding_cmd.add_argument("--mock-upload-path", help="Optional local JSON path for uploaded price payload")
    app_worker_holding_cmd.add_argument("--limit", type=int, help="Maximum symbols to refresh")
    app_worker_holding_cmd.add_argument("--loop", action="store_true", help="Run repeatedly using the configured holding price poll interval")
    app_worker_holding_cmd.add_argument("--max-iterations", type=int, help="Optional max loop iterations")
    app_worker_holding_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    factor_research_candidate_cmd = factor_subparsers.add_parser(
        "research-candidate-001",
        help="Generate Strategy Candidate 001 v2 factor research artifacts",
    )
    factor_research_candidate_cmd.add_argument("--snapshot-id", required=True, help="Snapshot id")
    factor_research_candidate_cmd.add_argument("--holding-days", type=int, default=20, help="Holding window in trading rows")
    factor_research_candidate_cmd.add_argument("--top", type=int, default=10, help="Candidate count")
    factor_research_candidate_cmd.add_argument("--report-id", help="Optional stable report id")
    factor_research_candidate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    factor_compute_daily_cmd = factor_subparsers.add_parser(
        "compute-daily",
        help="Compute Strategy Candidate 002 daily factor table from a snapshot",
    )
    factor_compute_daily_cmd.add_argument("--snapshot-id", required=True, help="Snapshot id")
    factor_compute_daily_cmd.add_argument("--start-date", required=True, help="Inclusive start date, such as 2026-01-01")
    factor_compute_daily_cmd.add_argument("--end-date", required=True, help="Inclusive end date, such as 2026-04-28")
    factor_compute_daily_cmd.add_argument("--run-id", help="Optional stable factor run id")
    factor_compute_daily_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    factor_evaluate_cmd = factor_subparsers.add_parser(
        "evaluate",
        help="Evaluate a daily factor run with IC, Rank IC, grouped returns, coverage, and correlation",
    )
    factor_evaluate_cmd.add_argument("--factor-run-id", required=True, help="Factor run id from factor compute-daily")
    factor_evaluate_cmd.add_argument("--forward-days", type=int, default=20, help="Forward return window in trading rows")
    factor_evaluate_cmd.add_argument("--groups", type=int, default=5, help="Number of factor groups for grouped returns")
    factor_evaluate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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
    backtest_candidate_cmd.add_argument(
        "--benchmark-symbol",
        action="append",
        help="Benchmark symbol for relative metrics; repeat for multiple benchmarks. Defaults to 000852.SH.",
    )
    backtest_candidate_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    rank_candidate_002_cmd = strategy_subparsers.add_parser(
        "rank-candidate-002",
        help="Rank Strategy Candidate 002 candidates from a factor run",
    )
    rank_candidate_002_cmd.add_argument("--factor-run-id", required=True, help="Factor run id from factor compute-daily")
    rank_candidate_002_cmd.add_argument("--trade-date", help="Trade date to rank; defaults to latest factor date")
    rank_candidate_002_cmd.add_argument("--top", type=int, default=20, help="Candidate count")
    rank_candidate_002_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    backtest_candidate_002_cmd = strategy_subparsers.add_parser(
        "backtest-candidate-002",
        help="Backtest Strategy Candidate 002 from a factor run",
    )
    backtest_candidate_002_cmd.add_argument("--factor-run-id", required=True, help="Factor run id from factor compute-daily")
    backtest_candidate_002_cmd.add_argument("--top", type=int, default=10, help="Candidate count per rebalance date")
    backtest_candidate_002_cmd.add_argument("--rebalance", choices=["daily", "weekly"], default="weekly", help="Rebalance frequency")
    backtest_candidate_002_cmd.add_argument(
        "--benchmark-symbol",
        action="append",
        help="Benchmark symbol for relative metrics; repeat for multiple benchmarks. Defaults to 000852.SH.",
    )
    backtest_candidate_002_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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
    fetch_cyq_batch_cmd.add_argument("--requests-per-minute", type=float, default=180.0, help="cyq_perf provider request rate limit")
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
    run_cyq_cmd.add_argument("--requests-per-minute", type=float, default=180.0, help="cyq_perf provider request rate limit")
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
    probe_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

    sync_latest_cmd = provider_subparsers.add_parser(
        "sync-latest",
        help="Find missing curated trading dates and fetch data through the latest provider trading day",
    )
    sync_latest_cmd.add_argument("--source", default="tushare", help="Provider source, currently tushare")
    sync_latest_cmd.add_argument(
        "--dataset",
        action="append",
        help="Dataset to sync; repeat for multiple datasets. Defaults to daily_prices, moneyflow_dc, and cyq_perf.",
    )
    sync_latest_cmd.add_argument("--start-date", help="Optional inclusive first trading date to inspect")
    sync_latest_cmd.add_argument("--end-date", help="Calendar end date for latest trading-day lookup; defaults to today")
    sync_latest_cmd.add_argument("--calendar-lookback-days", type=int, default=45, help="Calendar lookup fallback window")
    sync_latest_cmd.add_argument("--dry-run", action="store_true", help="Only display missing dates without fetching")
    sync_latest_cmd.add_argument("--run-id-prefix", default="sync_latest", help="Provider run id prefix")
    sync_latest_cmd.add_argument("--max-market-tasks", type=int, default=1000, help="Maximum market daily tasks to execute")
    sync_latest_cmd.add_argument("--requests-per-minute", type=float, default=40.0, help="Provider request rate limit")
    sync_latest_cmd.add_argument("--symbol-batch-size", type=int, default=1000, help="Symbols per moneyflow_dc task")
    sync_latest_cmd.add_argument("--max-cyq-batches", type=int, default=100000, help="Maximum cyq_perf symbol batches to execute")
    sync_latest_cmd.add_argument("--cyq-batch-size", type=int, default=100, help="Symbols per cyq_perf task")
    sync_latest_cmd.add_argument("--cyq-requests-per-minute", type=float, default=180.0, help="cyq_perf provider request rate limit")
    sync_latest_cmd.add_argument(
        "--benchmark-symbol",
        action="append",
        help="Benchmark index symbol to sync for index_daily; repeat for multiple symbols.",
    )
    sync_latest_cmd.add_argument("--retry", type=int, default=3, help="Retries per task after the first attempt")
    sync_latest_cmd.add_argument("--retry-wait-seconds", type=float, default=60.0, help="Initial retry wait")
    sync_latest_cmd.add_argument("--backoff-multiplier", type=float, default=2.0, help="Retry wait multiplier")
    sync_latest_cmd.add_argument(
        "--create-snapshot",
        action="store_true",
        help="Create a snapshot after successful promotion and quality checks",
    )
    sync_latest_cmd.add_argument(
        "--progress-every-tasks",
        type=int,
        default=50,
        help="Print market daily progress after every N tasks; use 0 to disable",
    )
    sync_latest_cmd.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Print cyq_perf progress after every N batches; use 0 to disable",
    )
    sync_latest_cmd.add_argument("--token-env", default="TUSHARE_TOKEN", help="Environment variable containing provider token")
    sync_latest_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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

    promote_raw_run_cmd = storage_subparsers.add_parser(
        "promote-raw-run",
        help="Promote all successful raw batches from one provider run into curated current stores",
    )
    promote_raw_run_cmd.add_argument("--run-id", required=True, help="Provider run id")
    promote_raw_run_cmd.add_argument("--dataset", help="Optional raw dataset filter, such as daily_prices")
    promote_raw_run_cmd.add_argument("--config", default="config/storage.yaml", help="Path to storage config")

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
    return after_execute(run_cli_command(args))


def run_cli_command(args: argparse.Namespace) -> CliCommandResult:
    before = before_execute(args)
    if isinstance(before, CliCommandResult):
        return before
    return _to_cli_result(execute_command(args, before))


def before_execute(args: argparse.Namespace) -> CliContext | CliCommandResult:
    config_path = Path(getattr(args, "config", "config/storage.yaml"))
    try:
        storage_config = load_storage_config(config_path)
    except Exception as error:
        return CliCommandResult(False, f"failed to load storage config: {error}")
    token_env = getattr(args, "token_env", None)
    if token_env:
        _load_env_value(token_env, storage_config.project_root / ".env")
    return CliContext(config_path=config_path, storage_config=storage_config)


def execute_command(args: argparse.Namespace, context: CliContext):
    if args.command == "storage" and args.storage_command == "init":
        return init_storage(context.config_path)

    if args.command == "storage" and args.storage_command == "validate":
        return validate_storage(context.config_path)

    if args.command == "storage" and args.storage_command == "register-schemas":
        return register_schemas(context.config_path)

    if args.command == "storage" and args.storage_command == "import-curated-csv":
        return import_curated_csv(
            config_path=context.config_path,
            dataset_id=args.dataset,
            input_path=Path(args.input),
            source=args.source,
            as_of_date=args.as_of_date,
        )

    if args.command == "storage" and args.storage_command == "promote-raw":
        return promote_raw_batch(
            config_path=context.config_path,
            source=args.source,
            dataset=args.dataset,
            as_of_date=args.as_of_date,
            batch_id=args.batch_id,
        )

    if args.command == "storage" and args.storage_command == "promote-raw-run":
        return promote_raw_run(
            config_path=context.config_path,
            run_id=args.run_id,
            dataset=args.dataset,
        )

    if args.command == "storage" and args.storage_command == "inspect-curated":
        return inspect_curated(context.config_path, args.dataset)

    if args.command == "storage" and args.storage_command == "preview-curated":
        return preview_curated(
            config_path=context.config_path,
            dataset_id=args.dataset,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            columns=args.columns,
            limit=args.limit,
        )

    if args.command == "storage" and args.storage_command == "list-runs":
        return list_runs(context.config_path, args.limit)

    if args.command == "storage" and args.storage_command == "inspect-run":
        return inspect_run(context.config_path, args.batch_id)

    if args.command == "storage" and args.storage_command == "check-quality":
        return check_curated_quality(context.config_path, args.dataset)

    if args.command == "storage" and args.storage_command == "create-snapshot":
        return create_snapshot(
            context.config_path,
            as_of_date=args.as_of_date,
            config_version=args.config_version,
        )

    if args.command == "storage" and args.storage_command == "inspect-snapshot":
        return inspect_snapshot(context.config_path, args.snapshot_id)

    if args.command == "provider" and args.provider_command == "fetch":
        return fetch_provider_raw(
            config_path=context.config_path,
            source=args.source,
            dataset=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            ts_code=args.ts_code,
            trade_date=args.trade_date,
            token_env=args.token_env,
        )

    if args.command == "provider" and args.provider_command == "fetch-cyq-perf-batch":
        return fetch_cyq_perf_batch(
            config_path=context.config_path,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            symbols=args.symbol,
            limit=args.limit,
            offset=args.offset,
            requests_per_minute=args.requests_per_minute,
            token_env=args.token_env,
        )

    if args.command == "provider" and args.provider_command == "run-cyq-perf-batches":
        progress_logger = _configure_progress_logger()
        return run_cyq_perf_batches(
            config_path=context.config_path,
            source=args.source,
            run_id=args.run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            requests_per_minute=args.requests_per_minute,
            token_env=args.token_env,
            progress_every_batches=args.progress_every_batches,
            progress_callback=progress_logger.info,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
        )

    if args.command == "provider" and args.provider_command == "run-market-daily":
        progress_logger = _configure_progress_logger()
        return run_market_daily(
            config_path=context.config_path,
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
            progress_callback=progress_logger.info,
        )

    if args.command == "provider" and args.provider_command == "probe":
        return probe_provider_api(
            source=args.source,
            api=args.api,
            ts_code=args.ts_code,
            trade_date=args.trade_date,
            start_date=args.start_date,
            end_date=args.end_date,
            token_env=args.token_env,
        )

    if args.command == "provider" and args.provider_command == "sync-latest":
        progress_logger = _configure_progress_logger()
        return sync_latest(
            config_path=context.config_path,
            source=args.source,
            datasets=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            calendar_lookback_days=args.calendar_lookback_days,
            dry_run=args.dry_run,
            run_id_prefix=args.run_id_prefix,
            max_market_tasks=args.max_market_tasks,
            requests_per_minute=args.requests_per_minute,
            symbol_batch_size=args.symbol_batch_size,
            max_cyq_batches=args.max_cyq_batches,
            cyq_batch_size=args.cyq_batch_size,
            cyq_requests_per_minute=args.cyq_requests_per_minute,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
            create_snapshot_after=args.create_snapshot,
            token_env=args.token_env,
            benchmark_symbols=args.benchmark_symbol,
            progress_every_tasks=args.progress_every_tasks,
            progress_every_batches=args.progress_every_batches,
            progress_callback=progress_logger.info,
        )

    if args.command == "strategy" and args.strategy_command == "rank-candidate-001":
        return rank_candidate_001(context.config_path, args.snapshot_id, args.top)

    if args.command == "strategy" and args.strategy_command == "backtest-candidate-001":
        return backtest_candidate_001(context.config_path, args.snapshot_id, args.holding_days, args.top, args.benchmark_symbol or "000852.SH")

    if args.command == "strategy" and args.strategy_command == "rank-candidate-002":
        return rank_candidate_002(context.config_path, args.factor_run_id, args.trade_date, args.top)

    if args.command == "strategy" and args.strategy_command == "backtest-candidate-002":
        return backtest_candidate_002(
            context.config_path,
            factor_run_id=args.factor_run_id,
            top=args.top,
            rebalance=args.rebalance,
            benchmark_symbol=args.benchmark_symbol or "000852.SH",
        )

    if args.command == "factor" and args.factor_command == "research-candidate-001":
        return research_candidate_001(
            context.config_path,
            snapshot_id=args.snapshot_id,
            holding_days=args.holding_days,
            top=args.top,
            report_id=args.report_id,
        )

    if args.command == "factor" and args.factor_command == "compute-daily":
        return compute_daily_factors(
            context.config_path,
            snapshot_id=args.snapshot_id,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=args.run_id,
        )

    if args.command == "factor" and args.factor_command == "evaluate":
        return evaluate_factor_run(
            context.config_path,
            factor_run_id=args.factor_run_id,
            forward_days=args.forward_days,
            groups=args.groups,
        )

    if args.command == "reports" and args.reports_command == "show-report":
        return show_report(
            context.config_path,
            report_id=args.report_id,
            limit=args.limit,
        )

    if args.command == "publish" and args.publish_command == "build-market-status":
        return build_market_status(
            context.config_path,
            trade_date=args.trade_date,
            output_path=Path(args.output) if args.output else None,
        )

    if args.command == "publish" and args.publish_command == "build-candidate-pool":
        return build_candidate_pool(
            context.config_path,
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            previous_candidate_pool_path=Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            previous_bundle_path=Path(args.previous_bundle) if args.previous_bundle else None,
            output_path=Path(args.output) if args.output else None,
            top=args.top,
        )

    if args.command == "publish" and args.publish_command == "build-daily-bundle":
        return build_daily_bundle(
            context.config_path,
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            previous_candidate_pool_path=Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            previous_bundle_path=Path(args.previous_bundle) if args.previous_bundle else None,
            output_path=Path(args.output) if args.output else None,
            top=args.top,
        )

    if args.command == "analysis" and args.analysis_command == "stock":
        return analyze_stock(
            context.config_path,
            symbol=args.symbol,
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            output_path=Path(args.output) if args.output else None,
        )

    if args.command == "workflow" and args.workflow_command == "sync-report":
        return sync_report_workflow(
            context.config_path,
            workflow_id=args.workflow_id,
            dry_run=args.dry_run or not args.confirm,
            confirmed=args.confirm,
            factor_run_id=args.factor_run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            top=args.top,
            emit=_json_event_printer if args.json_events else None,
        )

    if args.command == "workflow" and args.workflow_command == "stock-analysis":
        return stock_analysis_workflow(
            context.config_path,
            symbol=args.symbol,
            factor_run_id=args.factor_run_id,
            workflow_id=args.workflow_id,
            trade_date=args.trade_date,
            emit=_json_event_printer if args.json_events else None,
        )

    if args.command == "workflow" and args.workflow_command == "status":
        return workflow_status(context.config_path, args.workflow_id)

    if args.command == "workflow" and args.workflow_command == "pause":
        return pause_workflow(context.config_path, args.workflow_id)

    if args.command == "app-worker" and args.app_worker_command == "run-once":
        return run_worker_once(
            context.config_path,
            worker_config_path=Path(args.worker_config),
            mock_task_path=Path(args.mock_task) if args.mock_task else None,
        )

    if args.command == "app-worker" and args.app_worker_command == "run":
        return run_worker_loop(
            context.config_path,
            worker_config_path=Path(args.worker_config),
            max_iterations=args.max_iterations,
        )

    if args.command == "app-worker" and args.app_worker_command == "daily-check":
        return run_daily_check(
            context.config_path,
            worker_config_path=Path(args.worker_config),
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            previous_candidate_pool_path=Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            previous_bundle_path=Path(args.previous_bundle) if args.previous_bundle else None,
            top=args.top,
            force=args.force,
            mock_upload=args.mock_upload,
            mock_upload_path=Path(args.mock_upload_path) if args.mock_upload_path else None,
            json_events=args.json_events,
            auto_pipeline=args.auto_pipeline,
        )

    if args.command == "app-worker" and args.app_worker_command == "refresh-holding-prices":
        if args.loop:
            return run_holding_price_loop(
                worker_config_path=Path(args.worker_config),
                trade_date=args.trade_date,
                token_env=args.token_env,
                mock_watchlist_path=Path(args.mock_watchlist) if args.mock_watchlist else None,
                mock_upload_path=Path(args.mock_upload_path) if args.mock_upload_path else None,
                limit=args.limit,
                max_iterations=args.max_iterations,
            )
        return run_holding_price_refresh(
            worker_config_path=Path(args.worker_config),
            trade_date=args.trade_date,
            token_env=args.token_env,
            mock_watchlist_path=Path(args.mock_watchlist) if args.mock_watchlist else None,
            mock_upload_path=Path(args.mock_upload_path) if args.mock_upload_path else None,
            limit=args.limit,
        )

    return CliCommandResult(False, "unsupported command", 2)


def after_execute(result: CliCommandResult) -> int:
    print(result.message)
    if result.exit_code is not None:
        return result.exit_code
    return 0 if result.ok else 1


def _to_cli_result(result: object) -> CliCommandResult:
    if isinstance(result, CliCommandResult):
        return result
    ok = bool(getattr(result, "ok", False))
    message = str(getattr(result, "message", result))
    return CliCommandResult(ok=ok, message=message)


def _load_env_value(token_env: str, env_path: Path) -> str | None:
    token = os.environ.get(token_env)
    if token:
        return token
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != token_env:
            continue
        token = value.strip().strip("\"'")
        if token:
            os.environ[token_env] = token
            return token
    return None


def _configure_progress_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _json_event_printer(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
