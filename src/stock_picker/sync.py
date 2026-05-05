"""High-level provider synchronization workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.curated import promote_raw_batch, promote_raw_run
from stock_picker.provider import _fetch_tushare_dataset, run_cyq_perf_batches, run_market_daily, write_raw_batch
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class SyncLatestResult:
    ok: bool
    message: str


DEFAULT_SYNC_DATASETS = ["daily_prices", "moneyflow_dc", "cyq_perf"]
MARKET_SYNC_DATASETS = {"daily_prices", "moneyflow_dc"}
SUPPORTED_SYNC_DATASETS = set(DEFAULT_SYNC_DATASETS)


def sync_latest(
    config_path: Path,
    source: str = "tushare",
    datasets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    calendar_lookback_days: int = 45,
    dry_run: bool = False,
    run_id_prefix: str = "sync_latest",
    max_market_tasks: int = 1000,
    requests_per_minute: float = 40.0,
    symbol_batch_size: int = 1000,
    max_cyq_batches: int = 100000,
    cyq_batch_size: int = 100,
    cyq_requests_per_minute: float = 180.0,
    retry: int = 3,
    retry_wait_seconds: float = 60.0,
    backoff_multiplier: float = 2.0,
    create_snapshot_after: bool = False,
    token_env: str = "TUSHARE_TOKEN",
    progress_every_tasks: int = 50,
    progress_every_batches: int = 1,
    progress_callback=None,
) -> SyncLatestResult:
    if source != "tushare":
        return SyncLatestResult(False, f"unsupported provider source: {source}")
    selected_datasets = datasets or DEFAULT_SYNC_DATASETS
    unsupported = sorted(set(selected_datasets) - SUPPORTED_SYNC_DATASETS)
    if unsupported:
        return SyncLatestResult(False, "unsupported sync datasets: " + ", ".join(unsupported))
    if calendar_lookback_days <= 0:
        return SyncLatestResult(False, "sync-latest requires a positive --calendar-lookback-days")
    if max_market_tasks <= 0:
        return SyncLatestResult(False, "sync-latest requires a positive --max-market-tasks")
    if max_cyq_batches <= 0:
        return SyncLatestResult(False, "sync-latest requires a positive --max-cyq-batches")
    if cyq_requests_per_minute < 0:
        return SyncLatestResult(False, "sync-latest requires a non-negative --cyq-requests-per-minute")

    config = load_storage_config(config_path)
    token = _load_token(token_env, config.project_root / ".env")
    if not token:
        return SyncLatestResult(False, f"missing required environment variable or .env value: {token_env}")
    initialize_metadata_catalog(config.metadata_sqlite_path)
    today = end_date or date.today().isoformat()
    calendar_start = _calendar_fetch_start(config.current_curated_root / "trading_calendar" / "part-000.parquet", today, calendar_lookback_days)

    try:
        provider_calendar = _fetch_tushare_dataset(
            token=token,
            dataset="trading_calendar",
            start_date=calendar_start,
            end_date=today,
            ts_code=None,
            trade_date=None,
        )
    except ModuleNotFoundError:
        return SyncLatestResult(False, "missing dependency: install tushare to fetch provider data")
    except Exception as error:
        return SyncLatestResult(False, f"provider calendar fetch failed: {error}")

    provider_open_dates = _provider_calendar_open_dates(provider_calendar)
    if not provider_open_dates:
        return SyncLatestResult(False, f"provider calendar returned no trading days through {today}")
    latest_trade_date = max(provider_open_dates)

    local_calendar_dates = _local_trading_dates(config.current_curated_root / "trading_calendar" / "part-000.parquet")
    all_calendar_dates = sorted(set(local_calendar_dates) | set(provider_open_dates))
    selected_start = start_date or _default_sync_start(config.current_curated_root, selected_datasets, all_calendar_dates, latest_trade_date)
    trading_window = [value for value in all_calendar_dates if selected_start <= value <= latest_trade_date]
    if not trading_window:
        return SyncLatestResult(True, f"latest_trade_date: {latest_trade_date}\nno trading dates to inspect from {selected_start}")

    missing_by_dataset = {
        dataset: _missing_dates_for_dataset(config.current_curated_root, dataset, trading_window)
        for dataset in selected_datasets
    }
    lines = [
        f"latest_trade_date: {latest_trade_date}",
        f"inspect_window: {trading_window[0]}..{trading_window[-1]} trading_days={len(trading_window)}",
        "missing_dates:",
        *[
            f"- {dataset}: {len(missing)}" + (f" [{', '.join(missing)}]" if missing else "")
            for dataset, missing in missing_by_dataset.items()
        ],
    ]

    if dry_run:
        return SyncLatestResult(True, "\n".join(lines + ["dry_run: no data fetched or promoted"]))

    calendar_batch = write_raw_batch(
        config_path=config_path,
        source=source,
        dataset="trading_calendar",
        frame=provider_calendar,
        as_of_date=today,
    )
    if not calendar_batch.ok:
        return SyncLatestResult(False, "\n".join(lines + [calendar_batch.message]))
    calendar_promote = promote_raw_batch(
        config_path=config_path,
        source=source,
        dataset="trading_calendar",
        batch_id=calendar_batch.batch_id,
    )
    if not calendar_promote.ok:
        return SyncLatestResult(False, "\n".join(lines + [calendar_promote.message]))
    lines.append(calendar_promote.message)

    market_datasets = [dataset for dataset in selected_datasets if dataset in MARKET_SYNC_DATASETS and missing_by_dataset[dataset]]
    if market_datasets:
        market_missing = sorted({value for dataset in market_datasets for value in missing_by_dataset[dataset]})
        market_run_id = f"{run_id_prefix}_market_daily_{latest_trade_date.replace('-', '')}"
        market_result = run_market_daily(
            config_path=config_path,
            source=source,
            run_id=market_run_id,
            datasets=market_datasets,
            start_date=market_missing[0],
            end_date=market_missing[-1],
            as_of_date=latest_trade_date,
            max_tasks=max_market_tasks,
            requests_per_minute=requests_per_minute,
            retry=retry,
            retry_wait_seconds=retry_wait_seconds,
            backoff_multiplier=backoff_multiplier,
            symbol_batch_size=symbol_batch_size,
            token_env=token_env,
            progress_every_tasks=progress_every_tasks,
            progress_callback=progress_callback,
        )
        lines.append(market_result.message)
        if not market_result.ok:
            return SyncLatestResult(False, "\n".join(lines))
        market_promote = promote_raw_run(config_path=config_path, run_id=market_run_id)
        lines.append(market_promote.message)
        if not market_promote.ok:
            return SyncLatestResult(False, "\n".join(lines))

    if "cyq_perf" in selected_datasets and missing_by_dataset.get("cyq_perf"):
        cyq_missing = missing_by_dataset["cyq_perf"]
        cyq_run_id = f"{run_id_prefix}_cyq_perf_{latest_trade_date.replace('-', '')}"
        cyq_result = run_cyq_perf_batches(
            config_path=config_path,
            source=source,
            run_id=cyq_run_id,
            start_date=cyq_missing[0],
            end_date=cyq_missing[-1],
            as_of_date=latest_trade_date,
            batch_size=cyq_batch_size,
            max_batches=max_cyq_batches,
            requests_per_minute=cyq_requests_per_minute,
            token_env=token_env,
            progress_every_batches=progress_every_batches,
            progress_callback=progress_callback,
            retry=retry,
            retry_wait_seconds=retry_wait_seconds,
            backoff_multiplier=backoff_multiplier,
        )
        lines.append(cyq_result.message)
        if not cyq_result.ok:
            return SyncLatestResult(False, "\n".join(lines))
        cyq_promote = promote_raw_run(config_path=config_path, run_id=cyq_run_id, dataset="cyq_perf")
        lines.append(cyq_promote.message)
        if not cyq_promote.ok:
            return SyncLatestResult(False, "\n".join(lines))

    quality = check_curated_quality(config_path)
    lines.append(quality.message)
    if not quality.ok:
        return SyncLatestResult(False, "\n".join(lines))

    if create_snapshot_after:
        snapshot = create_snapshot(
            config_path=config_path,
            as_of_date=latest_trade_date,
            config_version=f"{run_id_prefix}_{latest_trade_date.replace('-', '')}",
        )
        lines.append(snapshot.message)
        if not snapshot.ok:
            return SyncLatestResult(False, "\n".join(lines))

    return SyncLatestResult(True, "\n".join(lines))


def _calendar_fetch_start(calendar_path: Path, end_date: str, lookback_days: int) -> str:
    fallback = date.fromisoformat(end_date) - timedelta(days=lookback_days)
    local_dates = _local_trading_dates(calendar_path)
    if not local_dates:
        return fallback.isoformat()
    next_day = date.fromisoformat(max(local_dates)) + timedelta(days=1)
    return min(next_day, fallback).isoformat()


def _load_token(token_env: str, env_path: Path) -> str | None:
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


def _provider_calendar_open_dates(frame: pl.DataFrame) -> list[str]:
    if frame.is_empty() or "cal_date" not in frame.columns or "is_open" not in frame.columns:
        return []
    opened = (
        frame.with_columns(
            [
                pl.col("cal_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("trade_date"),
                pl.col("is_open").cast(pl.Int64, strict=False).alias("is_open"),
            ]
        )
        .filter(pl.col("is_open") == 1)
        .select("trade_date")
        .drop_nulls()
        .unique()
        .sort("trade_date")
    )
    return [value.isoformat() for value in opened.to_series().to_list()]


def _local_trading_dates(calendar_path: Path) -> list[str]:
    if not calendar_path.exists():
        return []
    frame = pl.read_parquet(calendar_path)
    if "trade_date" not in frame.columns or "is_trading_day" not in frame.columns:
        return []
    opened = (
        frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
        .filter(pl.col("is_trading_day") == True)
        .select("trade_date")
        .drop_nulls()
        .unique()
        .sort("trade_date")
    )
    return [value.isoformat() for value in opened.to_series().to_list()]


def _default_sync_start(current_root: Path, datasets: list[str], calendar_dates: list[str], latest_trade_date: str) -> str:
    latest_values = [_latest_covered_date(current_root, dataset) for dataset in datasets]
    latest_values = [value for value in latest_values if value is not None]
    if latest_values:
        latest_common = min(latest_values)
        later_dates = [value for value in calendar_dates if latest_common < value <= latest_trade_date]
        return later_dates[0] if later_dates else latest_trade_date
    fallback_dates = [value for value in calendar_dates if value <= latest_trade_date]
    return fallback_dates[0] if fallback_dates else latest_trade_date


def _latest_covered_date(current_root: Path, dataset: str) -> str | None:
    dates = _covered_dates_for_dataset(current_root, dataset)
    return max(dates) if dates else None


def _missing_dates_for_dataset(current_root: Path, dataset: str, trading_window: list[str]) -> list[str]:
    covered = _covered_dates_for_dataset(current_root, dataset)
    return [value for value in trading_window if value not in covered]


def _covered_dates_for_dataset(current_root: Path, dataset: str) -> set[str]:
    if dataset == "daily_prices":
        return _dates_from_parquet(current_root / "daily_prices" / "part-000.parquet")
    if dataset == "moneyflow_dc":
        return _dates_from_capital_flow(current_root / "capital_flow_or_chip" / "part-000.parquet", "moneyflow_dc")
    if dataset == "cyq_perf":
        return _dates_from_capital_flow(current_root / "capital_flow_or_chip" / "part-000.parquet", "cyq_perf")
    return set()


def _dates_from_parquet(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns:
        return set()
    dates = (
        frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
        .select("trade_date")
        .drop_nulls()
        .unique()
    )
    return {value.isoformat() for value in dates.to_series().to_list()}


def _dates_from_capital_flow(path: Path, source_dataset: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns:
        return set()
    frame = frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
    if source_dataset == "moneyflow_dc":
        if "main_net_inflow_rate" not in frame.columns and "main_net_inflow" not in frame.columns:
            return set()
        filters = []
        if "main_net_inflow_rate" in frame.columns:
            filters.append(pl.col("main_net_inflow_rate").is_not_null())
        if "main_net_inflow" in frame.columns:
            filters.append(pl.col("main_net_inflow").is_not_null())
        predicate = filters[0]
        for item in filters[1:]:
            predicate = predicate | item
        frame = frame.filter(predicate)
    elif source_dataset == "cyq_perf":
        if "data_method" not in frame.columns:
            return set()
        frame = frame.filter(pl.col("data_method").cast(pl.Utf8).str.starts_with("tushare_cyq_perf").fill_null(False))
    dates = frame.select("trade_date").drop_nulls().unique()
    return {value.isoformat() for value in dates.to_series().to_list()}
