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


DEFAULT_BENCHMARK_SYMBOLS = ["000852.SH", "000300.SH", "000905.SH", "399006.SZ", "000688.SH", "899050.BJ"]
DEFAULT_SYNC_DATASETS = [
    "daily_prices",
    "moneyflow_dc",
    "cyq_perf",
    "adj_factor",
    "daily_basic",
    "stk_limit",
    "suspend_d",
    "index_daily",
    "index_classify",
    "sw_daily",
]
MARKET_SYNC_DATASETS = {"daily_prices", "moneyflow_dc"}
SUPPLEMENTAL_SYNC_DATASETS = {"adj_factor", "daily_basic", "stk_limit", "suspend_d"}
INDUSTRY_SYNC_DATASETS = {"index_classify", "sw_daily"}
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
    benchmark_symbols: list[str] | None = None,
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
    token = os.environ.get(token_env)
    if not token:
        return SyncLatestResult(False, f"missing required environment variable: {token_env}")
    selected_benchmark_symbols = benchmark_symbols or DEFAULT_BENCHMARK_SYMBOLS
    initialize_metadata_catalog(config.metadata_sqlite_path)
    today = end_date or date.today().isoformat()
    calendar_start = _calendar_fetch_start(config.current_curated_root / "trading_calendar" / "part-000.parquet", today, calendar_lookback_days)
    _progress(progress_callback, f"sync_latest: fetching trading calendar {calendar_start}..{today}")

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
    _progress(progress_callback, f"sync_latest: provider latest trading day {latest_trade_date}")

    local_calendar_dates = _local_trading_dates(config.current_curated_root / "trading_calendar" / "part-000.parquet")
    all_calendar_dates = sorted(set(local_calendar_dates) | set(provider_open_dates))
    selected_start = start_date or _default_sync_start(config.current_curated_root, selected_datasets, all_calendar_dates, latest_trade_date)
    trading_window = [value for value in all_calendar_dates if selected_start <= value <= latest_trade_date]
    if not trading_window:
        return SyncLatestResult(True, f"latest_trade_date: {latest_trade_date}\nno trading dates to inspect from {selected_start}")

    missing_by_dataset = {
        dataset: _missing_dates_for_dataset(config.current_curated_root, dataset, trading_window, selected_benchmark_symbols)
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

    _progress(progress_callback, "sync_latest: writing and promoting trading calendar")
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
        _progress(progress_callback, f"sync_latest: running market daily {market_missing[0]}..{market_missing[-1]} datasets={','.join(market_datasets)}")
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
        _progress(progress_callback, f"sync_latest: promoting market daily run {market_run_id}")
        market_promote = promote_raw_run(config_path=config_path, run_id=market_run_id)
        lines.append(market_promote.message)
        if not market_promote.ok:
            return SyncLatestResult(False, "\n".join(lines))

    if "cyq_perf" in selected_datasets and missing_by_dataset.get("cyq_perf"):
        cyq_missing = missing_by_dataset["cyq_perf"]
        cyq_run_id = f"{run_id_prefix}_cyq_perf_{latest_trade_date.replace('-', '')}"
        _progress(progress_callback, f"sync_latest: running cyq_perf {cyq_missing[0]}..{cyq_missing[-1]}")
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
        _progress(progress_callback, f"sync_latest: promoting cyq_perf run {cyq_run_id}")
        cyq_promote = promote_raw_run(config_path=config_path, run_id=cyq_run_id, dataset="cyq_perf")
        lines.append(cyq_promote.message)
        if not cyq_promote.ok:
            return SyncLatestResult(False, "\n".join(lines))

    for dataset in [value for value in selected_datasets if value in SUPPLEMENTAL_SYNC_DATASETS and missing_by_dataset[value]]:
        _progress(progress_callback, f"sync_latest: syncing {dataset} dates={len(missing_by_dataset[dataset])}")
        result = _sync_range_dataset(
            config_path=config_path,
            source=source,
            token=token,
            dataset=dataset,
            missing_dates=missing_by_dataset[dataset],
            as_of_date=latest_trade_date,
        )
        lines.append(result.message)
        if not result.ok:
            return SyncLatestResult(False, "\n".join(lines))

    for dataset in [value for value in selected_datasets if value in INDUSTRY_SYNC_DATASETS and missing_by_dataset[value]]:
        _progress(progress_callback, f"sync_latest: syncing {dataset} dates={len(missing_by_dataset[dataset])}")
        result = _sync_range_dataset(
            config_path=config_path,
            source=source,
            token=token,
            dataset=dataset,
            missing_dates=missing_by_dataset[dataset],
            as_of_date=latest_trade_date,
        )
        lines.append(result.message)
        if not result.ok:
            return SyncLatestResult(False, "\n".join(lines))

    if "index_daily" in selected_datasets and missing_by_dataset.get("index_daily"):
        for symbol in selected_benchmark_symbols:
            _progress(progress_callback, f"sync_latest: syncing index_daily {symbol} dates={len(missing_by_dataset['index_daily'])}")
            result = _sync_range_dataset(
                config_path=config_path,
                source=source,
                token=token,
                dataset="index_daily",
                missing_dates=missing_by_dataset["index_daily"],
                as_of_date=latest_trade_date,
                ts_code=symbol,
            )
            lines.append(result.message)
            if not result.ok:
                return SyncLatestResult(False, "\n".join(lines))

    _progress(progress_callback, "sync_latest: running quality check")
    quality = check_curated_quality(config_path)
    lines.append(quality.message)
    if not quality.ok:
        return SyncLatestResult(False, "\n".join(lines))

    if create_snapshot_after:
        _progress(progress_callback, f"sync_latest: creating snapshot for {latest_trade_date}")
        snapshot = create_snapshot(
            config_path=config_path,
            as_of_date=latest_trade_date,
            config_version=f"{run_id_prefix}_{latest_trade_date.replace('-', '')}",
        )
        lines.append(snapshot.message)
        if not snapshot.ok:
            return SyncLatestResult(False, "\n".join(lines))

    return SyncLatestResult(True, "\n".join(lines))


def _progress(callback, message: str) -> None:
    if callback:
        callback(message)


def _calendar_fetch_start(calendar_path: Path, end_date: str, lookback_days: int) -> str:
    fallback = date.fromisoformat(end_date) - timedelta(days=lookback_days)
    local_dates = _local_trading_dates(calendar_path)
    if not local_dates:
        return fallback.isoformat()
    next_day = date.fromisoformat(max(local_dates)) + timedelta(days=1)
    return min(next_day, fallback).isoformat()


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
    dates = _covered_dates_for_dataset(current_root, dataset, DEFAULT_BENCHMARK_SYMBOLS)
    return max(dates) if dates else None


def _missing_dates_for_dataset(current_root: Path, dataset: str, trading_window: list[str], benchmark_symbols: list[str] | None = None) -> list[str]:
    if dataset == "index_classify":
        path = current_root / "industry_classification" / "part-000.parquet"
        return [] if _parquet_has_rows(path) else trading_window[-1:]
    covered = _covered_dates_for_dataset(current_root, dataset, benchmark_symbols or DEFAULT_BENCHMARK_SYMBOLS)
    return [value for value in trading_window if value not in covered]


def _covered_dates_for_dataset(current_root: Path, dataset: str, benchmark_symbols: list[str]) -> set[str]:
    if dataset == "daily_prices":
        return _dates_from_daily_prices(current_root / "daily_prices" / "part-000.parquet", asset_type="stock")
    if dataset == "index_daily":
        return _dates_from_index_daily(current_root / "daily_prices" / "part-000.parquet", benchmark_symbols)
    if dataset == "adj_factor":
        return _dates_from_daily_price_field(current_root / "daily_prices" / "part-000.parquet", "adj_factor")
    if dataset == "daily_basic":
        return _dates_from_daily_price_field(current_root / "daily_prices" / "part-000.parquet", "turnover_rate")
    if dataset == "stk_limit":
        return _dates_from_daily_price_field(current_root / "daily_prices" / "part-000.parquet", "limit_up")
    if dataset == "suspend_d":
        return _dates_from_daily_price_field(current_root / "daily_prices" / "part-000.parquet", "is_suspended")
    if dataset == "moneyflow_dc":
        return _dates_from_capital_flow(current_root / "capital_flow_or_chip" / "part-000.parquet", "moneyflow_dc")
    if dataset == "cyq_perf":
        return _dates_from_capital_flow(current_root / "capital_flow_or_chip" / "part-000.parquet", "cyq_perf")
    if dataset == "sw_daily":
        return _dates_from_parquet(current_root / "industry_daily" / "part-000.parquet")
    return set()


def _sync_range_dataset(
    config_path: Path,
    source: str,
    token: str,
    dataset: str,
    missing_dates: list[str],
    as_of_date: str,
    ts_code: str | None = None,
) -> SyncLatestResult:
    if not missing_dates:
        return SyncLatestResult(True, f"{dataset}: no missing dates")
    start_date = missing_dates[0]
    end_date = missing_dates[-1]
    label = f"{dataset}" + (f"/{ts_code}" if ts_code else "")
    try:
        frame = _fetch_tushare_dataset(
            token=token,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            ts_code=ts_code,
            trade_date=None,
        )
    except ModuleNotFoundError:
        return SyncLatestResult(False, f"{label}: missing dependency: install tushare to fetch provider data")
    except Exception as error:
        return SyncLatestResult(False, f"{label}: provider fetch failed: {error}")
    batch = write_raw_batch(
        config_path=config_path,
        source=source,
        dataset=dataset,
        frame=frame,
        as_of_date=as_of_date,
    )
    if not batch.ok:
        return SyncLatestResult(False, f"{label}: {batch.message}")
    promote = promote_raw_batch(
        config_path=config_path,
        source=source,
        dataset=dataset,
        batch_id=batch.batch_id,
    )
    if not promote.ok:
        return SyncLatestResult(False, f"{label}: {promote.message}")
    return SyncLatestResult(True, f"{label}: fetched {start_date}..{end_date} rows={batch.row_count}; {promote.message}")


def _dates_from_daily_prices(path: Path, asset_type: str) -> set[str]:
    if not path.exists():
        return set()
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns:
        return set()
    frame = frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
    if "asset_type" in frame.columns:
        frame = frame.filter(pl.col("asset_type").fill_null("stock") == asset_type)
    return {value.isoformat() for value in frame.select("trade_date").drop_nulls().unique().to_series().to_list()}


def _dates_from_daily_price_field(path: Path, field: str, allow_false: bool = False) -> set[str]:
    if not path.exists():
        return set()
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns or field not in frame.columns:
        return set()
    frame = frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
    if "asset_type" in frame.columns:
        frame = frame.filter(pl.col("asset_type").fill_null("stock") == "stock")
    predicate = pl.col(field).is_not_null()
    if allow_false:
        predicate = predicate & (pl.col(field).cast(pl.Boolean, strict=False) == False)
    dates = frame.filter(predicate).select("trade_date").drop_nulls().unique()
    return {value.isoformat() for value in dates.to_series().to_list()}


def _dates_from_index_daily(path: Path, benchmark_symbols: list[str]) -> set[str]:
    if not path.exists():
        return set()
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns or "symbol" not in frame.columns:
        return set()
    frame = frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
    if "asset_type" in frame.columns:
        frame = frame.filter(pl.col("asset_type") == "index")
    if benchmark_symbols:
        frame = frame.filter(pl.col("symbol").is_in(benchmark_symbols))
    counts = frame.group_by("trade_date").agg(pl.col("symbol").n_unique().alias("symbol_count"))
    required = len(set(benchmark_symbols))
    dates = counts.filter(pl.col("symbol_count") >= required).select("trade_date")
    return {value.isoformat() for value in dates.to_series().to_list()}


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


def _parquet_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return not pl.read_parquet(path).is_empty()
    except Exception:
        return False


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
