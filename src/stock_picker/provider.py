"""Provider raw data fetchers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class ProviderFetchResult:
    ok: bool
    message: str
    batch_id: str | None = None
    raw_path: Path | None = None
    row_count: int = 0


@dataclass(frozen=True)
class ProviderProbeResult:
    ok: bool
    message: str
    row_count: int = 0


TUSHARE_SUPPORTED_DATASETS = {
    "security_master",
    "trading_calendar",
    "daily_prices",
    "moneyflow_dc",
    "cyq_perf",
}

MARKET_DAILY_DATASETS = {"daily_prices", "moneyflow_dc"}


TUSHARE_EXPECTED_FIELDS = {
    "stock_basic": {"ts_code", "symbol", "name", "market", "list_date"},
    "trade_cal": {"exchange", "cal_date", "is_open"},
    "daily": {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"},
    "moneyflow_dc": {"ts_code", "trade_date", "net_amount", "net_amount_rate"},
    "moneyflow_ths": {"ts_code", "trade_date", "net_d5_amount"},
    "cyq_perf": {"ts_code", "trade_date", "winner_rate"},
}


def fetch_provider_raw(
    config_path: Path,
    source: str,
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    ts_code: str | None = None,
    trade_date: str | None = None,
    token_env: str = "TUSHARE_TOKEN",
) -> ProviderFetchResult:
    if source != "tushare":
        return ProviderFetchResult(False, f"unsupported provider source: {source}")
    if dataset not in TUSHARE_SUPPORTED_DATASETS:
        return ProviderFetchResult(False, f"unsupported tushare dataset: {dataset}")
    if dataset == "cyq_perf" and not ts_code:
        return ProviderFetchResult(False, "cyq_perf fetch requires --ts-code")

    token = os.environ.get(token_env)
    if not token:
        return ProviderFetchResult(False, f"missing required environment variable: {token_env}")

    try:
        frame = _fetch_tushare_dataset(token, dataset, start_date, end_date, ts_code, trade_date)
    except ModuleNotFoundError:
        return ProviderFetchResult(False, "missing dependency: install tushare to fetch provider data")
    except Exception as error:
        return ProviderFetchResult(False, f"provider fetch failed: {error}")

    return write_raw_batch(
        config_path=config_path,
        source=source,
        dataset=dataset,
        frame=frame,
        as_of_date=as_of_date or trade_date or end_date or start_date or datetime.now(UTC).date().isoformat(),
    )


def fetch_cyq_perf_batch(
    config_path: Path,
    source: str = "tushare",
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    symbols: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    delay_seconds: float = 0.0,
    token_env: str = "TUSHARE_TOKEN",
) -> ProviderFetchResult:
    if source != "tushare":
        return ProviderFetchResult(False, f"unsupported provider source: {source}")
    if limit is not None and limit <= 0:
        return ProviderFetchResult(False, "cyq_perf batch fetch requires a positive --limit")
    if offset < 0:
        return ProviderFetchResult(False, "cyq_perf batch fetch requires a non-negative --offset")
    if delay_seconds < 0:
        return ProviderFetchResult(False, "cyq_perf batch fetch requires a non-negative --delay-seconds")

    token = os.environ.get(token_env)
    if not token:
        return ProviderFetchResult(False, f"missing required environment variable: {token_env}")

    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    try:
        selected_symbols = symbols or _load_security_master_symbols(config.current_curated_root / "security_master" / "part-000.parquet")
    except ValueError as error:
        return ProviderFetchResult(False, str(error))
    selected_symbols = selected_symbols[offset:]
    if limit is not None:
        selected_symbols = selected_symbols[:limit]
    if not selected_symbols:
        return ProviderFetchResult(False, "cyq_perf batch fetch found no symbols to fetch")

    frames: list[pl.DataFrame] = []
    failures: list[str] = []
    try:
        import tushare as ts  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return ProviderFetchResult(False, "missing dependency: install tushare to fetch provider data")

    pro = ts.pro_api(token)
    for index, symbol in enumerate(selected_symbols):
        try:
            frame = _fetch_tushare_cyq_perf_for_symbol(pro, symbol, start_date, end_date)
            if not frame.is_empty():
                frames.append(frame)
        except Exception as error:
            failures.append(f"{symbol}: {error}")
        if delay_seconds and index < len(selected_symbols) - 1:
            time.sleep(delay_seconds)

    if not frames:
        failure_note = "; ".join(failures[:5])
        suffix = f"; failures: {failure_note}" if failure_note else ""
        return ProviderFetchResult(False, f"cyq_perf batch fetch returned no rows{suffix}")

    combined = pl.concat(frames, how="diagonal_relaxed")
    result = write_raw_batch(
        config_path=config_path,
        source=source,
        dataset="cyq_perf",
        frame=combined,
        as_of_date=as_of_date or end_date or start_date or datetime.now(UTC).date().isoformat(),
    )
    if not result.ok:
        return result
    failure_summary = f" failures={len(failures)}" if failures else " failures=0"
    return ProviderFetchResult(
        True,
        f"{result.message} symbols_requested={len(selected_symbols)} symbols_with_rows={len(frames)}{failure_summary}",
        result.batch_id,
        result.raw_path,
        result.row_count,
    )


def run_cyq_perf_batches(
    config_path: Path,
    source: str = "tushare",
    run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    batch_size: int = 100,
    max_batches: int = 1,
    delay_seconds: float = 0.0,
    token_env: str = "TUSHARE_TOKEN",
) -> ProviderFetchResult:
    if source != "tushare":
        return ProviderFetchResult(False, f"unsupported provider source: {source}")
    if batch_size <= 0:
        return ProviderFetchResult(False, "cyq_perf run requires a positive --batch-size")
    if max_batches <= 0:
        return ProviderFetchResult(False, "cyq_perf run requires a positive --max-batches")
    if delay_seconds < 0:
        return ProviderFetchResult(False, "cyq_perf run requires a non-negative --delay-seconds")

    token = os.environ.get(token_env)
    if not token:
        return ProviderFetchResult(False, f"missing required environment variable: {token_env}")

    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    try:
        all_symbols = _load_security_master_symbols(config.current_curated_root / "security_master" / "part-000.parquet")
    except ValueError as error:
        return ProviderFetchResult(False, str(error))
    if not all_symbols:
        return ProviderFetchResult(False, "cyq_perf run found no symbols to fetch")

    run = _load_or_create_provider_run(
        config.metadata_sqlite_path,
        source=source,
        dataset="cyq_perf",
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date or end_date or start_date or datetime.now(UTC).date().isoformat(),
        total_symbols=len(all_symbols),
        batch_size=batch_size,
    )
    if run["status"] == "completed":
        return ProviderFetchResult(True, f"cyq_perf run already completed: {run['run_id']} next_offset={run['next_offset']}")

    batch_results: list[ProviderFetchResult] = []
    for _ in range(max_batches):
        next_offset = int(run["next_offset"])
        if next_offset >= len(all_symbols):
            _mark_provider_run_completed(config.metadata_sqlite_path, str(run["run_id"]))
            break
        result = fetch_cyq_perf_batch(
            config_path=config_path,
            source=source,
            start_date=str(run["start_date"] or "") or start_date,
            end_date=str(run["end_date"] or "") or end_date,
            as_of_date=str(run["as_of_date"]),
            symbols=all_symbols,
            limit=int(run["batch_size"]),
            offset=next_offset,
            delay_seconds=delay_seconds,
            token_env=token_env,
        )
        if not result.ok:
            _update_provider_run_failure(config.metadata_sqlite_path, str(run["run_id"]), result.message)
            return result
        batch_results.append(result)
        symbols_requested = min(int(run["batch_size"]), len(all_symbols) - next_offset)
        _advance_provider_run(
            config.metadata_sqlite_path,
            run_id=str(run["run_id"]),
            next_offset=next_offset + symbols_requested,
            symbols_requested=symbols_requested,
            symbols_with_rows=_message_int(result.message, "symbols_with_rows"),
            failures=_message_int(result.message, "failures"),
            row_count=result.row_count,
            batch_id=str(result.batch_id),
        )
        run = _load_provider_run(config.metadata_sqlite_path, str(run["run_id"]))
        if int(run["next_offset"]) >= len(all_symbols):
            _mark_provider_run_completed(config.metadata_sqlite_path, str(run["run_id"]))
            run = _load_provider_run(config.metadata_sqlite_path, str(run["run_id"]))
            break

    last = batch_results[-1] if batch_results else None
    message = (
        f"cyq_perf run: {run['run_id']} status={run['status']} "
        f"next_offset={run['next_offset']}/{run['total_symbols']} "
        f"batches={len(batch_results)} rows={run['row_count']} "
        f"raw_batches={', '.join(json.loads(str(run['raw_batch_ids'])))}"
    )
    return ProviderFetchResult(True, message, last.batch_id if last else None, last.raw_path if last else None, int(run["row_count"]))


def run_market_daily(
    config_path: Path,
    source: str = "tushare",
    run_id: str | None = None,
    datasets: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    as_of_date: str | None = None,
    max_tasks: int = 1,
    requests_per_minute: float = 40.0,
    retry: int = 3,
    retry_wait_seconds: float = 60.0,
    backoff_multiplier: float = 2.0,
    symbol_batch_size: int = 1000,
    token_env: str = "TUSHARE_TOKEN",
) -> ProviderFetchResult:
    if source != "tushare":
        return ProviderFetchResult(False, f"unsupported provider source: {source}")
    selected_datasets = datasets or ["daily_prices", "moneyflow_dc"]
    unsupported = sorted(set(selected_datasets) - MARKET_DAILY_DATASETS)
    if unsupported:
        return ProviderFetchResult(False, "unsupported market daily datasets: " + ", ".join(unsupported))
    if not start_date or not end_date:
        return ProviderFetchResult(False, "market daily run requires --start-date and --end-date")
    if max_tasks <= 0:
        return ProviderFetchResult(False, "market daily run requires a positive --max-tasks")
    if requests_per_minute < 0:
        return ProviderFetchResult(False, "market daily run requires a non-negative --requests-per-minute")
    if retry < 0:
        return ProviderFetchResult(False, "market daily run requires a non-negative --retry")
    if retry_wait_seconds < 0:
        return ProviderFetchResult(False, "market daily run requires a non-negative --retry-wait-seconds")
    if backoff_multiplier < 1:
        return ProviderFetchResult(False, "market daily run requires --backoff-multiplier >= 1")
    if symbol_batch_size <= 0:
        return ProviderFetchResult(False, "market daily run requires a positive --symbol-batch-size")

    token = os.environ.get(token_env)
    if not token:
        return ProviderFetchResult(False, f"missing required environment variable: {token_env}")

    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    try:
        trade_dates = _load_trading_dates(config.current_curated_root / "trading_calendar" / "part-000.parquet", start_date, end_date)
    except ValueError as error:
        return ProviderFetchResult(False, str(error))
    if not trade_dates:
        return ProviderFetchResult(False, "market daily run found no trading dates")
    symbols: list[str] = []
    if "moneyflow_dc" in selected_datasets:
        try:
            symbols = _load_security_master_symbols(config.current_curated_root / "security_master" / "part-000.parquet")
        except ValueError as error:
            return ProviderFetchResult(False, str(error))
        if not symbols:
            return ProviderFetchResult(False, "market daily run found no active symbols for moneyflow_dc")

    selected_run_id = run_id or f"{source}_market_daily_{_clean_date(start_date)}_{_clean_date(end_date)}"
    task_count = _market_daily_task_count(selected_datasets, trade_dates, symbols, symbol_batch_size)
    _load_or_create_market_daily_run(
        config.metadata_sqlite_path,
        run_id=selected_run_id,
        source=source,
        datasets=selected_datasets,
        start_date=start_date,
        end_date=end_date,
        as_of_date=as_of_date or end_date,
        task_count=task_count,
    )
    _ensure_market_daily_tasks(
        config.metadata_sqlite_path,
        run_id=selected_run_id,
        source=source,
        datasets=selected_datasets,
        trade_dates=trade_dates,
        symbols=symbols,
        symbol_batch_size=symbol_batch_size,
    )

    completed: list[ProviderFetchResult] = []
    last_request_at: float | None = None
    for _ in range(max_tasks):
        task = _next_market_daily_task(config.metadata_sqlite_path, selected_run_id)
        if task is None:
            _mark_provider_run_completed(config.metadata_sqlite_path, selected_run_id)
            break
        result, last_request_at = _run_market_daily_task(
            config_path=config_path,
            sqlite_path=config.metadata_sqlite_path,
            task=task,
            token=token,
            symbols=symbols,
            as_of_date=as_of_date or end_date,
            retry=retry,
            retry_wait_seconds=retry_wait_seconds,
            backoff_multiplier=backoff_multiplier,
            requests_per_minute=requests_per_minute,
            last_request_at=last_request_at,
        )
        if not result.ok:
            _update_provider_run_failure(config.metadata_sqlite_path, selected_run_id, result.message)
            return result
        completed.append(result)

    stats = _market_daily_run_stats(config.metadata_sqlite_path, selected_run_id)
    if stats["remaining"] == 0:
        _mark_provider_run_completed(config.metadata_sqlite_path, selected_run_id)
        status = "completed"
    else:
        status = "running"
    _update_market_daily_run_summary(config.metadata_sqlite_path, selected_run_id, status=status)
    batch_ids = [str(result.batch_id) for result in completed if result.batch_id]
    message = (
        f"market daily run: {selected_run_id} status={status} "
        f"tasks_success={stats['success']}/{stats['total']} "
        f"tasks_failed={stats['failed']} tasks_remaining={stats['remaining']} "
        f"rows={stats['rows']} raw_batches={', '.join(batch_ids)}"
    )
    last = completed[-1] if completed else None
    return ProviderFetchResult(True, message, last.batch_id if last else None, last.raw_path if last else None, int(stats["rows"]))


def probe_provider_api(
    source: str,
    api: str,
    ts_code: str | None = None,
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    token_env: str = "TUSHARE_TOKEN",
) -> ProviderProbeResult:
    if source != "tushare":
        return ProviderProbeResult(False, f"unsupported provider source: {source}")
    if api not in TUSHARE_EXPECTED_FIELDS:
        return ProviderProbeResult(False, f"unsupported tushare api: {api}")

    token = os.environ.get(token_env)
    if not token:
        return ProviderProbeResult(False, f"missing required environment variable: {token_env}")

    try:
        frame = _probe_tushare_api(token, api, ts_code, trade_date, start_date, end_date)
    except ModuleNotFoundError:
        return ProviderProbeResult(False, "missing dependency: install tushare to probe provider data")
    except Exception as error:
        return ProviderProbeResult(False, f"provider probe failed: {error}")

    expected = TUSHARE_EXPECTED_FIELDS[api]
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    columns = ", ".join(frame.columns)
    if missing:
        return ProviderProbeResult(
            False,
            "\n".join(
                [
                    f"provider probe failed: {api}",
                    f"row_count: {frame.height}",
                    f"columns: {columns}",
                    "missing_expected_fields: " + ", ".join(missing),
                ]
            ),
            frame.height,
        )
    return ProviderProbeResult(
        True,
        "\n".join(
            [
                f"provider probe succeeded: {api}",
                f"row_count: {frame.height}",
                f"columns: {columns}",
                "missing_expected_fields: none",
            ]
        ),
        frame.height,
    )


def write_raw_batch(
    config_path: Path,
    source: str,
    dataset: str,
    frame: pl.DataFrame,
    as_of_date: str,
) -> ProviderFetchResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    batch_id = _next_batch_id(config.metadata_sqlite_path, source, dataset, as_of_date)
    raw_dir = (
        config.raw_root
        / f"source={source}"
        / f"dataset={dataset}"
        / f"as_of_date={as_of_date}"
        / f"batch_id={batch_id}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "part.csv"
    frame.write_csv(raw_path)
    row_count = frame.height
    checksum = _sha256_file(raw_path)
    schema_hash = hashlib.sha256(",".join(frame.columns).encode("utf-8")).hexdigest()

    with sqlite3.connect(config.metadata_sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO data_batches (
              batch_id,
              source,
              dataset_name,
              retrieved_at,
              business_date,
              raw_path,
              format,
              row_count,
              schema_hash,
              content_checksum,
              status,
              notes
            )
            VALUES (?, ?, ?, ?, ?, ?, 'csv', ?, ?, ?, 'success', ?)
            """,
            (
                batch_id,
                source,
                dataset,
                created_at,
                as_of_date,
                str(raw_path),
                row_count,
                schema_hash,
                checksum,
                "provider raw fetch",
            ),
        )

    return ProviderFetchResult(
        True,
        f"fetched raw batch: {batch_id} rows={row_count}",
        batch_id,
        raw_path,
        row_count,
    )


def _fetch_tushare_dataset(
    token: str,
    dataset: str,
    start_date: str | None,
    end_date: str | None,
    ts_code: str | None,
    trade_date: str | None,
) -> pl.DataFrame:
    import tushare as ts  # type: ignore[import-not-found]

    pro = ts.pro_api(token)
    if dataset == "security_master":
        pandas_frame = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date",
        )
    elif dataset == "trading_calendar":
        pandas_frame = pro.trade_cal(
            exchange="SSE",
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    elif dataset == "daily_prices":
        pandas_frame = pro.daily(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    elif dataset == "moneyflow_dc":
        pandas_frame = pro.moneyflow_dc(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    else:
        pandas_frame = pro.cyq_perf(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    return pl.from_pandas(pandas_frame)


def _fetch_tushare_cyq_perf_for_symbol(
    pro: object,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
) -> pl.DataFrame:
    pandas_frame = pro.cyq_perf(
        ts_code=symbol,
        start_date=_compact_date(start_date),
        end_date=_compact_date(end_date),
    )
    return pl.from_pandas(pandas_frame)


def _load_trading_dates(path: Path, start_date: str, end_date: str) -> list[str]:
    if not path.exists():
        raise ValueError(f"trading_calendar current Parquet not found: {path}")
    frame = pl.read_parquet(path)
    if "trade_date" not in frame.columns:
        raise ValueError("trading_calendar current Parquet is missing column: trade_date")
    if "is_trading_day" not in frame.columns:
        raise ValueError("trading_calendar current Parquet is missing column: is_trading_day")
    start = _parse_iso_date(start_date)
    end = _parse_iso_date(end_date)
    if start > end:
        raise ValueError("market daily run requires start date <= end date")
    filtered = (
        frame.with_columns(pl.col("trade_date").cast(pl.Date).alias("trade_date"))
        .filter((pl.col("is_trading_day") == True) & (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        .select("trade_date")
        .unique()
        .sort("trade_date")
    )
    return [value.isoformat() for value in filtered.to_series().to_list()]


def _load_or_create_market_daily_run(
    sqlite_path: Path,
    run_id: str,
    source: str,
    datasets: list[str],
    start_date: str,
    end_date: str,
    as_of_date: str,
    task_count: int,
) -> None:
    existing = _load_provider_run(sqlite_path, run_id)
    if existing is not None:
        return
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO provider_runs (
              run_id,
              source,
              dataset_name,
              start_date,
              end_date,
              as_of_date,
              status,
              total_symbols,
              next_offset,
              batch_size,
              requested_symbols,
              symbols_with_rows,
              failed_symbols,
              row_count,
              raw_batch_ids,
              failure_json,
              created_at,
              updated_at,
              notes
            )
            VALUES (?, ?, 'market_daily', ?, ?, ?, 'running', 0, 0, 0, ?, 0, 0, 0, '[]', '[]', ?, ?, ?)
            """,
            (
                run_id,
                source,
                start_date,
                end_date,
                as_of_date,
                task_count,
                now,
                now,
                "datasets=" + ",".join(datasets),
            ),
        )


def _ensure_market_daily_tasks(
    sqlite_path: Path,
    run_id: str,
    source: str,
    datasets: list[str],
    trade_dates: list[str],
    symbols: list[str],
    symbol_batch_size: int,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows: list[tuple[object, ...]] = []
    for dataset in datasets:
        for trade_date in trade_dates:
            if dataset == "moneyflow_dc":
                for start_offset in range(0, len(symbols), symbol_batch_size):
                    end_offset = min(start_offset + symbol_batch_size, len(symbols))
                    rows.append(
                        (
                            _market_daily_task_id(run_id, dataset, trade_date, start_offset, end_offset),
                            run_id,
                            source,
                            dataset,
                            trade_date,
                            start_offset,
                            end_offset,
                            now,
                            now,
                        )
                    )
            else:
                rows.append(
                    (
                        _market_daily_task_id(run_id, dataset, trade_date, None, None),
                        run_id,
                        source,
                        dataset,
                        trade_date,
                        None,
                        None,
                        now,
                        now,
                    )
                )
    with sqlite3.connect(sqlite_path) as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO provider_run_tasks (
              task_id,
              run_id,
              source,
              dataset_name,
              trade_date,
              symbol_start_offset,
              symbol_end_offset,
              status,
              attempts,
              raw_batch_id,
              row_count,
              error_message,
              created_at,
              updated_at,
              started_at,
              finished_at,
              notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, 0, NULL, ?, ?, NULL, NULL, 'market daily date task')
            """,
            rows,
        )


def _next_market_daily_task(sqlite_path: Path, run_id: str) -> dict[str, object] | None:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT task_id, run_id, source, dataset_name, trade_date, symbol_start_offset, symbol_end_offset, attempts
            FROM provider_run_tasks
            WHERE run_id = ?
              AND status IN ('pending', 'running', 'failed')
            ORDER BY trade_date, dataset_name, COALESCE(symbol_start_offset, -1)
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    keys = ["task_id", "run_id", "source", "dataset_name", "trade_date", "symbol_start_offset", "symbol_end_offset", "attempts"]
    return dict(zip(keys, row, strict=True))


def _run_market_daily_task(
    config_path: Path,
    sqlite_path: Path,
    task: dict[str, object],
    token: str,
    symbols: list[str],
    as_of_date: str,
    retry: int,
    retry_wait_seconds: float,
    backoff_multiplier: float,
    requests_per_minute: float,
    last_request_at: float | None,
) -> tuple[ProviderFetchResult, float | None]:
    task_id = str(task["task_id"])
    dataset = str(task["dataset_name"])
    trade_date = str(task["trade_date"])
    ts_code = _task_ts_code(task, symbols)
    max_attempts = retry + 1
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        _mark_market_daily_task_running(sqlite_path, task_id)
        last_request_at = _throttle_request(requests_per_minute, last_request_at)
        try:
            frame = _fetch_tushare_dataset(
                token=token,
                dataset=dataset,
                start_date=None,
                end_date=None,
                ts_code=ts_code,
                trade_date=trade_date,
            )
            result = write_raw_batch(
                config_path=config_path,
                source=str(task["source"]),
                dataset=dataset,
                frame=frame,
                as_of_date=as_of_date,
            )
            if not result.ok:
                _mark_market_daily_task_failed(sqlite_path, task_id, attempt, result.message)
                return result, last_request_at
            _mark_market_daily_task_success(sqlite_path, task_id, attempt, str(result.batch_id), result.row_count)
            return result, last_request_at
        except ModuleNotFoundError:
            message = "missing dependency: install tushare to fetch provider data"
            _mark_market_daily_task_failed(sqlite_path, task_id, attempt, message)
            return ProviderFetchResult(False, message), last_request_at
        except Exception as error:
            message = str(error)
            if _is_permission_error(message):
                _mark_market_daily_task_failed(sqlite_path, task_id, attempt, message)
                return ProviderFetchResult(False, f"provider permission failed: {message}"), last_request_at
            if attempt >= max_attempts:
                _mark_market_daily_task_failed(sqlite_path, task_id, attempt, message)
                return ProviderFetchResult(False, f"provider fetch failed after {attempt} attempts: {message}"), last_request_at
            wait_seconds = retry_wait_seconds * (backoff_multiplier ** (attempt - 1))
            if _is_rate_limit_error(message):
                time.sleep(wait_seconds)
            else:
                time.sleep(wait_seconds)
    return ProviderFetchResult(False, "provider fetch failed unexpectedly"), last_request_at


def _mark_market_daily_task_running(sqlite_path: Path, task_id: str) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'running', updated_at = ?, started_at = COALESCE(started_at, ?)
            WHERE task_id = ?
            """,
            (now, now, task_id),
        )


def _mark_market_daily_task_success(sqlite_path: Path, task_id: str, attempts: int, batch_id: str, row_count: int) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'success',
                attempts = ?,
                raw_batch_id = ?,
                row_count = ?,
                error_message = NULL,
                updated_at = ?,
                finished_at = ?
            WHERE task_id = ?
            """,
            (attempts, batch_id, row_count, now, now, task_id),
        )


def _mark_market_daily_task_failed(sqlite_path: Path, task_id: str, attempts: int, message: str) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'failed',
                attempts = ?,
                error_message = ?,
                updated_at = ?,
                finished_at = ?
            WHERE task_id = ?
            """,
            (attempts, message, now, now, task_id),
        )


def _market_daily_run_stats(sqlite_path: Path, run_id: str) -> dict[str, int]:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success,
              SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS remaining,
              SUM(row_count) AS rows
            FROM provider_run_tasks
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    return {
        "total": int(row[0] or 0),
        "success": int(row[1] or 0),
        "failed": int(row[2] or 0),
        "remaining": int(row[3] or 0),
        "rows": int(row[4] or 0),
    }


def _update_market_daily_run_summary(sqlite_path: Path, run_id: str, status: str) -> None:
    stats = _market_daily_run_stats(sqlite_path, run_id)
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            SELECT raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = ? AND raw_batch_id IS NOT NULL
            ORDER BY trade_date, dataset_name
            """,
            (run_id,),
        ).fetchall()
        connection.execute(
            """
            UPDATE provider_runs
            SET status = ?,
                requested_symbols = ?,
                row_count = ?,
                raw_batch_ids = ?,
                updated_at = ?
            WHERE run_id = ?
            """,
            (
                status,
                stats["success"],
                stats["rows"],
                json.dumps([row[0] for row in rows]),
                datetime.now(UTC).isoformat(timespec="seconds"),
                run_id,
            ),
        )


def _market_daily_task_count(
    datasets: list[str],
    trade_dates: list[str],
    symbols: list[str],
    symbol_batch_size: int,
) -> int:
    total = 0
    symbol_batches = (len(symbols) + symbol_batch_size - 1) // symbol_batch_size if symbols else 0
    for dataset in datasets:
        if dataset == "moneyflow_dc":
            total += len(trade_dates) * symbol_batches
        else:
            total += len(trade_dates)
    return total


def _task_ts_code(task: dict[str, object], symbols: list[str]) -> str | None:
    if str(task["dataset_name"]) != "moneyflow_dc":
        return None
    start = task.get("symbol_start_offset")
    end = task.get("symbol_end_offset")
    if start is None or end is None:
        return None
    selected = symbols[int(start) : int(end)]
    return ",".join(selected)


def _throttle_request(requests_per_minute: float, last_request_at: float | None) -> float:
    if requests_per_minute <= 0:
        return time.monotonic()
    interval = 60.0 / requests_per_minute
    now = time.monotonic()
    if last_request_at is not None:
        elapsed = now - last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
    return time.monotonic()


def _is_rate_limit_error(message: str) -> bool:
    lowered = message.lower()
    markers = ("每分钟", "频次", "频率", "访问该接口", "最多访问", "超过", "limit", "rate")
    return any(marker in lowered for marker in markers)


def _is_permission_error(message: str) -> bool:
    lowered = message.lower()
    markers = ("权限", "没有权限", "积分", "2002", "permission")
    return any(marker in lowered for marker in markers)


def _market_daily_task_id(
    run_id: str,
    dataset: str,
    trade_date: str,
    symbol_start_offset: int | None,
    symbol_end_offset: int | None,
) -> str:
    base = f"{run_id}:{dataset}:{_clean_date(trade_date)}"
    if symbol_start_offset is None or symbol_end_offset is None:
        return base
    return f"{base}:{symbol_start_offset}-{symbol_end_offset}"


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value.replace("/", "-"))


def _clean_date(value: str) -> str:
    return value.replace("-", "").replace("/", "")


def _probe_tushare_api(
    token: str,
    api: str,
    ts_code: str | None,
    trade_date: str | None,
    start_date: str | None,
    end_date: str | None,
) -> pl.DataFrame:
    import tushare as ts  # type: ignore[import-not-found]

    pro = ts.pro_api(token)
    if api == "stock_basic":
        pandas_frame = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,market,list_date",
        )
    elif api == "trade_cal":
        pandas_frame = pro.trade_cal(
            exchange="SSE",
            start_date=_compact_date(start_date or trade_date),
            end_date=_compact_date(end_date or trade_date),
        )
    elif api == "daily":
        pandas_frame = pro.daily(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    elif api == "moneyflow_dc":
        pandas_frame = pro.moneyflow_dc(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    elif api == "moneyflow_ths":
        pandas_frame = pro.moneyflow_ths(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    else:
        if not ts_code:
            raise ValueError("cyq_perf probe requires --ts-code")
        pandas_frame = pro.cyq_perf(
            ts_code=ts_code,
            trade_date=_compact_date(trade_date),
            start_date=_compact_date(start_date),
            end_date=_compact_date(end_date),
        )
    return pl.from_pandas(pandas_frame)


def _compact_date(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("-", "")


def _next_batch_id(sqlite_path: Path, source: str, dataset: str, as_of_date: str) -> str:
    prefix = f"{source}_{dataset}_{as_of_date.replace('-', '')}_"
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            "SELECT batch_id FROM data_batches WHERE batch_id LIKE ?",
            (prefix + "%",),
        ).fetchall()
    max_suffix = 0
    for (batch_id,) in rows:
        suffix = str(batch_id).removeprefix(prefix)
        if suffix.isdigit():
            max_suffix = max(max_suffix, int(suffix))
    return f"{prefix}{max_suffix + 1:03d}"


def _load_security_master_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise ValueError(f"security_master current Parquet not found: {path}")
    frame = pl.read_parquet(path)
    if "symbol" not in frame.columns:
        raise ValueError("security_master current Parquet is missing column: symbol")
    if "status" in frame.columns:
        frame = frame.filter(pl.col("status") == "active")
    return frame.select("symbol").drop_nulls().unique().sort("symbol").to_series().to_list()


def _load_or_create_provider_run(
    sqlite_path: Path,
    source: str,
    dataset: str,
    run_id: str | None,
    start_date: str | None,
    end_date: str | None,
    as_of_date: str,
    total_symbols: int,
    batch_size: int,
) -> dict[str, object]:
    selected_run_id = run_id or f"{source}_{dataset}_{as_of_date.replace('-', '')}_run"
    existing = _load_provider_run(sqlite_path, selected_run_id)
    if existing is not None:
        return existing
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            INSERT INTO provider_runs (
              run_id,
              source,
              dataset_name,
              start_date,
              end_date,
              as_of_date,
              status,
              total_symbols,
              next_offset,
              batch_size,
              requested_symbols,
              symbols_with_rows,
              failed_symbols,
              row_count,
              raw_batch_ids,
              failure_json,
              created_at,
              updated_at,
              notes
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0, ?, 0, 0, 0, 0, '[]', '[]', ?, ?, ?)
            """,
            (
                selected_run_id,
                source,
                dataset,
                start_date,
                end_date,
                as_of_date,
                total_symbols,
                batch_size,
                now,
                now,
                "resumable provider run",
            ),
        )
    loaded = _load_provider_run(sqlite_path, selected_run_id)
    if loaded is None:
        raise RuntimeError(f"failed to create provider run: {selected_run_id}")
    return loaded


def _load_provider_run(sqlite_path: Path, run_id: str) -> dict[str, object] | None:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
              run_id,
              source,
              dataset_name,
              start_date,
              end_date,
              as_of_date,
              status,
              total_symbols,
              next_offset,
              batch_size,
              requested_symbols,
              symbols_with_rows,
              failed_symbols,
              row_count,
              raw_batch_ids,
              failure_json
            FROM provider_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    keys = [
        "run_id",
        "source",
        "dataset_name",
        "start_date",
        "end_date",
        "as_of_date",
        "status",
        "total_symbols",
        "next_offset",
        "batch_size",
        "requested_symbols",
        "symbols_with_rows",
        "failed_symbols",
        "row_count",
        "raw_batch_ids",
        "failure_json",
    ]
    return dict(zip(keys, row, strict=True))


def _advance_provider_run(
    sqlite_path: Path,
    run_id: str,
    next_offset: int,
    symbols_requested: int,
    symbols_with_rows: int,
    failures: int,
    row_count: int,
    batch_id: str,
) -> None:
    run = _load_provider_run(sqlite_path, run_id)
    if run is None:
        raise ValueError(f"provider run not found: {run_id}")
    raw_batch_ids = json.loads(str(run["raw_batch_ids"]))
    raw_batch_ids.append(batch_id)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_runs
            SET
              status = 'running',
              next_offset = ?,
              requested_symbols = requested_symbols + ?,
              symbols_with_rows = symbols_with_rows + ?,
              failed_symbols = failed_symbols + ?,
              row_count = row_count + ?,
              raw_batch_ids = ?,
              updated_at = ?
            WHERE run_id = ?
            """,
            (
                next_offset,
                symbols_requested,
                symbols_with_rows,
                failures,
                row_count,
                json.dumps(raw_batch_ids),
                now,
                run_id,
            ),
        )


def _update_provider_run_failure(sqlite_path: Path, run_id: str, message: str) -> None:
    run = _load_provider_run(sqlite_path, run_id)
    if run is None:
        return
    failures = json.loads(str(run["failure_json"]))
    failures.append({"message": message, "at": datetime.now(UTC).isoformat(timespec="seconds")})
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_runs
            SET status = 'failed', failure_json = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (json.dumps(failures), datetime.now(UTC).isoformat(timespec="seconds"), run_id),
        )


def _mark_provider_run_completed(sqlite_path: Path, run_id: str) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_runs
            SET status = 'completed', updated_at = ?
            WHERE run_id = ?
            """,
            (datetime.now(UTC).isoformat(timespec="seconds"), run_id),
        )


def _message_int(message: str, key: str) -> int:
    marker = f"{key}="
    for part in message.split():
        if part.startswith(marker):
            return int(part.removeprefix(marker).strip(","))
    return 0


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
