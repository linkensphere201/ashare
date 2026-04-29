"""Provider raw data fetchers."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
