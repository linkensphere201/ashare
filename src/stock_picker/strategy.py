"""Strategy ranking helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.storage import initialize_metadata_catalog


@dataclass(frozen=True)
class StrategyResult:
    ok: bool
    message: str


def rank_candidate_001(config_path: Path, snapshot_id: str, top: int = 10) -> StrategyResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    paths = _load_snapshot_dataset_paths(config.metadata_sqlite_path, snapshot_id)
    missing = [
        dataset
        for dataset in ("security_master", "daily_prices", "capital_flow_or_chip")
        if dataset not in paths
    ]
    if missing:
        return StrategyResult(False, "strategy ranking failed; missing snapshot datasets: " + ", ".join(missing))

    security = pl.read_parquet(paths["security_master"])
    daily = pl.read_parquet(paths["daily_prices"])
    capital = pl.read_parquet(paths["capital_flow_or_chip"])
    trading_calendar = pl.read_parquet(paths["trading_calendar"]) if "trading_calendar" in paths else None
    if daily.is_empty() or capital.is_empty() or security.is_empty():
        return StrategyResult(False, "strategy ranking failed; one or more input datasets are empty")

    latest_trade_date = daily.select(pl.col("trade_date").max()).item()
    candidates = (
        _candidate_001_frame(daily, capital, security)
        .join(_trade_date_successors(daily, trading_calendar), on="trade_date", how="left")
        .filter(pl.col("trade_date") == latest_trade_date)
        .sort(_CANDIDATE_SORT_COLUMNS, descending=_CANDIDATE_SORT_DESCENDING)
        .head(top)
        .with_row_index("rank", offset=1)
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("net_amount_rate>0; MACD golden cross; winner_rate="),
                    pl.col("close_profit_ratio").round(2).cast(pl.Utf8),
                ]
            ).alias("reason")
        )
    )

    if candidates.is_empty():
        return StrategyResult(True, f"no candidates matched Strategy Candidate 001 v2 for {latest_trade_date}")

    display = candidates.select(
        [
            "rank",
            "symbol",
            "name",
            pl.col("trade_date").alias("signal_date"),
            "recommend_trade_date",
            "close",
            "main_net_inflow_rate",
            "close_profit_ratio",
            "return_20d",
            "reason",
        ]
    )
    return StrategyResult(
        True,
        "Strategy Candidate 001 v2 candidates:\n" + display.write_csv(),
    )


def backtest_candidate_001(
    config_path: Path,
    snapshot_id: str,
    holding_days: int = 20,
    top: int = 10,
) -> StrategyResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    paths = _load_snapshot_dataset_paths(config.metadata_sqlite_path, snapshot_id)
    missing = [
        dataset
        for dataset in ("security_master", "daily_prices", "capital_flow_or_chip")
        if dataset not in paths
    ]
    if missing:
        return StrategyResult(False, "strategy backtest failed; missing snapshot datasets: " + ", ".join(missing))

    security = pl.read_parquet(paths["security_master"])
    daily = pl.read_parquet(paths["daily_prices"])
    capital = pl.read_parquet(paths["capital_flow_or_chip"])
    if daily.is_empty() or capital.is_empty() or security.is_empty():
        return StrategyResult(False, "strategy backtest failed; one or more input datasets are empty")

    if holding_days <= 0:
        return StrategyResult(False, "strategy backtest failed; holding_days must be positive")

    candidates = _candidate_001_frame(daily, capital, security)
    forward_prices = _entry_exit_prices(daily, holding_days)
    backtest_rows = (
        candidates.join(forward_prices, on=["symbol", "trade_date"], how="left")
        .with_columns(((pl.col("exit_price") / pl.col("entry_price")) - 1).alias("forward_return"))
        .sort(["trade_date", *_CANDIDATE_SORT_COLUMNS], descending=[False, *_CANDIDATE_SORT_DESCENDING])
        .group_by("trade_date", maintain_order=True)
        .head(top)
        .filter(pl.col("forward_return").is_not_null())
    )

    if backtest_rows.is_empty():
        return StrategyResult(
            True,
            f"Strategy Candidate 001 v2 backtest found no complete {holding_days}-day holding rows",
        )

    metrics = backtest_rows.select(
        [
            pl.len().alias("trade_count"),
            pl.col("trade_date").n_unique().alias("signal_dates"),
            pl.col("forward_return").mean().alias("avg_forward_return"),
            pl.col("forward_return").median().alias("median_forward_return"),
            (pl.col("forward_return") > 0).mean().alias("win_rate"),
        ]
    ).row(0, named=True)
    display = backtest_rows.select(
        [
            pl.col("trade_date").alias("signal_date"),
            "recommend_trade_date",
            "symbol",
            "name",
            pl.col("close").alias("signal_close"),
            "entry_price",
            "exit_trade_date",
            "exit_price",
            "forward_return",
            "main_net_inflow_rate",
            "close_profit_ratio",
        ]
    )
    return StrategyResult(
        True,
        "\n".join(
            [
                "Strategy Candidate 001 v2 backtest:",
                "signal timing: T close signal, T+1 entry",
                "entry_price: T+1 open when available, otherwise T+1 close",
                f"holding_days: {holding_days}",
                f"top_per_date: {top}",
                f"signal_dates: {metrics['signal_dates']}",
                f"trade_count: {metrics['trade_count']}",
                f"avg_forward_return: {metrics['avg_forward_return']:.6f}",
                f"median_forward_return: {metrics['median_forward_return']:.6f}",
                f"win_rate: {metrics['win_rate']:.6f}",
                "trades:",
                display.write_csv(),
            ]
        ),
    )


_CANDIDATE_SORT_COLUMNS = ["main_net_inflow_rate", "close_profit_ratio", "return_20d"]
_CANDIDATE_SORT_DESCENDING = [True, True, True]


def _candidate_001_frame(daily: pl.DataFrame, capital: pl.DataFrame, security: pl.DataFrame) -> pl.DataFrame:
    daily_with_factors = _calculate_price_factors(daily)
    return (
        daily_with_factors.join(capital, on=["symbol", "trade_date"], how="inner")
        .join(security.select(["symbol", "name", "status"]), on="symbol", how="left")
        .filter(pl.col("status") == "active")
        .filter(~pl.col("name").fill_null("").str.contains(r"\*?ST"))
        .filter(pl.col("close").is_not_null())
        .filter(pl.col("amount").fill_null(0) > 0)
        .filter(pl.col("main_net_inflow_rate").fill_null(-999999) > 0)
        .filter(pl.col("macd_golden_cross").fill_null(False))
        .filter(pl.col("close_profit_ratio").fill_null(-999999) > 80)
    )


def _trade_date_successors(daily: pl.DataFrame, trading_calendar: pl.DataFrame | None = None) -> pl.DataFrame:
    if trading_calendar is not None and not trading_calendar.is_empty() and "trade_date" in trading_calendar.columns:
        dates = trading_calendar
        if "is_trading_day" in dates.columns:
            dates = dates.filter(pl.col("is_trading_day").fill_null(False))
        return (
            dates.select("trade_date")
            .unique()
            .sort("trade_date")
            .with_columns(pl.col("trade_date").shift(-1).alias("recommend_trade_date"))
        )
    return (
        daily.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(pl.col("trade_date").shift(-1).alias("recommend_trade_date"))
    )


def _entry_exit_prices(daily: pl.DataFrame, holding_days: int) -> pl.DataFrame:
    sorted_daily = daily.sort(["symbol", "trade_date"])
    entry_price_expr = pl.col("close").shift(-1).over("symbol")
    if "open" in daily.columns:
        entry_price_expr = pl.coalesce(
            [
                pl.col("open").shift(-1).over("symbol"),
                pl.col("close").shift(-1).over("symbol"),
            ]
        )
    return (
        sorted_daily.select(["symbol", "trade_date", "close", *([] if "open" not in daily.columns else ["open"])])
        .with_columns(
            [
                pl.col("trade_date").shift(-1).over("symbol").alias("recommend_trade_date"),
                entry_price_expr.alias("entry_price"),
                pl.col("trade_date").shift(-(holding_days + 1)).over("symbol").alias("exit_trade_date"),
                pl.col("close").shift(-(holding_days + 1)).over("symbol").alias("exit_price"),
            ]
        )
        .select(["symbol", "trade_date", "recommend_trade_date", "entry_price", "exit_trade_date", "exit_price"])
    )


def _calculate_price_factors(daily: pl.DataFrame) -> pl.DataFrame:
    sorted_daily = daily.sort(["symbol", "trade_date"])
    with_ema = sorted_daily.with_columns(
        [
            pl.col("close").ewm_mean(span=12, adjust=False).over("symbol").alias("ema12"),
            pl.col("close").ewm_mean(span=26, adjust=False).over("symbol").alias("ema26"),
            ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1).alias("return_20d"),
        ]
    ).with_columns((pl.col("ema12") - pl.col("ema26")).alias("dif"))
    return with_ema.with_columns(
        pl.col("dif").ewm_mean(span=9, adjust=False).over("symbol").alias("dea")
    ).with_columns(
        (
            (pl.col("dif").shift(1).over("symbol") <= pl.col("dea").shift(1).over("symbol"))
            & (pl.col("dif") > pl.col("dea"))
        ).alias("macd_golden_cross")
    )


def _load_snapshot_dataset_paths(sqlite_path: Path, snapshot_id: str) -> dict[str, Path]:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            "SELECT manifest_json FROM snapshot_manifests WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"snapshot not found: {snapshot_id}")
    manifest = json.loads(row[0])
    curated_versions = manifest.get("curated_versions", {})
    return {
        dataset_id: Path(details["path"])
        for dataset_id, details in curated_versions.items()
        if "path" in details
    }
