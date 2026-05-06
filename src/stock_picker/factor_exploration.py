"""Factor exploration and Strategy Candidate 002 helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.storage import initialize_metadata_catalog
from stock_picker.strategy import (
    _candidate_001_portfolio_metrics,
    _entry_exit_prices,
    _format_optional_float,
    _load_snapshot_dataset_paths,
    _with_adjusted_prices,
)


FACTOR_VERSION = "flow_momentum_quality_v001"
DEFAULT_FACTOR_COLUMNS = [
    "mom_20",
    "mom_60",
    "ma_strength",
    "flow_5d",
    "flow_20d",
    "flow_persistence",
    "winner_rate",
    "winner_rate_change_5d",
    "vol_20",
    "max_drawdown_60",
    "avg_amount_20",
    "total_score",
]


@dataclass(frozen=True)
class FactorExplorationResult:
    ok: bool
    message: str
    run_dir: Path | None = None


def compute_daily_factors(
    config_path: Path,
    snapshot_id: str,
    start_date: str,
    end_date: str,
    run_id: str | None = None,
) -> FactorExplorationResult:
    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    paths = _load_snapshot_dataset_paths(config.metadata_sqlite_path, snapshot_id)
    missing = [dataset for dataset in ("security_master", "daily_prices", "capital_flow_or_chip") if dataset not in paths]
    if missing:
        return FactorExplorationResult(False, "factor compute failed; missing snapshot datasets: " + ", ".join(missing))

    security = pl.read_parquet(paths["security_master"])
    daily = pl.read_parquet(paths["daily_prices"])
    capital = pl.read_parquet(paths["capital_flow_or_chip"])
    if security.is_empty() or daily.is_empty() or capital.is_empty():
        return FactorExplorationResult(False, "factor compute failed; one or more input datasets are empty")

    selected_run_id = run_id or _default_factor_run_id(snapshot_id)
    run_dir = config.reports_root / "factor_exploration" / selected_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    factors = _compute_factor_frame(security, daily, capital, start_date, end_date)
    factors_path = run_dir / "stock_factors_daily.csv"
    metadata_path = run_dir / "factor_run_metadata.json"
    factors.write_csv(factors_path)
    metadata = {
        "factor_run_id": selected_run_id,
        "snapshot_id": snapshot_id,
        "factor_version": FACTOR_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_paths": {dataset: str(path) for dataset, path in paths.items()},
        "artifacts": {"stock_factors_daily": str(factors_path)},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return FactorExplorationResult(
        True,
        "\n".join(
            [
                f"factor_run_id: {selected_run_id}",
                f"run_dir: {run_dir}",
                f"stock_factors_daily: {factors_path}",
                f"row_count: {factors.height}",
            ]
        ),
        run_dir,
    )


def evaluate_factor_run(
    config_path: Path,
    factor_run_id: str,
    forward_days: int = 20,
    groups: int = 5,
) -> FactorExplorationResult:
    if forward_days <= 0:
        return FactorExplorationResult(False, "factor evaluate failed; forward_days must be positive")
    if groups < 2:
        return FactorExplorationResult(False, "factor evaluate failed; groups must be >= 2")

    config = load_storage_config(config_path)
    run_dir = config.reports_root / "factor_exploration" / factor_run_id
    factors_path = run_dir / "stock_factors_daily.csv"
    metadata_path = run_dir / "factor_run_metadata.json"
    if not factors_path.exists() or not metadata_path.exists():
        return FactorExplorationResult(False, f"factor run not found: {factor_run_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    factors = pl.read_csv(factors_path, try_parse_dates=True)
    daily_path = Path(metadata["dataset_paths"]["daily_prices"])
    daily = _with_date_column(pl.read_parquet(daily_path), "trade_date")
    evaluation = _factor_evaluation_frames(factors, daily, forward_days, groups)

    summary_path = run_dir / "evaluation_summary.json"
    ic_path = run_dir / "ic_by_date.csv"
    group_path = run_dir / "group_returns.csv"
    coverage_path = run_dir / "coverage.csv"
    correlation_path = run_dir / "correlation.csv"

    evaluation["ic_by_date"].write_csv(ic_path)
    evaluation["group_returns"].write_csv(group_path)
    evaluation["coverage"].write_csv(coverage_path)
    evaluation["correlation"].write_csv(correlation_path)
    summary_path.write_text(json.dumps(evaluation["summary"], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return FactorExplorationResult(
        True,
        "\n".join(
            [
                f"factor evaluation: {factor_run_id}",
                f"evaluation_summary: {summary_path}",
                f"ic_by_date: {ic_path}",
                f"group_returns: {group_path}",
                f"coverage: {coverage_path}",
                f"correlation: {correlation_path}",
            ]
        ),
        run_dir,
    )


def rank_candidate_002(
    config_path: Path,
    factor_run_id: str,
    trade_date: str | None = None,
    top: int = 20,
) -> FactorExplorationResult:
    if top <= 0:
        return FactorExplorationResult(False, "candidate 002 ranking failed; top must be positive")
    factors, run_dir = _load_factor_run(config_path, factor_run_id)
    selected_date = trade_date or str(factors.select(pl.col("trade_date").max()).item())
    ranked = (
        factors.filter((pl.col("trade_date").cast(pl.Utf8) == selected_date) & pl.col("tradable_flag").fill_null(False))
        .sort("total_score", descending=True)
        .head(top)
        .with_row_index("rank", offset=1)
        .select(
            [
                "rank",
                "symbol",
                "name",
                "trade_date",
                "total_score",
                "momentum_score",
                "flow_score",
                "chip_score",
                "risk_score",
                "liquidity_score",
                "exclusion_reason",
            ]
        )
    )
    ranking_path = run_dir / f"candidate_002_ranking_{selected_date.replace('-', '')}.csv"
    ranked.write_csv(ranking_path)
    if ranked.is_empty():
        return FactorExplorationResult(True, f"no Strategy Candidate 002 candidates for {selected_date}", run_dir)
    return FactorExplorationResult(
        True,
        "Strategy Candidate 002 ranking:\n" + ranked.write_csv() + f"ranking_artifact: {ranking_path}",
        run_dir,
    )


def backtest_candidate_002(
    config_path: Path,
    factor_run_id: str,
    top: int = 10,
    rebalance: str = "weekly",
    benchmark_symbol: str | list[str] = "000852.SH",
) -> FactorExplorationResult:
    if top <= 0:
        return FactorExplorationResult(False, "candidate 002 backtest failed; top must be positive")
    if rebalance not in {"daily", "weekly"}:
        return FactorExplorationResult(False, "candidate 002 backtest failed; rebalance must be daily or weekly")

    factors, run_dir = _load_factor_run(config_path, factor_run_id)
    metadata = json.loads((run_dir / "factor_run_metadata.json").read_text(encoding="utf-8"))
    daily = _with_date_column(pl.read_parquet(Path(metadata["dataset_paths"]["daily_prices"])), "trade_date")
    holding_days = 1 if rebalance == "daily" else 5
    signal_dates = _rebalance_dates(factors, rebalance)
    ranked = (
        factors.filter(pl.col("trade_date").is_in(signal_dates) & pl.col("tradable_flag").fill_null(False))
        .sort(["trade_date", "total_score"], descending=[False, True])
        .group_by("trade_date", maintain_order=True)
        .head(top)
    )
    forward = _entry_exit_prices(daily, holding_days)
    rows = (
        ranked.join(forward, on=["symbol", "trade_date"], how="left")
        .with_columns(((pl.col("exit_price") / pl.col("entry_price")) - 1).alias("forward_return"))
        .filter(pl.col("forward_return").is_not_null())
    )
    if rows.is_empty():
        return FactorExplorationResult(True, f"Strategy Candidate 002 backtest found no complete {rebalance} holding rows", run_dir)

    metrics, warnings = _candidate_001_portfolio_metrics(daily, rows, holding_days, benchmark_symbol)
    benchmark_symbols = benchmark_symbol if isinstance(benchmark_symbol, list) else [benchmark_symbol]
    trade_metrics = rows.select(
        [
            pl.len().alias("trade_count"),
            pl.col("trade_date").n_unique().alias("signal_dates"),
            pl.col("forward_return").mean().alias("avg_forward_return"),
            pl.col("forward_return").median().alias("median_forward_return"),
            (pl.col("forward_return") > 0).mean().alias("win_rate"),
            pl.col("forward_return").max().alias("best_trade_return"),
            pl.col("forward_return").min().alias("worst_trade_return"),
        ]
    ).row(0, named=True)
    output = rows.select(
        [
            "trade_date",
            "symbol",
            "name",
            "total_score",
            "entry_price",
            "exit_trade_date",
            "exit_price",
            "forward_return",
        ]
    )
    trades_path = run_dir / f"candidate_002_backtest_{rebalance}.csv"
    metrics_path = run_dir / f"candidate_002_backtest_{rebalance}_metrics.json"
    output.write_csv(trades_path)
    metrics_payload = {**_json_ready(trade_metrics), **_json_ready(metrics), "warnings": warnings}
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    message_lines = [
        "Strategy Candidate 002 backtest:",
        f"rebalance: {rebalance}",
        f"holding_days: {holding_days}",
        f"top_per_date: {top}",
        f"trade_count: {trade_metrics['trade_count']}",
        f"signal_dates: {trade_metrics['signal_dates']}",
        f"avg_forward_return: {float(trade_metrics['avg_forward_return']):.6f}",
        f"median_forward_return: {float(trade_metrics['median_forward_return']):.6f}",
        f"win_rate: {float(trade_metrics['win_rate']):.6f}",
        f"annual_return: {_format_optional_float(metrics.get('annual_return'))}",
        f"annual_volatility: {_format_optional_float(metrics.get('annual_volatility'))}",
        f"sharpe_ratio: {_format_optional_float(metrics.get('sharpe_ratio'))}",
        f"max_drawdown: {_format_optional_float(metrics.get('max_drawdown'))}",
        f"benchmark_symbol: {benchmark_symbols[0] if benchmark_symbols else '000852.SH'}",
        f"benchmark_symbols: {', '.join(benchmark_symbols or ['000852.SH'])}",
        f"excess_return: {_format_optional_float(metrics.get('excess_return'))}",
        f"benchmark_metrics: {json.dumps(metrics.get('benchmark_metrics', {}), ensure_ascii=False, sort_keys=True)}",
        "warnings: " + ("; ".join(warnings) if warnings else "none"),
        f"trades_artifact: {trades_path}",
        f"metrics_artifact: {metrics_path}",
    ]
    return FactorExplorationResult(True, "\n".join(message_lines), run_dir)


def _compute_factor_frame(
    security: pl.DataFrame,
    daily: pl.DataFrame,
    capital: pl.DataFrame,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    daily = _with_date_column(daily, "trade_date")
    capital = _with_date_column(capital, "trade_date")
    if "list_date" in security.columns:
        security = _with_date_column(security, "list_date")
    daily_stock = _with_adjusted_prices(daily).filter(pl.col("asset_type").fill_null("stock") != "index")
    price = (
        daily_stock.sort(["symbol", "trade_date"])
        .with_columns(
            [
                ((pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol")) - 1).alias("daily_return"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(20).over("symbol")) - 1).alias("mom_20"),
                ((pl.col("adj_close") / pl.col("adj_close").shift(60).over("symbol")) - 1).alias("mom_60"),
                ((pl.col("adj_close") / pl.col("adj_close").rolling_mean(20).over("symbol")) - 1).alias("ma_strength"),
                pl.col("amount").rolling_mean(20).over("symbol").alias("avg_amount_20"),
                (pl.col("amount").rolling_std(20).over("symbol") / pl.col("amount").rolling_mean(20).over("symbol")).alias("turnover_stability"),
            ]
        )
        .with_columns(
            [
                pl.col("adj_close").ewm_mean(span=12, adjust=False).over("symbol").alias("_ema12"),
                pl.col("adj_close").ewm_mean(span=26, adjust=False).over("symbol").alias("_ema26"),
                pl.col("daily_return").rolling_std(20).over("symbol").alias("vol_20"),
                pl.when(pl.col("daily_return") < 0).then(pl.col("daily_return")).otherwise(None).rolling_std(20).over("symbol").alias("downside_vol_20"),
                ((pl.col("adj_close") / pl.col("adj_close").rolling_max(60).over("symbol")) - 1).alias("max_drawdown_60"),
            ]
        )
        .with_columns((pl.col("_ema12") - pl.col("_ema26")).alias("_dif"))
        .with_columns(pl.col("_dif").ewm_mean(span=9, adjust=False).over("symbol").alias("_dea"))
        .with_columns(((pl.col("_dif") > pl.col("_dea")) & (pl.col("_dif").shift(1).over("symbol") <= pl.col("_dea").shift(1).over("symbol"))).alias("macd_signal"))
    )
    flow = (
        capital.sort(["symbol", "trade_date"])
        .with_columns(
            [
                pl.col("main_net_inflow_rate").rolling_sum(5).over("symbol").alias("flow_5d"),
                pl.col("main_net_inflow_rate").rolling_sum(20).over("symbol").alias("flow_20d"),
                (pl.col("main_net_inflow_rate") > 0).cast(pl.Float64).rolling_mean(10).over("symbol").alias("flow_persistence"),
                pl.col("close_profit_ratio").alias("winner_rate"),
                (pl.col("close_profit_ratio") - pl.col("close_profit_ratio").shift(5).over("symbol")).alias("winner_rate_change_5d"),
                (pl.col("close_profit_ratio") - pl.col("close_profit_ratio").shift(20).over("symbol")).alias("winner_rate_change_20d"),
            ]
        )
        .select(["symbol", "trade_date", "flow_5d", "flow_20d", "flow_persistence", "winner_rate", "winner_rate_change_5d", "winner_rate_change_20d"])
    )
    base = (
        price.join(flow, on=["symbol", "trade_date"], how="left")
        .join(security.select([column for column in ["symbol", "name", "market_segment", "industry", "list_date", "status"] if column in security.columns]), on="symbol", how="left")
        .filter((pl.col("trade_date") >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")) & (pl.col("trade_date") <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")))
    )
    base = _ensure_factor_input_columns(base)
    return _score_factor_frame(base)


def _with_date_column(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        return frame
    return frame.with_columns(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias(column))


def _score_factor_frame(frame: pl.DataFrame) -> pl.DataFrame:
    scored = frame.with_columns(
        [
            _rank_score("mom_20").alias("_mom20_score"),
            _rank_score("mom_60").alias("_mom60_score"),
            _rank_score("ma_strength").alias("_ma_score"),
            _rank_score("flow_5d").alias("_flow5_score"),
            _rank_score("flow_20d").alias("_flow20_score"),
            _rank_score("flow_persistence").alias("_flow_persist_score"),
            _rank_score("vol_20", inverse=True).alias("_vol_score"),
            _rank_score("max_drawdown_60").alias("_drawdown_score"),
            _rank_score("downside_vol_20", inverse=True).alias("_downside_score"),
            _rank_score("avg_amount_20").alias("_amount_score"),
            _rank_score("turnover_stability", inverse=True).alias("_turnover_stability_score"),
            _chip_score_expr().alias("chip_score"),
        ]
    ).with_columns(
        [
            (pl.col("_mom20_score") * 0.35 + pl.col("_mom60_score") * 0.35 + pl.col("_ma_score") * 0.20 + pl.col("macd_signal").fill_null(False).cast(pl.Float64) * 0.10).alias("momentum_score"),
            (pl.col("_flow5_score") * 0.40 + pl.col("_flow20_score") * 0.30 + pl.col("_flow_persist_score") * 0.30).alias("flow_score"),
            (pl.col("_vol_score") * 0.40 + pl.col("_drawdown_score") * 0.40 + pl.col("_downside_score") * 0.20).alias("risk_score"),
            (pl.col("_amount_score") * 0.70 + pl.col("_turnover_stability_score") * 0.30).alias("liquidity_score"),
        ]
    ).with_columns(
        (
            pl.col("momentum_score") * 0.30
            + pl.col("flow_score") * 0.25
            + pl.col("chip_score") * 0.15
            + pl.col("risk_score") * 0.15
            + pl.col("liquidity_score") * 0.15
        ).alias("total_score")
    )
    scored = scored.with_columns(_exclusion_reason_expr().alias("exclusion_reason"))
    return scored.with_columns((pl.col("exclusion_reason") == "none").alias("tradable_flag")).with_columns(pl.lit(FACTOR_VERSION).alias("factor_version")).select(
        [
            "symbol",
            "trade_date",
            "name",
            "market_segment",
            "industry",
            "mom_20",
            "mom_60",
            "ma_strength",
            "macd_signal",
            "flow_5d",
            "flow_20d",
            "flow_persistence",
            "winner_rate",
            "winner_rate_change_5d",
            "winner_rate_change_20d",
            "vol_20",
            "max_drawdown_60",
            "downside_vol_20",
            "avg_amount_20",
            "turnover_stability",
            "momentum_score",
            "flow_score",
            "chip_score",
            "risk_score",
            "liquidity_score",
            "total_score",
            "tradable_flag",
            "exclusion_reason",
            "factor_version",
        ]
    )


def _rank_score(column: str, inverse: bool = False) -> pl.Expr:
    count = pl.col(column).count().over("trade_date")
    rank = pl.col(column).rank().over("trade_date")
    score = rank / count
    if inverse:
        score = 1.0 - score + (1.0 / count)
    return score.fill_null(0.0)


def _ensure_factor_input_columns(frame: pl.DataFrame) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "name": None,
        "market_segment": None,
        "industry": None,
        "list_date": None,
        "status": "active",
        "is_suspended": False,
        "limit_up": None,
    }
    expressions = [pl.lit(value).alias(column) for column, value in defaults.items() if column not in frame.columns]
    if expressions:
        frame = frame.with_columns(expressions)
    return frame


def _chip_score_expr() -> pl.Expr:
    raw = 1.0 - ((pl.col("winner_rate") - 80.0).abs() / 30.0)
    return pl.when(raw > 1).then(1.0).when(raw < 0).then(0.0).otherwise(raw).fill_null(0.0)


def _exclusion_reason_expr() -> pl.Expr:
    list_age_days = (pl.col("trade_date") - pl.col("list_date")).dt.total_days()
    return (
        pl.when(pl.col("status").fill_null("unknown") != "active")
        .then(pl.lit("inactive_security"))
        .when(pl.col("name").fill_null("").str.contains(r"\*?ST"))
        .then(pl.lit("st_security"))
        .when(pl.col("is_suspended").fill_null(False))
        .then(pl.lit("suspended"))
        .when(pl.col("limit_up").is_not_null() & (pl.col("close") >= pl.col("limit_up")))
        .then(pl.lit("limit_up"))
        .when(list_age_days.is_not_null() & (list_age_days < 120))
        .then(pl.lit("listing_age_lt_120"))
        .when(pl.col("avg_amount_20").fill_null(0) < 50000)
        .then(pl.lit("low_liquidity"))
        .otherwise(pl.lit("none"))
    )


def _factor_evaluation_frames(factors: pl.DataFrame, daily: pl.DataFrame, forward_days: int, groups: int) -> dict[str, Any]:
    forward = _entry_exit_prices(daily, forward_days).with_columns(((pl.col("exit_price") / pl.col("entry_price")) - 1).alias("forward_return"))
    frame = factors.join(forward.select(["symbol", "trade_date", "forward_return"]), on=["symbol", "trade_date"], how="left")
    ic_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for factor_name in DEFAULT_FACTOR_COLUMNS:
        for date_value in frame.select("trade_date").unique().sort("trade_date").to_series().to_list():
            subset = frame.filter((pl.col("trade_date") == date_value) & pl.col(factor_name).is_not_null() & pl.col("forward_return").is_not_null())
            if subset.height < 3:
                continue
            factor_values = [float(value) for value in subset[factor_name].to_list()]
            returns = [float(value) for value in subset["forward_return"].to_list()]
            ic_rows.append(
                {
                    "factor": factor_name,
                    "trade_date": date_value,
                    "ic": _pearson(factor_values, returns),
                    "rank_ic": _pearson(_ranks(factor_values), _ranks(returns)),
                    "sample_count": subset.height,
                }
            )
        bucketed = (
            frame.filter(pl.col(factor_name).is_not_null() & pl.col("forward_return").is_not_null())
            .with_columns(((pl.col(factor_name).rank().over("trade_date") - 1) / pl.col(factor_name).count().over("trade_date") * groups).floor().cast(pl.Int64).clip(0, groups - 1).alias("group"))
            .group_by("group")
            .agg(pl.col("forward_return").mean().alias("avg_forward_return"), pl.len().alias("sample_count"))
            .sort("group")
        )
        for row in bucketed.to_dicts():
            group_rows.append({"factor": factor_name, **row})
    ic_by_date = pl.DataFrame(ic_rows) if ic_rows else pl.DataFrame({"factor": [], "trade_date": [], "ic": [], "rank_ic": [], "sample_count": []})
    coverage = pl.DataFrame(
        [
            {
                "factor": factor_name,
                "non_null_count": int(factors[factor_name].drop_nulls().len()),
                "row_count": factors.height,
                "coverage": float(factors[factor_name].drop_nulls().len() / factors.height) if factors.height else 0.0,
            }
            for factor_name in DEFAULT_FACTOR_COLUMNS
        ]
    )
    correlation = _correlation_frame(factors, DEFAULT_FACTOR_COLUMNS)
    summary = _evaluation_summary(ic_by_date)
    return {
        "summary": summary,
        "ic_by_date": ic_by_date,
        "group_returns": pl.DataFrame(group_rows) if group_rows else pl.DataFrame({"factor": [], "group": [], "avg_forward_return": [], "sample_count": []}),
        "coverage": coverage,
        "correlation": correlation,
    }


def _evaluation_summary(ic_by_date: pl.DataFrame) -> dict[str, Any]:
    if ic_by_date.is_empty():
        return {"factor_count": 0, "factors": {}}
    rows = {}
    for row in ic_by_date.group_by("factor").agg(
        [
            pl.col("ic").mean().alias("mean_ic"),
            pl.col("ic").std().alias("std_ic"),
            pl.col("rank_ic").mean().alias("mean_rank_ic"),
            pl.col("rank_ic").std().alias("std_rank_ic"),
            pl.len().alias("date_count"),
        ]
    ).to_dicts():
        std_ic = row["std_ic"] or 0.0
        std_rank_ic = row["std_rank_ic"] or 0.0
        rows[str(row["factor"])] = {
            "mean_ic": row["mean_ic"],
            "icir": row["mean_ic"] / std_ic if std_ic else None,
            "mean_rank_ic": row["mean_rank_ic"],
            "rank_icir": row["mean_rank_ic"] / std_rank_ic if std_rank_ic else None,
            "date_count": row["date_count"],
        }
    return {"factor_count": len(rows), "factors": rows}


def _correlation_frame(factors: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    rows = []
    for left in columns:
        row = {"factor": left}
        for right in columns:
            subset = factors.filter(pl.col(left).is_not_null() & pl.col(right).is_not_null())
            row[right] = _pearson([float(value) for value in subset[left].to_list()], [float(value) for value in subset[right].to_list()]) if subset.height >= 2 else None
        rows.append(row)
    return pl.DataFrame(rows)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_denominator = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_denominator = sum((b - right_mean) ** 2 for b in right) ** 0.5
    if not left_denominator or not right_denominator:
        return None
    return numerator / (left_denominator * right_denominator)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    for rank, (_value, index) in enumerate(ordered, start=1):
        ranks[index] = float(rank)
    return ranks


def _rebalance_dates(factors: pl.DataFrame, rebalance: str) -> list[object]:
    dates = factors.select("trade_date").unique().sort("trade_date")
    if rebalance == "daily":
        return dates.to_series().to_list()
    return (
        dates.with_columns(pl.col("trade_date").dt.strftime("%G-%V").alias("week"))
        .group_by("week", maintain_order=True)
        .agg(pl.col("trade_date").min().alias("trade_date"))
        .sort("trade_date")
        .select("trade_date")
        .to_series()
        .to_list()
    )


def _load_factor_run(config_path: Path, factor_run_id: str) -> tuple[pl.DataFrame, Path]:
    config = load_storage_config(config_path)
    run_dir = config.reports_root / "factor_exploration" / factor_run_id
    factors_path = run_dir / "stock_factors_daily.csv"
    if not factors_path.exists():
        raise ValueError(f"factor run not found: {factor_run_id}")
    return pl.read_csv(factors_path, try_parse_dates=True), run_dir


def _json_ready(values: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            value = value.item()
        output[key] = value
    return output


def _default_factor_run_id(snapshot_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"factor_exploration_{snapshot_id}_{timestamp}"
