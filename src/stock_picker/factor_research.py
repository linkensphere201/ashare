"""Factor research report generation."""

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
    _CANDIDATE_SORT_COLUMNS,
    _CANDIDATE_SORT_DESCENDING,
    _candidate_001_frame,
    _entry_exit_prices,
    _load_snapshot_dataset_paths,
    _trade_date_successors,
)


@dataclass(frozen=True)
class FactorResearchResult:
    ok: bool
    message: str
    report_dir: Path | None = None


def research_candidate_001(
    config_path: Path,
    snapshot_id: str,
    holding_days: int = 20,
    top: int = 10,
    report_id: str | None = None,
) -> FactorResearchResult:
    if holding_days <= 0:
        return FactorResearchResult(False, "factor research failed: holding_days must be positive")
    if top <= 0:
        return FactorResearchResult(False, "factor research failed: top must be positive")

    config = load_storage_config(config_path)
    initialize_metadata_catalog(config.metadata_sqlite_path)
    paths = _load_snapshot_dataset_paths(config.metadata_sqlite_path, snapshot_id)
    missing = [
        dataset
        for dataset in ("security_master", "daily_prices", "capital_flow_or_chip")
        if dataset not in paths
    ]
    if missing:
        return FactorResearchResult(False, "factor research failed; missing snapshot datasets: " + ", ".join(missing))

    security = pl.read_parquet(paths["security_master"])
    daily = pl.read_parquet(paths["daily_prices"])
    capital = pl.read_parquet(paths["capital_flow_or_chip"])
    trading_calendar = pl.read_parquet(paths["trading_calendar"]) if "trading_calendar" in paths else None
    if daily.is_empty() or capital.is_empty() or security.is_empty():
        return FactorResearchResult(False, "factor research failed; one or more input datasets are empty")

    latest_trade_date = daily.select(pl.col("trade_date").max()).item()
    candidates = _candidate_001_frame(daily, capital, security)
    trade_date_successors = _trade_date_successors(daily, trading_calendar)
    ranking = _ranking_frame(candidates, trade_date_successors, latest_trade_date, top)
    factor_results = _factor_results_frame(candidates, trade_date_successors, latest_trade_date)
    backtest_rows = _backtest_frame(candidates, daily, holding_days, top)
    metrics = _backtest_metrics(backtest_rows)
    selected_report_id = report_id or _default_report_id(snapshot_id)
    factor_summary = _factor_summary(
        snapshot_id=snapshot_id,
        report_id=selected_report_id,
        latest_trade_date=str(latest_trade_date),
        holding_days=holding_days,
        top=top,
    )

    report_dir = config.reports_root / "factor_research" / selected_report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    factor_summary_path = report_dir / "factor_summary.json"
    factor_results_path = report_dir / "factor_results.csv"
    ranking_path = report_dir / "ranking.csv"
    backtest_path = report_dir / "backtest_trades.csv"
    metrics_path = report_dir / "metrics.json"
    summary_path = report_dir / "summary.md"

    factor_summary_path.write_text(json.dumps(factor_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    factor_results.write_csv(factor_results_path)
    ranking.write_csv(ranking_path)
    backtest_rows.write_csv(backtest_path)
    metrics_path.write_text(json.dumps(_json_ready(metrics), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(
        _summary_markdown(
            factor_summary=factor_summary,
            ranking=ranking,
            metrics=metrics,
            factor_results_path=factor_results_path,
            ranking_path=ranking_path,
            backtest_path=backtest_path,
            metrics_path=metrics_path,
        ),
        encoding="utf-8",
    )

    return FactorResearchResult(
        True,
        "\n".join(
            [
                f"factor research report: {selected_report_id}",
                f"report_dir: {report_dir}",
                f"summary: {summary_path}",
                f"factor_summary: {factor_summary_path}",
                f"factor_results: {factor_results_path}",
                f"ranking: {ranking_path}",
                f"backtest: {backtest_path}",
                f"metrics: {metrics_path}",
            ]
        ),
        report_dir,
    )


def _ranking_frame(
    candidates: pl.DataFrame,
    trade_date_successors: pl.DataFrame,
    latest_trade_date: object,
    top: int,
) -> pl.DataFrame:
    return (
        candidates.filter(pl.col("trade_date") == latest_trade_date)
        .join(trade_date_successors, on="trade_date", how="left")
        .sort(_CANDIDATE_SORT_COLUMNS, descending=_CANDIDATE_SORT_DESCENDING)
        .head(top)
        .with_row_index("rank", offset=1)
        .with_columns([_factor_value_expr(), _candidate_reason_expr()])
        .select(
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
                "factor_value",
                "reason",
            ]
        )
    )


def _factor_results_frame(
    candidates: pl.DataFrame,
    trade_date_successors: pl.DataFrame,
    latest_trade_date: object,
) -> pl.DataFrame:
    return (
        candidates.filter(pl.col("trade_date") == latest_trade_date)
        .join(trade_date_successors, on="trade_date", how="left")
        .sort(_CANDIDATE_SORT_COLUMNS, descending=_CANDIDATE_SORT_DESCENDING)
        .with_row_index("factor_rank", offset=1)
        .with_columns(
            [
                (pl.col("main_net_inflow_rate") > 0).alias("pass_main_net_inflow_rate"),
                pl.col("macd_golden_cross").fill_null(False).alias("pass_macd_golden_cross"),
                (pl.col("close_profit_ratio") > 80).alias("pass_close_profit_ratio"),
                pl.lit(True).alias("pass_exclude_st"),
                pl.lit(True).alias("factor_pass"),
                _factor_value_expr(),
                _candidate_reason_expr(),
            ]
        )
        .select(
            [
                "factor_rank",
                "symbol",
                "name",
                pl.col("trade_date").alias("signal_date"),
                "recommend_trade_date",
                "close",
                "main_net_inflow_rate",
                "pass_main_net_inflow_rate",
                "macd_golden_cross",
                "pass_macd_golden_cross",
                "close_profit_ratio",
                "pass_close_profit_ratio",
                "return_20d",
                "pass_exclude_st",
                "factor_pass",
                "factor_value",
                "reason",
            ]
        )
    )


def _backtest_frame(candidates: pl.DataFrame, daily: pl.DataFrame, holding_days: int, top: int) -> pl.DataFrame:
    forward_prices = _entry_exit_prices(daily, holding_days)
    return (
        candidates.join(forward_prices, on=["symbol", "trade_date"], how="left")
        .with_columns(((pl.col("exit_price") / pl.col("entry_price")) - 1).alias("forward_return"))
        .sort(["trade_date", *_CANDIDATE_SORT_COLUMNS], descending=[False, *_CANDIDATE_SORT_DESCENDING])
        .group_by("trade_date", maintain_order=True)
        .head(top)
        .filter(pl.col("forward_return").is_not_null())
        .select(
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
    )


def _backtest_metrics(backtest_rows: pl.DataFrame) -> dict[str, Any]:
    if backtest_rows.is_empty():
        return {
            "signal_dates": 0,
            "trade_count": 0,
            "avg_forward_return": None,
            "median_forward_return": None,
            "win_rate": None,
        }
    return backtest_rows.select(
        [
            pl.len().alias("trade_count"),
            pl.col("signal_date").n_unique().alias("signal_dates"),
            pl.col("forward_return").mean().alias("avg_forward_return"),
            pl.col("forward_return").median().alias("median_forward_return"),
            (pl.col("forward_return") > 0).mean().alias("win_rate"),
        ]
    ).row(0, named=True)


def _candidate_reason_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.lit("main_net_inflow_rate>0; MACD golden cross; close_profit_ratio>80; exclude ST/*ST; close_profit_ratio="),
            pl.col("close_profit_ratio").round(2).cast(pl.Utf8),
        ]
    ).alias("reason")


def _factor_value_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.lit("main_net_inflow_rate="),
            pl.col("main_net_inflow_rate").round(4).cast(pl.Utf8).fill_null("null"),
            pl.lit("; macd_golden_cross="),
            pl.col("macd_golden_cross").fill_null(False).cast(pl.Utf8),
            pl.lit("; close_profit_ratio="),
            pl.col("close_profit_ratio").round(4).cast(pl.Utf8).fill_null("null"),
            pl.lit("; return_20d="),
            pl.col("return_20d").round(6).cast(pl.Utf8).fill_null("null"),
        ]
    ).alias("factor_value")


def _factor_summary(
    snapshot_id: str,
    report_id: str,
    latest_trade_date: str,
    holding_days: int,
    top: int,
) -> dict[str, Any]:
    return {
        "factor_id": "candidate_001_v2",
        "factor_name": "策略候选 001 v2",
        "strategy_description": (
            "基于日频数据的 A 股规则型候选因子。T 日收盘后计算信号，推荐从 T+1 交易日开始观察或交易。"
            "先筛选主力资金净流入为正、MACD 金叉、"
            "收盘获利比例较高且非 ST 的股票，再按主力资金净流入比例、收盘获利比例、"
            "20 日涨跌幅排序。"
        ),
        "factor_value_description": (
            "factor_value 是可解释的因子指标值集合，不是合成加权分数。"
            "它记录当前规则因子实际使用到的各项指标值。"
        ),
        "ranking_description": (
            "排序逻辑：在通过因子条件的股票中，先按主力资金净流入比例从高到低排序；"
            "若接近或相同，再按收盘获利比例从高到低排序；最后按 20 日涨跌幅从高到低排序。"
            "Top N 表示最多取 N 只，不会强行补足。ranking_results 的 signal_date 是因子信号日，"
            "recommend_trade_date 是下一交易日。"
        ),
        "backtest_description": (
            "回测口径：T 日收盘后产生信号，T+1 入场；entry_price 优先使用 T+1 open，"
            "若标准层没有 open 或 open 为空则使用 T+1 close；exit_price 使用持有期结束日 close。"
        ),
        "snapshot_id": snapshot_id,
        "report_id": report_id,
        "latest_trade_date": latest_trade_date,
        "ranking_top": top,
        "holding_days": holding_days,
        "entry_price_rule": "T+1 open if available, otherwise T+1 close",
        "exit_price_rule": f"T+1+{holding_days} close",
        "conditions": [
            {"name": "main_net_inflow_rate", "rule": "> 0", "description": "主力资金净流入比例为正，用于近似表示主力资金参与度为正。"},
            {"name": "macd_golden_cross", "rule": "true", "description": "MACD DIF 上穿 DEA，表示短期趋势出现正向拐点。"},
            {"name": "close_profit_ratio", "rule": "> 80", "description": "由 Tushare cyq_perf.winner_rate 映射而来，表示收盘价对应的获利筹码比例较高。"},
            {"name": "exclude_st", "rule": "name does not contain ST or *ST", "description": "排除 ST 或 *ST 股票，降低特殊处理股票对排序和回测的影响。"},
        ],
        "ranking_order": ["main_net_inflow_rate desc", "close_profit_ratio desc", "return_20d desc"],
    }


def _summary_markdown(
    factor_summary: dict[str, Any],
    ranking: pl.DataFrame,
    metrics: dict[str, Any],
    factor_results_path: Path,
    ranking_path: Path,
    backtest_path: Path,
    metrics_path: Path,
) -> str:
    lines = [
        "# Factor Research: Strategy Candidate 001 v2",
        "",
        "## Factor Summary",
        "",
        f"- Factor name: {factor_summary['factor_name']}",
        f"- Snapshot: `{factor_summary['snapshot_id']}`",
        f"- Report id: `{factor_summary['report_id']}`",
        f"- Latest ranking date: {factor_summary['latest_trade_date']}",
        f"- Ranking output: Top {factor_summary['ranking_top']}",
        f"- Backtest holding window: {factor_summary['holding_days']} trading days",
        f"- Entry price rule: {factor_summary['entry_price_rule']}",
        f"- Exit price rule: {factor_summary['exit_price_rule']}",
        "",
        "## Explanation",
        "",
        "- Main-fund flow condition: `main_net_inflow_rate > 0`, used as a proxy for positive main capital participation.",
        "- Price momentum condition: MACD golden cross, used to require a fresh positive trend turn.",
        "- Chip/profit condition: `close_profit_ratio > 80`, using Tushare `cyq_perf.winner_rate` mapped into `close_profit_ratio`.",
        "- Risk guard: stock names containing `ST` or `*ST` are excluded from ranking and backtest candidates.",
        "- Timing: signals are generated after T close and recommendations are for T+1.",
        "- Ranking order: `main_net_inflow_rate desc`, then `close_profit_ratio desc`, then `return_20d desc`.",
        "",
        "## Ranking Result",
        "",
    ]
    if ranking.is_empty():
        lines.append("No candidates matched the factor on the latest ranking date.")
    else:
        lines.extend(_markdown_table(ranking.select(["rank", "symbol", "name", "signal_date", "recommend_trade_date", "factor_value"])))
    lines.extend(
        [
            "",
            "## Backtest Result",
            "",
            f"- Signal dates: {metrics['signal_dates']}",
            f"- Trade count: {metrics['trade_count']}",
            f"- Average forward return: {_format_percent(metrics['avg_forward_return'])}",
            f"- Median forward return: {_format_percent(metrics['median_forward_return'])}",
            f"- Win rate: {_format_percent(metrics['win_rate'])}",
            "",
            "## Artifacts",
            "",
            f"- Factor results CSV: `{factor_results_path.name}`",
            f"- Ranking CSV: `{ranking_path.name}`",
            f"- Backtest trades CSV: `{backtest_path.name}`",
            f"- Metrics JSON: `{metrics_path.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pl.DataFrame) -> list[str]:
    columns = frame.columns
    rows = frame.to_dicts()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return lines


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _json_ready(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in values.items()}


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _default_report_id(snapshot_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"candidate_001_v2_{snapshot_id}_{timestamp}"
