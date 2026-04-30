"""Report display helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from prettytable import PrettyTable

from stock_picker.config import load_storage_config


@dataclass(frozen=True)
class ReportDisplayResult:
    ok: bool
    message: str


def show_report(config_path: Path, report_id: str, limit: int = 10) -> ReportDisplayResult:
    if limit <= 0:
        return ReportDisplayResult(False, "show-report failed; limit must be positive")
    config = load_storage_config(config_path)
    report_dir = config.reports_root / "factor_research" / report_id
    if not report_dir.exists():
        return ReportDisplayResult(False, f"factor research report not found: {report_id}")

    summary_path = report_dir / "factor_summary.json"
    ranking_path = report_dir / "ranking.csv"
    backtest_path = report_dir / "backtest_trades.csv"
    metrics_path = report_dir / "metrics.json"
    factor_results_path = report_dir / "factor_results.csv"
    missing = [
        path.name
        for path in [summary_path, factor_results_path, ranking_path, backtest_path, metrics_path]
        if not path.exists()
    ]
    if missing:
        return ReportDisplayResult(False, "show-report failed; missing artifacts: " + ", ".join(missing))

    factor_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ranking = pl.read_csv(ranking_path)

    return ReportDisplayResult(
        True,
        "\n".join(
            [
                f"report_id: {report_id}",
                "factor_description:",
                _factor_description_table(factor_summary),
                "factor_conditions:",
                _conditions_table(factor_summary),
                "ranking_results:",
                _frame_table(_ranking_display(ranking).head(limit)),
                "backtest_result:",
                _metrics_table(metrics),
                f"backtest_trades_artifact: {backtest_path}",
            ]
        ),
    )


def _factor_description_table(summary: dict[str, Any]) -> str:
    rows = [
        ["factor_id", summary.get("factor_id")],
        ["factor_name", summary.get("factor_name")],
        ["strategy_description", summary.get("strategy_description")],
        ["factor_value_description", summary.get("factor_value_description")],
        ["ranking_description", summary.get("ranking_description")],
        ["backtest_description", summary.get("backtest_description")],
        ["snapshot_id", summary.get("snapshot_id")],
        ["latest_trade_date", summary.get("latest_trade_date")],
        ["ranking_top", summary.get("ranking_top")],
        ["holding_days", summary.get("holding_days")],
        ["entry_price_rule", summary.get("entry_price_rule")],
        ["exit_price_rule", summary.get("exit_price_rule")],
        ["ranking_order", ", ".join(str(value) for value in summary.get("ranking_order", []))],
    ]
    return _rows_table(["field", "value"], rows)


def _conditions_table(summary: dict[str, Any]) -> str:
    rows = [
        [condition.get("name"), condition.get("rule"), condition.get("description")]
        for condition in summary.get("conditions", [])
        if isinstance(condition, dict)
    ]
    return _rows_table(["factor", "rule", "description"], rows)


def _factor_result_display(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "factor_rank",
        "symbol",
        "name",
        "factor_value",
    ]
    return frame.select([column for column in columns if column in frame.columns])


def _ranking_display(frame: pl.DataFrame) -> pl.DataFrame:
    if "rank" in frame.columns and "factor_rank" not in frame.columns:
        frame = frame.rename({"rank": "factor_rank"})
    columns = [
        "factor_rank",
        "symbol",
        "name",
        "factor_value",
    ]
    return frame.select([column for column in columns if column in frame.columns])


def _metrics_table(metrics: dict[str, Any]) -> str:
    rows = [
        ["signal_dates", metrics.get("signal_dates")],
        ["trade_count", metrics.get("trade_count")],
        ["avg_forward_return", _format_percent(metrics.get("avg_forward_return"))],
        ["median_forward_return", _format_percent(metrics.get("median_forward_return"))],
        ["win_rate", _format_percent(metrics.get("win_rate"))],
    ]
    return _rows_table(["metric", "value"], rows)


def _frame_table(frame: pl.DataFrame) -> str:
    rows = [list(row) for row in frame.iter_rows()]
    return _rows_table(frame.columns, rows)


def _rows_table(field_names: list[str], rows: list[list[object]]) -> str:
    table = PrettyTable()
    table.field_names = field_names
    table.align = "l"
    table.max_width = 48
    for row in rows:
        table.add_row([_format_cell(value) for value in row])
    return table.get_string()


def _format_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _format_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"
