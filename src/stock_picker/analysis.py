"""Specific-stock analysis outputs for the commercial app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.factor_exploration import FACTOR_VERSION
from stock_picker.publish import factor_definitions, validate_safe_payload


@dataclass(frozen=True)
class AnalysisResult:
    ok: bool
    message: str
    artifact_path: Path | None = None
    artifact: dict[str, Any] | None = None


def analyze_stock(
    config_path: Path,
    symbol: str,
    factor_run_id: str,
    trade_date: str | None = None,
    output_path: Path | None = None,
) -> AnalysisResult:
    config = load_storage_config(config_path)
    run_dir = config.reports_root / "factor_exploration" / factor_run_id
    factors_path = run_dir / "stock_factors_daily.csv"
    metadata_path = run_dir / "factor_run_metadata.json"
    if not factors_path.exists() or not metadata_path.exists():
        return AnalysisResult(False, f"factor run not found: {factor_run_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        factors = pl.read_csv(factors_path, try_parse_dates=True)
    except Exception as error:
        return AnalysisResult(False, f"stock analysis failed; could not read factor CSV: {error}")
    if factors.is_empty():
        return AnalysisResult(False, "stock analysis failed; factor run has no rows")
    selected_date = trade_date or str(factors.select(pl.col("trade_date").max()).item())
    rows = factors.filter((pl.col("trade_date").cast(pl.Utf8) == selected_date) & (pl.col("symbol") == symbol))
    if rows.is_empty():
        artifact = _missing_payload(symbol, selected_date, factor_run_id, str(metadata.get("snapshot_id", "")))
    else:
        row = rows.to_dicts()[0]
        artifact = _stock_payload(row, selected_date, factor_run_id, str(metadata.get("snapshot_id", "")))
    errors = _validate_stock_analysis(artifact)
    if errors:
        return AnalysisResult(False, "stock analysis failed validation: " + "; ".join(errors), artifact=artifact)
    selected_output = output_path or run_dir / f"stock_analysis_{symbol.replace('.', '_')}_{selected_date.replace('-', '')}.json"
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return AnalysisResult(True, f"stock_analysis: {selected_output}", selected_output, artifact)


def _stock_payload(row: dict[str, Any], trade_date: str, factor_run_id: str, snapshot_id: str) -> dict[str, Any]:
    symbol = str(row.get("symbol"))
    return {
        "analysis_metadata": {
            "schema_version": "stock_app_stock_analysis_v001",
            "symbol": symbol,
            "trade_date": trade_date,
            "factor_run_id": factor_run_id,
            "source_snapshot_id": snapshot_id,
            "source_strategy_version": FACTOR_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "stock": {"symbol": symbol, "stock_name": row.get("name"), "industry": row.get("industry"), "market": row.get("market_segment")},
        "candidate_status": {
            "in_latest_candidate_pool": bool(row.get("tradable_flag")),
            "exclusion_reason": row.get("exclusion_reason"),
            "total_score": _float(row.get("total_score")),
        },
        "recent_behavior_summary": (
            f"Momentum score {_float(row.get('momentum_score')):.3f}, "
            f"flow score {_float(row.get('flow_score')):.3f}, risk score {_float(row.get('risk_score')):.3f}."
        ),
        "factor_values": _factor_values(row),
        "factor_definitions": factor_definitions(),
        "risk_notes": _risk_notes(row),
        "data_quality_notes": ["Uses the selected local factor run; verify freshness before publishing."],
        "manual_review_questions": [
            "Does recent behavior match the factor evidence?",
            "Are there company or industry events not captured by structured factors?",
            "Is data freshness sufficient for this analysis?",
        ],
    }


def _missing_payload(symbol: str, trade_date: str, factor_run_id: str, snapshot_id: str) -> dict[str, Any]:
    return {
        "analysis_metadata": {
            "schema_version": "stock_app_stock_analysis_v001",
            "symbol": symbol,
            "trade_date": trade_date,
            "factor_run_id": factor_run_id,
            "source_snapshot_id": snapshot_id,
            "source_strategy_version": FACTOR_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "stock": {"symbol": symbol, "stock_name": None, "industry": None, "market": None},
        "candidate_status": {"in_latest_candidate_pool": False, "exclusion_reason": "not_found_in_factor_run", "total_score": None},
        "recent_behavior_summary": "No factor row was found for this symbol and date.",
        "factor_values": [],
        "factor_definitions": factor_definitions(),
        "risk_notes": ["No current factor evidence is available for this symbol."],
        "data_quality_notes": ["The symbol may be absent, filtered out, or missing required data in the selected factor run."],
        "manual_review_questions": ["Confirm the symbol and selected report date.", "Check whether the local data snapshot includes this stock."],
    }


def _factor_values(row: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ["mom_20", "mom_60", "ma_strength", "flow_5d", "flow_20d", "winner_rate", "vol_20", "max_drawdown_60", "avg_amount_20", "momentum_score", "flow_score", "chip_score", "risk_score", "liquidity_score", "total_score"]
    return [{"factor_code": field, "value": _float(row.get(field)) if row.get(field) is not None else None, "missing_data": row.get(field) is None} for field in fields]


def _risk_notes(row: dict[str, Any]) -> list[str]:
    notes = []
    reason = row.get("exclusion_reason")
    if reason and reason != "none":
        notes.append(str(reason))
    if _float(row.get("risk_score"), 1.0) < 0.35:
        notes.append("Recent volatility or drawdown score is weak.")
    return notes or ["No blocking risk flag in the selected factor run."]


def _validate_stock_analysis(artifact: dict[str, Any]) -> list[str]:
    errors = []
    for section in ["analysis_metadata", "stock", "candidate_status", "factor_values", "risk_notes", "manual_review_questions"]:
        if section not in artifact:
            errors.append(f"missing section: {section}")
    return errors + validate_safe_payload(artifact)


def _float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
