"""Commercial app publish artifact builders."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.factor_exploration import FACTOR_VERSION


PUBLISH_SCHEMA_VERSION = "stock_app_publish_v001"
PROHIBITED_TERMS = ["买入", "卖出", "目标价", "仓位", "保证收益", "必涨", "buy", "sell", "target price"]


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    message: str
    artifact_path: Path | None = None
    artifact: dict[str, Any] | None = None


def factor_definitions() -> list[dict[str, Any]]:
    definitions = [
        ("mom_20", "20日动量", "近20个交易日调整后收盘价涨跌幅", "观察短期价格强弱", "短期动量可能反转", ["daily_prices.adj_close"], 10),
        ("mom_60", "60日动量", "近60个交易日调整后收盘价涨跌幅", "观察中期趋势", "上市时间短或停牌会影响覆盖", ["daily_prices.adj_close"], 20),
        ("ma_strength", "均线强度", "调整后收盘价相对20日均线的偏离", "观察价格是否强于近期均值", "过高偏离可能意味着波动风险", ["daily_prices.adj_close"], 30),
        ("flow_5d", "5日资金流", "近5日主力净流入比例累加", "观察短期资金参与", "资金流字段来自第三方数据源估算", ["capital_flow_or_chip.main_net_inflow_rate"], 40),
        ("flow_20d", "20日资金流", "近20日主力净流入比例累加", "观察阶段性资金参与", "不能单独代表真实持仓变化", ["capital_flow_or_chip.main_net_inflow_rate"], 50),
        ("flow_persistence", "资金持续性", "近10日主力净流入为正的比例", "观察资金流是否连续", "连续性不等于确定趋势", ["capital_flow_or_chip.main_net_inflow_rate"], 60),
        ("winner_rate", "获利筹码比例", "Tushare cyq_perf winner_rate 映射值", "观察当前价格附近筹码获利状态", "缺失或延迟时需要降权解读", ["capital_flow_or_chip.close_profit_ratio"], 70),
        ("winner_rate_change_5d", "5日获利筹码变化", "当前获利筹码比例减去5日前数值", "观察筹码状态变化", "筹码口径依赖数据供应商", ["capital_flow_or_chip.close_profit_ratio"], 80),
        ("vol_20", "20日波动率", "近20日收益率标准差", "观察短期波动风险", "低波动不代表低风险", ["daily_prices.adj_close"], 90),
        ("max_drawdown_60", "60日回撤", "当前价格相对60日高点的回撤", "观察近期回撤压力", "反弹和继续回撤都需要人工复核", ["daily_prices.adj_close"], 100),
        ("avg_amount_20", "20日成交额", "近20日成交额均值", "观察流动性基础", "成交额突然放大需要结合事件判断", ["daily_prices.amount"], 110),
        ("momentum_score", "动量得分", "由动量、均线强度和MACD信号组合", "汇总价格趋势证据", "是研究评分，不是交易指令", ["mom_20", "mom_60", "ma_strength"], 120),
        ("flow_score", "资金得分", "由短中期资金流和持续性组合", "汇总资金流证据", "资金数据存在估算误差", ["flow_5d", "flow_20d", "flow_persistence"], 130),
        ("chip_score", "筹码得分", "基于获利筹码比例与理想区间的距离", "观察筹码状态是否适中", "不适合单独使用", ["winner_rate"], 140),
        ("risk_score", "风险得分", "由波动、回撤和下行波动组合", "汇总近期风险状态", "风险得分只表示历史状态", ["vol_20", "max_drawdown_60"], 150),
        ("liquidity_score", "流动性得分", "由成交额和稳定性组合", "观察交易可承载性", "不能替代真实成交约束", ["avg_amount_20"], 160),
        ("total_score", "综合研究得分", "动量、资金、筹码、风险和流动性的加权组合", "用于候选池排序和人工复核优先级", "不是买卖建议或收益承诺", ["momentum_score", "flow_score", "chip_score", "risk_score", "liquidity_score"], 170),
    ]
    return [
        {
            "factor_code": code,
            "display_name": name,
            "calculation_rule_summary": rule,
            "interpretation": interpretation,
            "caveats": caveats,
            "source_fields": fields,
            "version": FACTOR_VERSION,
            "display_order": order,
            "status": "active",
        }
        for code, name, rule, interpretation, caveats, fields, order in definitions
    ]


def build_report_artifact(
    config_path: Path,
    factor_run_id: str,
    trade_date: str | None = None,
    previous_artifact_path: Path | None = None,
    output_path: Path | None = None,
    top: int = 20,
) -> PublishResult:
    if top <= 0:
        return PublishResult(False, "publish artifact failed; top must be positive")
    config = load_storage_config(config_path)
    run_dir = config.reports_root / "factor_exploration" / factor_run_id
    factors_path = run_dir / "stock_factors_daily.csv"
    metadata_path = run_dir / "factor_run_metadata.json"
    if not factors_path.exists() or not metadata_path.exists():
        return PublishResult(False, f"factor run not found: {factor_run_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        factors = pl.read_csv(factors_path, try_parse_dates=True)
    except Exception as error:
        return PublishResult(False, f"publish artifact failed; could not read factor CSV: {error}")
    if factors.is_empty():
        return PublishResult(False, "publish artifact failed; factor run has no rows")
    selected_date = trade_date or str(factors.select(pl.col("trade_date").max()).item())
    current = (
        factors.filter((pl.col("trade_date").cast(pl.Utf8) == selected_date) & pl.col("tradable_flag").fill_null(False))
        .sort("total_score", descending=True)
        .head(top)
        .with_row_index("rank", offset=1)
    )
    previous = _load_previous_candidates(previous_artifact_path)
    artifact = _artifact_payload(
        factor_run_id=factor_run_id,
        snapshot_id=str(metadata.get("snapshot_id", "")),
        trade_date=selected_date,
        current=current,
        previous=previous,
        warnings=_artifact_warnings(run_dir),
    )
    payload_without_hash = json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact["publish_metadata"]["artifact_hash"] = hashlib.sha256(payload_without_hash).hexdigest()
    validation_errors = validate_publish_artifact(artifact)
    if validation_errors:
        return PublishResult(False, "publish artifact failed validation: " + "; ".join(validation_errors), artifact=artifact)

    selected_output = output_path or run_dir / f"publish_artifact_{selected_date.replace('-', '')}.json"
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return PublishResult(True, f"publish_artifact: {selected_output}", selected_output, artifact)


def validate_publish_artifact(artifact: dict[str, Any]) -> list[str]:
    required = ["publish_metadata", "market_summary", "candidate_pool", "stock_cards", "factor_definitions", "factor_explanations", "backtest_summary", "data_quality"]
    errors = [f"missing section: {name}" for name in required if name not in artifact]
    if artifact.get("publish_metadata", {}).get("schema_version") != PUBLISH_SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    text = json.dumps(artifact, ensure_ascii=False).lower()
    for term in PROHIBITED_TERMS:
        if term.lower() in text:
            errors.append(f"prohibited wording: {term}")
    for forbidden in ["tushare_token", "provider token", "raw_payload"]:
        if forbidden in text:
            errors.append(f"forbidden sensitive field: {forbidden}")
    return errors


def _artifact_payload(
    factor_run_id: str,
    snapshot_id: str,
    trade_date: str,
    current: pl.DataFrame,
    previous: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    rows = current.to_dicts()
    candidates = [_candidate_row(row, previous) for row in rows]
    cards = [_stock_card(row) for row in rows]
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "publish_metadata": {
            "schema_version": PUBLISH_SCHEMA_VERSION,
            "publish_id": f"publish_{factor_run_id}_{trade_date.replace('-', '')}",
            "report_date": trade_date,
            "generated_at": generated_at,
            "source_snapshot_id": snapshot_id,
            "source_strategy_version": FACTOR_VERSION,
            "artifact_hash": None,
        },
        "market_summary": {
            "market_state": "research_update",
            "summary_text": f"{trade_date} Candidate 002 research pool generated for manual review.",
            "benchmark_context": "See local factor/backtest artifacts when available.",
        },
        "candidate_pool": {"items": candidates, "count": len(candidates), "change_summary": _change_summary(candidates)},
        "stock_cards": cards,
        "factor_definitions": factor_definitions(),
        "factor_explanations": _factor_explanations(rows),
        "backtest_summary": {"status": "available_when_generated", "summary_text": "Historical diagnostics are research context, not a performance promise."},
        "data_quality": {"status": "ok" if not warnings else "warning", "warnings": warnings, "freshness_date": trade_date},
    }


def _candidate_row(row: dict[str, Any], previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    symbol = str(row.get("symbol", ""))
    old = previous.get(symbol)
    rank = int(row.get("rank") or 0)
    change = "newly_added"
    if old:
        old_rank = int(old.get("rank") or rank)
        if rank < old_rank:
            change = "upgraded"
        elif rank > old_rank:
            change = "downgraded"
        else:
            change = "retained"
    return {
        "symbol": symbol,
        "stock_name": row.get("name"),
        "market": row.get("market_segment"),
        "industry": row.get("industry"),
        "rank": rank,
        "total_score": _float(row.get("total_score")),
        "change_status": change,
        "factor_labels": _labels(row),
        "risk_labels": _risk_labels(row),
    }


def _stock_card(row: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol", ""))
    name = row.get("name") or symbol
    return {
        "symbol": symbol,
        "title": f"{name} research card",
        "summary_text": "Candidate appears in the research pool based on factor evidence and data availability checks.",
        "inclusion_reasons": _labels(row),
        "factor_breakdown": _factor_breakdown(row),
        "recent_behavior_summary": _recent_behavior(row),
        "risk_notes": _risk_labels(row),
        "manual_review_questions": [
            "Is recent price behavior consistent with the factor evidence?",
            "Are there announcements, industry events, or liquidity changes that need manual review?",
            "Does the data-quality status look sufficient for further research?",
        ],
    }


def _factor_breakdown(row: dict[str, Any]) -> list[dict[str, Any]]:
    codes = ["momentum_score", "flow_score", "chip_score", "risk_score", "liquidity_score", "total_score"]
    return [{"factor_code": code, "score": _float(row.get(code)), "raw_value": _float(row.get(code)), "missing_data": row.get(code) is None} for code in codes]


def _factor_explanations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"symbol": str(row.get("symbol", "")), "items": _factor_breakdown(row)} for row in rows]


def _labels(row: dict[str, Any]) -> list[str]:
    labels = []
    if _float(row.get("momentum_score"), 0) >= 0.7:
        labels.append("momentum evidence")
    if _float(row.get("flow_score"), 0) >= 0.7:
        labels.append("capital-flow evidence")
    if _float(row.get("liquidity_score"), 0) >= 0.7:
        labels.append("liquidity support")
    return labels or ["factor evidence"]


def _risk_labels(row: dict[str, Any]) -> list[str]:
    labels = []
    reason = row.get("exclusion_reason")
    if reason and reason != "none":
        labels.append(str(reason))
    if _float(row.get("risk_score"), 1) < 0.35:
        labels.append("elevated recent volatility or drawdown")
    return labels or ["no blocking risk flag in current factor run"]


def _recent_behavior(row: dict[str, Any]) -> str:
    return (
        f"Momentum score {_float(row.get('momentum_score')):.3f}, "
        f"flow score {_float(row.get('flow_score')):.3f}, "
        f"liquidity score {_float(row.get('liquidity_score')):.3f}."
    )


def _change_summary(candidates: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for item in candidates:
        status = str(item.get("change_status"))
        summary[status] = summary.get(status, 0) + 1
    return summary


def _load_previous_candidates(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("candidate_pool", {}).get("items", [])
    return {str(item.get("symbol")): item for item in items if item.get("symbol")}


def _artifact_warnings(run_dir: Path) -> list[str]:
    warnings = []
    for expected in ["evaluation_summary.json", "candidate_002_backtest_weekly_metrics.json"]:
        if not (run_dir / expected).exists():
            warnings.append(f"optional diagnostic artifact missing: {expected}")
    return warnings


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
