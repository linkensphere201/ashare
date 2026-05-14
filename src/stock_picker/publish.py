"""Commercial app daily bundle builders."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from stock_picker.config import load_storage_config
from stock_picker.factor_exploration import FACTOR_VERSION


MARKET_STATUS_SCHEMA_VERSION = "market_status_v001"
CANDIDATE_POOL_SCHEMA_VERSION = "candidate_pool_v001"
DAILY_BUNDLE_SCHEMA_VERSION = "daily_publish_bundle_v001"
STOCK_ANALYSIS_SCHEMA_VERSION = "stock_app_stock_analysis_v001"
PROHIBITED_TERMS = ["买入", "卖出", "加仓", "减仓", "目标价", "仓位", "保证收益", "必涨", "buy", "sell", "target price"]
SENSITIVE_TERMS = ["tushare_token", "provider token", "raw_payload", "raw provider"]
WINDOWS_ABSOLUTE_PATH = re.compile(r"[a-zA-Z]:[\\/]")

BROAD_INDEXES = {
    "000300.SH": {"name": "沪深300", "role": "大盘核心资产"},
    "000905.SH": {"name": "中证500", "role": "中盘"},
    "000852.SH": {"name": "中证1000", "role": "小盘"},
    "399006.SZ": {"name": "创业板指", "role": "成长"},
    "000688.SH": {"name": "科创50", "role": "硬科技成长"},
    "899050.BJ": {"name": "北证50", "role": "北交所小微高波动"},
}

SECTOR_THEME = {
    "电子": "科技成长",
    "计算机": "科技成长",
    "通信": "科技成长",
    "传媒": "科技成长",
    "电力设备": "高端制造",
    "机械设备": "高端制造",
    "国防军工": "高端制造",
    "汽车": "高端制造",
    "煤炭": "资源周期",
    "有色金属": "资源周期",
    "钢铁": "资源周期",
    "石油石化": "资源周期",
    "基础化工": "资源周期",
    "银行": "金融地产",
    "非银金融": "金融地产",
    "房地产": "金融地产",
    "建筑材料": "金融地产",
    "建筑装饰": "金融地产",
    "食品饮料": "消费医药",
    "商贸零售": "消费医药",
    "社会服务": "消费医药",
    "美容护理": "消费医药",
    "医药生物": "消费医药",
    "公用事业": "防御高股息",
    "交通运输": "防御高股息",
    "环保": "防御高股息",
}


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    message: str
    artifact_path: Path | None = None
    artifact: dict[str, Any] | None = None


def factor_definitions() -> list[dict[str, Any]]:
    components = [
        ("momentum_score", "价格趋势", "观察短期和中期价格表现及均线状态。"),
        ("flow_score", "资金活跃度", "观察近期资金流入和持续性。"),
        ("chip_score", "筹码状态", "观察获利筹码比例及变化。"),
        ("risk_score", "风险波动", "观察近期波动和回撤压力。"),
        ("liquidity_score", "流动性", "过滤流动性不足或成交承载较弱的股票。"),
        ("total_score", "综合研究得分", "汇总多维研究证据并用于候选池排序。"),
    ]
    return [
        {
            "factor_code": code,
            "display_name": name,
            "calculation_rule_summary": summary,
            "interpretation": summary,
            "caveats": "研究排序依据，不是交易指令。",
            "source_fields": [],
            "version": FACTOR_VERSION,
            "display_order": index * 10,
            "status": "active",
        }
        for index, (code, name, summary) in enumerate(components, start=1)
    ]


def build_market_status(
    config_path: Path,
    trade_date: str | None = None,
    output_path: Path | None = None,
) -> PublishResult:
    config = load_storage_config(config_path)
    selected_date = trade_date
    generated_at = _now()
    index_metrics, index_warnings = _load_index_metrics(config.current_curated_root / "daily_prices" / "part-000.parquet", selected_date)
    if index_metrics and not selected_date:
        selected_date = str(index_metrics[0]["trade_date"])
    sector_metrics, sector_warnings = _load_sector_metrics(
        config.current_curated_root / "industry_daily" / "part-000.parquet",
        config.current_curated_root / "industry_classification" / "part-000.parquet",
        selected_date,
    )
    selected_date = selected_date or datetime.now(UTC).date().isoformat()

    strongest_index, weakest_index = _strongest_weakest(index_metrics)
    strongest_sectors = sorted(sector_metrics, key=lambda row: _float(row.get("pct_change")), reverse=True)[:3]
    weakest_sectors = sorted(sector_metrics, key=lambda row: _float(row.get("pct_change")))[:3]
    warnings = index_warnings + sector_warnings
    status = "complete" if not warnings else ("partial" if index_metrics or sector_metrics else "unavailable")
    payload = {
        "market_status_metadata": {
            "schema_version": MARKET_STATUS_SCHEMA_VERSION,
            "trade_date": selected_date,
            "generated_at": generated_at,
            "status": status,
            "data_sources": ["tushare.index_daily", "tushare.index_classify", "tushare.sw_daily"],
            "warnings": warnings,
        },
        "index_markets": {
            "strongest": _index_payload(strongest_index, index_metrics, "strongest"),
            "weakest": _index_payload(weakest_index, index_metrics, "weakest"),
        },
        "sectors": {
            "strongest": [_sector_name(row) for row in strongest_sectors],
            "weakest": [_sector_name(row) for row in weakest_sectors],
            "summary_text": _sector_summary(sector_metrics, strongest_sectors, weakest_sectors),
        },
        "structural_risks": _structural_risks(index_metrics, strongest_sectors, weakest_sectors, warnings),
        "internal_diagnostics": {
            "visible_to_client": False,
            "index_metrics": index_metrics,
            "sector_metrics": sector_metrics,
            "risk_triggers": [],
            "template_rule_ids": [],
        },
    }
    errors = validate_market_status(payload)
    if errors:
        return PublishResult(False, "market status failed validation: " + "; ".join(errors), artifact=payload)
    path = output_path or config.reports_root / "market_status" / f"market_status_{selected_date.replace('-', '')}.json"
    _write_json(path, payload)
    return PublishResult(True, f"market_status: {path}", path, payload)


def build_candidate_pool(
    config_path: Path,
    factor_run_id: str,
    trade_date: str | None = None,
    previous_candidate_pool_path: Path | None = None,
    previous_bundle_path: Path | None = None,
    output_path: Path | None = None,
    top: int = 10,
) -> PublishResult:
    if top <= 0:
        return PublishResult(False, "candidate pool failed; top must be positive")
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
        return PublishResult(False, f"candidate pool failed; could not read factor CSV: {error}")
    if factors.is_empty():
        return PublishResult(False, "candidate pool failed; factor run has no rows")
    selected_date = trade_date or str(factors.select(pl.col("trade_date").max()).item())
    current = (
        factors.filter((pl.col("trade_date").cast(pl.Utf8) == selected_date) & pl.col("tradable_flag").fill_null(False))
        .sort("total_score", descending=True)
        .head(top)
        .with_row_index("rank", offset=1)
    )
    if current.is_empty():
        return PublishResult(False, f"candidate pool failed; no tradable rows for {selected_date}")

    previous_items = _load_previous_candidate_items(previous_candidate_pool_path, previous_bundle_path)
    top_stocks = [_candidate_stock(row, previous_items) for row in current.to_dicts()]
    diff = _candidate_diff(top_stocks, previous_items)
    version_id = f"candidate_pool_{selected_date.replace('-', '')}_001"
    payload = {
        "candidate_pool_metadata": {
            "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
            "version_id": version_id,
            "trade_date": selected_date,
            "generated_at": _now(),
            "source_strategy": "candidate_002",
            "strategy_version": str(metadata.get("factor_version") or FACTOR_VERSION),
            "source_snapshot_id": str(metadata.get("snapshot_id", "")),
            "previous_version_id": diff.get("previous_version_id"),
            "status": "complete" if diff.get("status") == "compared" else "no_previous_version",
        },
        "strategy": _candidate_strategy(),
        "summary": _diff_summary(diff, len(top_stocks)),
        "top_stocks": top_stocks,
        "diff": diff,
        "internal_diagnostics": {
            "visible_to_client": False,
            "factor_run_id": factor_run_id,
            "score_columns": ["momentum_score", "flow_score", "chip_score", "risk_score", "liquidity_score", "total_score"],
        },
    }
    errors = validate_candidate_pool(payload)
    if errors:
        return PublishResult(False, "candidate pool failed validation: " + "; ".join(errors), artifact=payload)
    path = output_path or config.reports_root / "candidate_pools" / f"{version_id}.json"
    _write_json(path, payload)
    return PublishResult(True, f"candidate_pool: {path}", path, payload)


def build_daily_bundle(
    config_path: Path,
    factor_run_id: str,
    trade_date: str | None = None,
    previous_bundle_path: Path | None = None,
    previous_candidate_pool_path: Path | None = None,
    output_path: Path | None = None,
    top: int = 10,
) -> PublishResult:
    config = load_storage_config(config_path)
    selected_date = trade_date or _latest_factor_trade_date(config, factor_run_id)
    market = build_market_status(config_path, trade_date=selected_date)
    if not market.ok:
        return market
    candidate = build_candidate_pool(
        config_path,
        factor_run_id=factor_run_id,
        trade_date=selected_date or market.artifact["market_status_metadata"]["trade_date"],
        previous_candidate_pool_path=previous_candidate_pool_path,
        previous_bundle_path=previous_bundle_path,
        top=top,
    )
    if not candidate.ok:
        return candidate
    selected_date = candidate.artifact["candidate_pool_metadata"]["trade_date"]
    payload = {
        "bundle_metadata": {
            "schema_version": DAILY_BUNDLE_SCHEMA_VERSION,
            "bundle_id": f"daily_bundle_{selected_date.replace('-', '')}_001",
            "trade_date": selected_date,
            "generated_at": _now(),
            "source_system": "stock-picker",
            "source_snapshot_id": candidate.artifact["candidate_pool_metadata"].get("source_snapshot_id"),
            "bundle_hash": None,
            "contains": [MARKET_STATUS_SCHEMA_VERSION, CANDIDATE_POOL_SCHEMA_VERSION],
        },
        "market_status": market.artifact,
        "candidate_pool": candidate.artifact,
    }
    payload["bundle_metadata"]["bundle_hash"] = _payload_hash(payload)
    errors = validate_daily_bundle(payload)
    if errors:
        return PublishResult(False, "daily bundle failed validation: " + "; ".join(errors), artifact=payload)
    path = output_path or config.reports_root / "daily_bundles" / f"{payload['bundle_metadata']['bundle_id']}.json"
    _write_json(path, payload)
    return PublishResult(True, f"daily_bundle: {path}", path, payload)


def _latest_factor_trade_date(config, factor_run_id: str) -> str | None:
    factors_path = config.reports_root / "factor_exploration" / factor_run_id / "stock_factors_daily.csv"
    if not factors_path.exists():
        return None
    try:
        factors = pl.read_csv(factors_path, try_parse_dates=True)
    except Exception:
        return None
    if factors.is_empty() or "trade_date" not in factors.columns:
        return None
    if "tradable_flag" in factors.columns:
        tradable = factors.filter(pl.col("tradable_flag").fill_null(False))
        if not tradable.is_empty():
            factors = tradable
    return str(factors.select(pl.col("trade_date").max()).item())


def validate_market_status(payload: dict[str, Any]) -> list[str]:
    errors = _required(payload, ["market_status_metadata", "index_markets", "sectors", "structural_risks"])
    if payload.get("market_status_metadata", {}).get("schema_version") != MARKET_STATUS_SCHEMA_VERSION:
        errors.append("unsupported market_status schema_version")
    return errors + validate_safe_payload(payload)


def validate_candidate_pool(payload: dict[str, Any]) -> list[str]:
    errors = _required(payload, ["candidate_pool_metadata", "strategy", "summary", "top_stocks", "diff"])
    if payload.get("candidate_pool_metadata", {}).get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
        errors.append("unsupported candidate_pool schema_version")
    return errors + validate_safe_payload(payload)


def validate_daily_bundle(payload: dict[str, Any]) -> list[str]:
    errors = _required(payload, ["bundle_metadata", "market_status", "candidate_pool"])
    metadata = payload.get("bundle_metadata", {})
    if metadata.get("schema_version") != DAILY_BUNDLE_SCHEMA_VERSION:
        errors.append("unsupported bundle schema_version")
    expected_hash = metadata.get("bundle_hash")
    if expected_hash and expected_hash != _payload_hash(payload):
        errors.append("bundle_hash mismatch")
    errors += validate_market_status(payload.get("market_status", {}))
    errors += validate_candidate_pool(payload.get("candidate_pool", {}))
    return errors + validate_safe_payload(payload)


def validate_safe_payload(payload: dict[str, Any]) -> list[str]:
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    errors = []
    for term in PROHIBITED_TERMS:
        if term.lower() in text:
            errors.append(f"prohibited wording: {term}")
    for term in SENSITIVE_TERMS:
        if term in text:
            errors.append(f"forbidden sensitive field: {term}")
    if WINDOWS_ABSOLUTE_PATH.search(text):
        errors.append("forbidden local absolute path")
    return errors


def _load_index_metrics(path: Path, trade_date: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], ["missing daily_prices current data"]
    frame = pl.read_parquet(path)
    if "asset_type" in frame.columns:
        frame = frame.filter(pl.col("asset_type") == "index")
    frame = frame.filter(pl.col("symbol").is_in(list(BROAD_INDEXES)))
    if frame.is_empty():
        return [], ["missing broad index rows"]
    selected = trade_date or str(frame.select(pl.col("trade_date").max()).item())
    latest = frame.filter(pl.col("trade_date").cast(pl.Utf8) == selected)
    rows = []
    for symbol, meta in BROAD_INDEXES.items():
        index_rows = frame.filter(pl.col("symbol") == symbol).sort("trade_date")
        current = latest.filter(pl.col("symbol") == symbol)
        if current.is_empty():
            continue
        row = current.to_dicts()[0]
        rows.append(
            {
                "code": symbol,
                "name": meta["name"],
                "role": meta["role"],
                "trade_date": selected,
                "return_1d": _float(row.get("pct_change")),
                "return_5d": _period_return(index_rows, selected, 5),
                "return_20d": _period_return(index_rows, selected, 20),
            }
        )
    return rows, [] if rows else ["missing selected broad index rows"]


def _load_sector_metrics(path: Path, classification_path: Path, trade_date: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], ["missing industry_daily current data"]
    if not classification_path.exists():
        return [], ["missing industry_classification current data"]
    frame = pl.read_parquet(path)
    classification = pl.read_parquet(classification_path)
    if frame.is_empty():
        return [], ["industry_daily has no rows"]
    if classification.is_empty():
        return [], ["industry_classification has no rows"]
    l1 = classification.filter((pl.col("level") == "L1") & (pl.col("source_system") == "SW2021")).select(["index_code", "industry_name"])
    if l1.is_empty():
        return [], ["missing SW2021 L1 industry classification rows"]
    selected = trade_date or str(frame.select(pl.col("trade_date").max()).item())
    rows = (
        frame.filter(pl.col("trade_date").cast(pl.Utf8) == selected)
        .drop("industry_name")
        .join(l1, on="index_code", how="inner")
        .to_dicts()
    )
    return rows, [] if rows else ["missing selected Shenwan L1 sector rows"]


def _strongest_weakest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not rows:
        return None, None
    sorted_rows = sorted(rows, key=lambda row: _float(row.get("return_1d")), reverse=True)
    return sorted_rows[0], sorted_rows[-1]


def _index_payload(row: dict[str, Any] | None, all_rows: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "code": row["code"],
        "name": row["name"],
        "role": row["role"],
        "return_1d": _float(row.get("return_1d")),
        "reason": _index_reason(row, all_rows, kind),
    }


def _index_reason(row: dict[str, Any], all_rows: list[dict[str, Any]], kind: str) -> str:
    role = str(row.get("role"))
    positives = sum(1 for item in all_rows if _float(item.get("return_1d")) > 0)
    negatives = sum(1 for item in all_rows if _float(item.get("return_1d")) < 0)
    if kind == "strongest":
        if negatives == len(all_rows):
            return f"虽然仍然下跌，但在主要宽基中跌幅最小，{role}方向相对抗跌。"
        if positives == len(all_rows):
            return f"在主要宽基普遍上涨的情况下涨幅领先，{role}方向弹性更强。"
        if _rank(row, all_rows, "return_5d", reverse=True) <= 2:
            return f"近 1 日和 5 日表现均领先主要宽基指数，{role}方向相对占优。"
        return f"今日表现领先主要宽基指数，{role}方向出现阶段性修复。"
    if positives == len(all_rows):
        return f"虽然仍然上涨，但在主要宽基中涨幅最小，{role}方向弹性不足。"
    if negatives == len(all_rows):
        return f"在主要宽基普遍下跌的情况下跌幅更明显，{role}方向承压更重。"
    if _rank(row, all_rows, "return_5d", reverse=True) >= max(1, len(all_rows) - 1):
        return f"近 1 日和 5 日表现均落后主要宽基指数，{role}方向短期承压。"
    return f"今日表现落后主要宽基指数，{role}方向短期出现回落。"


def _structural_risks(
    indexes: list[dict[str, Any]],
    strong_sectors: list[dict[str, Any]],
    weak_sectors: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    risks = []
    if warnings:
        risks.append(_risk("data_insufficient", "数据覆盖不足", "部分市场状态数据不足，本页仅展示可确认部分。", "low", {}, warnings))
    index_down_ratio = _down_ratio(indexes, "return_1d")
    sector_down_ratio = _down_ratio(strong_sectors + weak_sectors, "pct_change")
    weak_names = [_sector_name(row) for row in weak_sectors]
    weak_themes = _themes(weak_names)
    strong_names = [_sector_name(row) for row in strong_sectors]
    strong_themes = _themes(strong_names)
    if index_down_ratio >= 0.67 and sector_down_ratio >= 0.60:
        risks.append(_risk("general_weakness", "主要市场同步走弱", "主要宽基和多数行业同步走弱，短期环境偏谨慎。", "high", {"themes": list(set(weak_themes))}, []))
    if any(row.get("code") in {"399006.SZ", "000688.SH"} for row in sorted(indexes, key=lambda item: _float(item.get("return_1d")))[:2]) or "科技成长" in weak_themes:
        risks.append(_risk("growth_pressure", "成长方向承压", "创业板、科创或科技成长方向表现靠后，成长风格短期承压。", "medium", {"market_segments": ["创业板", "科创板"], "themes": ["科技成长"], "industries_l1": ["电子", "计算机", "通信", "传媒"]}, []))
    if any(row.get("code") in {"000852.SH", "899050.BJ"} for row in sorted(indexes, key=lambda item: _float(item.get("return_1d")))[:2]):
        risks.append(_risk("small_cap_pressure", "小盘方向承压", "中证1000或北证50表现靠后，小盘和高波动方向短期承压。", "medium", {"market_segments": ["北交所"], "themes": ["小盘高波动"]}, []))
    if strong_sectors and weak_sectors and _float(strong_sectors[0].get("pct_change")) - _float(weak_sectors[0].get("pct_change")) >= 2.0:
        risks.append(_risk("sector_divergence", "行业分化扩大", "强弱行业表现差距较大，持仓需要结合所属行业和主题进行复核。", "medium", {"industries_l1": weak_names, "themes": list(set(weak_themes))}, []))
    if weak_themes and weak_themes.count(weak_themes[0]) >= 2:
        risks.append(_risk("weak_theme_concentration", f"{weak_themes[0]}压力集中", f"弱势压力集中在{weak_themes[0]}，需要关注板块共振风险。", "medium", {"themes": [weak_themes[0]], "industries_l1": weak_names}, []))
    return risks[:3]


def _risk(risk_type: str, title: str, summary: str, severity: str, match_rules: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {"risk_type": risk_type, "title": title, "summary": summary, "severity": severity, "match_rules": match_rules, "warnings": warnings}


def _candidate_strategy() -> dict[str, Any]:
    return {
        "strategy_id": "candidate_002",
        "strategy_name": "因子002",
        "summary": "因子002综合考虑价格趋势、资金活跃度、筹码状态、风险波动和流动性条件，对可交易股票进行标准化评分，并按综合研究得分筛选出前 10 只候选股票。",
        "score_components": [
            {"name": "价格趋势", "description": "观察短期和中期价格表现及均线状态。"},
            {"name": "资金活跃度", "description": "观察近期资金流入和持续性。"},
            {"name": "筹码状态", "description": "观察获利筹码比例及变化。"},
            {"name": "风险波动", "description": "观察近期波动和回撤压力。"},
            {"name": "流动性", "description": "过滤流动性不足或成交承载较弱的股票。"},
        ],
        "selection_method": "按综合研究得分排序，取 Top 10。",
    }


def _candidate_stock(row: dict[str, Any], previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    symbol = str(row.get("symbol"))
    industry = row.get("industry")
    total_score = _float(row.get("total_score"))
    rank = int(row.get("rank") or 0)
    previous_item = previous.get(symbol)
    previous_rank = int(previous_item.get("rank")) if previous_item and previous_item.get("rank") else None
    payload = {
        "rank": rank,
        "current_rank": rank,
        "symbol": symbol,
        "name": row.get("name"),
        "industry_l1": industry,
        "theme": SECTOR_THEME.get(str(industry), "其他"),
        "change_status": _change_status(rank, previous_item),
        "reason_summary": _reason_summary(row),
        "risk_summary": _risk_summary(row),
        "score": {"visible_to_client": False, "total_score": total_score},
    }
    if previous_rank is not None:
        payload["previous_rank"] = previous_rank
    return payload


def _candidate_diff(current: list[dict[str, Any]], previous: dict[str, dict[str, Any]]) -> dict[str, Any]:
    previous_version = next((item.get("version_id") for item in previous.values() if item.get("version_id")), None)
    diff = {"previous_version_id": previous_version, "status": "compared" if previous else "no_previous_version", "newly_added": [], "retained": [], "upgraded": [], "downgraded": [], "removed": []}
    current_symbols = {item["symbol"] for item in current}
    for item in current:
        diff[str(item["change_status"])].append(item)
    for symbol, old in previous.items():
        if symbol not in current_symbols:
            diff["removed"].append({"symbol": symbol, "name": old.get("name") or old.get("stock_name"), "previous_rank": old.get("rank"), "removed_reason": "本期综合排序未进入 Top 10。"})
    return diff


def _load_previous_candidate_items(candidate_path: Path | None, bundle_path: Path | None) -> dict[str, dict[str, Any]]:
    path = candidate_path or bundle_path
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "candidate_pool" in payload:
        payload = payload["candidate_pool"]
    items = payload.get("top_stocks") or payload.get("items") or []
    version_id = payload.get("candidate_pool_metadata", {}).get("version_id")
    previous = {}
    for item in items:
        if item.get("symbol"):
            copied = dict(item)
            copied["version_id"] = version_id
            previous[str(item["symbol"])] = copied
    return previous


def _change_status(rank: int, old: dict[str, Any] | None) -> str:
    if not old:
        return "newly_added"
    previous_rank = int(old.get("rank") or rank)
    if previous_rank - rank >= 2:
        return "upgraded"
    if rank - previous_rank >= 2:
        return "downgraded"
    return "retained"


def _diff_summary(diff: dict[str, Any], total: int) -> dict[str, int]:
    return {key: len(diff.get(key, [])) for key in ["newly_added", "retained", "upgraded", "downgraded", "removed"]} | {"total": total}


def _required(payload: dict[str, Any], sections: list[str]) -> list[str]:
    return [f"missing section: {section}" for section in sections if section not in payload]


def _payload_hash(payload: dict[str, Any]) -> str:
    copied = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    if "bundle_metadata" in copied:
        copied["bundle_metadata"].pop("bundle_hash", None)
    return hashlib.sha256(json.dumps(copied, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _period_return(frame: pl.DataFrame, selected_date: str, days: int) -> float | None:
    rows = frame.filter(pl.col("trade_date").cast(pl.Utf8) <= selected_date).tail(days).to_dicts()
    if len(rows) < 2:
        return None
    first = _float(rows[0].get("close"))
    last = _float(rows[-1].get("close"))
    return None if first == 0 else (last / first - 1) * 100


def _rank(row: dict[str, Any], rows: list[dict[str, Any]], field: str, reverse: bool) -> int:
    ranked = sorted(rows, key=lambda item: _float(item.get(field)), reverse=reverse)
    return next((index for index, item in enumerate(ranked, start=1) if item.get("code") == row.get("code")), len(rows))


def _sector_name(row: dict[str, Any]) -> str:
    return str(row.get("industry_name") or row.get("name") or row.get("index_code"))


def _sector_summary(rows: list[dict[str, Any]], strong: list[dict[str, Any]], weak: list[dict[str, Any]]) -> str:
    if not rows:
        return "申万一级行业数据不足，暂无法形成板块摘要。"
    up_ratio = sum(1 for row in rows if _float(row.get("pct_change")) > 0) / len(rows)
    strong_themes = ", ".join(sorted(set(_themes([_sector_name(row) for row in strong]))))
    weak_themes = ", ".join(sorted(set(_themes([_sector_name(row) for row in weak]))))
    if up_ratio >= 0.7:
        return f"申万一级行业多数上涨，强势方向主要集中在{strong_themes or '部分强势行业'}。"
    if up_ratio <= 0.3:
        return f"申万一级行业多数走弱，强势板块更多体现为相对抗跌，弱势压力集中在{weak_themes or '部分弱势行业'}。"
    if strong and weak and _float(strong[0].get("pct_change")) - _float(weak[0].get("pct_change")) >= 2.0:
        return f"行业表现分化明显，强势方向集中在{strong_themes or '部分强势行业'}，弱势方向集中在{weak_themes or '部分弱势行业'}。"
    return "行业表现相对均衡，强弱方向暂未形成清晰主线。"


def _themes(names: list[str]) -> list[str]:
    return [SECTOR_THEME.get(name, "其他") for name in names]


def _down_ratio(rows: list[dict[str, Any]], field: str) -> float:
    return 0.0 if not rows else sum(1 for row in rows if _float(row.get(field)) < 0) / len(rows)


def _reason_summary(row: dict[str, Any]) -> str:
    labels = []
    if _float(row.get("momentum_score")) >= 0.7:
        labels.append("价格趋势")
    if _float(row.get("flow_score")) >= 0.7:
        labels.append("资金活跃度")
    if _float(row.get("liquidity_score")) >= 0.7:
        labels.append("流动性")
    return f"{'、'.join(labels) if labels else '综合研究证据'}相对靠前，进入本期候选池。"


def _risk_summary(row: dict[str, Any]) -> str:
    reason = row.get("exclusion_reason")
    if reason and reason != "none":
        return f"存在{reason}风险提示，需要人工复核。"
    if _float(row.get("risk_score"), 1.0) < 0.35:
        return "近期波动或回撤压力较高，需要人工复核。"
    return "暂无阻断性风险提示。"


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
