"""Desktop workflow runtime for stock-picker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from stock_picker.analysis import analyze_stock
from stock_picker.config import load_storage_config
from stock_picker.factor_exploration import backtest_candidate_002, compute_daily_factors, rank_candidate_002
from stock_picker.publish import build_daily_bundle
from stock_picker.quality import check_curated_quality
from stock_picker.snapshot import create_snapshot
from stock_picker.sync import sync_latest


WorkflowEvent = dict[str, Any]
EventSink = Callable[[WorkflowEvent], None]


@dataclass(frozen=True)
class WorkflowResult:
    ok: bool
    message: str
    workflow_id: str
    state_path: Path | None = None


@dataclass(frozen=True)
class StepResult:
    ok: bool
    message: str


def workflow_status(config_path: Path, workflow_id: str | None = None) -> WorkflowResult:
    config = load_storage_config(config_path)
    root = _workflow_root(config)
    if workflow_id:
        state = _load_state(root / workflow_id / "state.json")
        return WorkflowResult(True, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), workflow_id, root / workflow_id / "state.json")
    items = []
    if root.exists():
        for path in sorted(root.glob("*/state.json"), key=lambda value: value.stat().st_mtime, reverse=True)[:20]:
            state = _load_state(path)
            items.append({key: state.get(key) for key in ["workflow_id", "type", "status", "current_step", "updated_at"]})
    return WorkflowResult(True, json.dumps({"workflows": items}, ensure_ascii=False, indent=2, sort_keys=True), "all", root)


def pause_workflow(config_path: Path, workflow_id: str) -> WorkflowResult:
    config = load_storage_config(config_path)
    state_path = _workflow_root(config) / workflow_id / "state.json"
    state = _load_state(state_path)
    state["status"] = "paused"
    state["updated_at"] = _now()
    _write_state(state_path, state)
    return WorkflowResult(True, f"workflow paused: {workflow_id}", workflow_id, state_path)


def sync_report_workflow(
    config_path: Path,
    workflow_id: str | None = None,
    dry_run: bool = True,
    confirmed: bool = False,
    factor_run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top: int = 20,
    emit: EventSink | None = None,
) -> WorkflowResult:
    config = load_storage_config(config_path)
    selected_id = workflow_id or f"sync_report_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    state_path = _workflow_root(config) / selected_id / "state.json"
    state = _load_or_create_state(state_path, selected_id, "sync_report")
    if state.get("status") == "paused":
        state["status"] = "pending"
    _event(emit, state, "workflow_started", "sync report workflow started")

    if dry_run or not confirmed:
        result = _run_step(state_path, state, "sync_dry_run", emit, lambda: sync_latest(config_path=config_path, start_date=start_date, end_date=end_date, dry_run=True))
        status = "completed" if result.ok else "failed"
        state["status"] = status
        state["requires_confirmation"] = bool(result.ok)
        _write_state(state_path, state)
        return WorkflowResult(result.ok, result.message, selected_id, state_path)

    sync_result = _run_step(state_path, state, "sync_latest", emit, lambda: sync_latest(config_path=config_path, start_date=start_date, end_date=end_date, dry_run=False, create_snapshot_after=True))
    if not sync_result.ok:
        return _fail(selected_id, state_path, sync_result.message, state, emit)
    quality = _run_step(state_path, state, "quality_check", emit, lambda: check_curated_quality(config_path))
    if not quality.ok:
        return _fail(selected_id, state_path, quality.message, state, emit)
    snapshot_id = _latest_snapshot_id(config.metadata_sqlite_path)
    if not snapshot_id:
        snapshot = _run_step(state_path, state, "create_snapshot", emit, lambda: create_snapshot(config_path, as_of_date=end_date or datetime.now(UTC).date().isoformat()))
        if not snapshot.ok:
            return _fail(selected_id, state_path, snapshot.message, state, emit)
        snapshot_id = _latest_snapshot_id(config.metadata_sqlite_path)
    selected_factor_run = factor_run_id or f"factor_002_{snapshot_id}"
    compute = _run_step(
        state_path,
        state,
        "compute_candidate_002_factors",
        emit,
        lambda: compute_daily_factors(config_path, snapshot_id=snapshot_id, start_date=start_date or "2026-01-01", end_date=end_date or datetime.now(UTC).date().isoformat(), run_id=selected_factor_run),
    )
    if not compute.ok:
        return _fail(selected_id, state_path, compute.message, state, emit)
    rank = _run_step(state_path, state, "rank_candidate_002", emit, lambda: rank_candidate_002(config_path, selected_factor_run, trade_date=end_date, top=top))
    if not rank.ok:
        return _fail(selected_id, state_path, rank.message, state, emit)
    backtest = _run_step(state_path, state, "backtest_candidate_002", emit, lambda: backtest_candidate_002(config_path, selected_factor_run, top=min(10, top), rebalance="weekly"))
    if not backtest.ok:
        return _fail(selected_id, state_path, backtest.message, state, emit)
    publish = _run_step(state_path, state, "build_daily_bundle", emit, lambda: build_daily_bundle(config_path, selected_factor_run, trade_date=end_date, top=top))
    if not publish.ok:
        return _fail(selected_id, state_path, publish.message, state, emit)
    state["status"] = "completed"
    state["artifacts"]["daily_bundle"] = str(getattr(publish, "artifact_path", "") or "")
    state["updated_at"] = _now()
    _write_state(state_path, state)
    _event(emit, state, "workflow_completed", "sync report workflow completed")
    return WorkflowResult(True, "sync report workflow completed", selected_id, state_path)


def stock_analysis_workflow(
    config_path: Path,
    symbol: str,
    factor_run_id: str,
    workflow_id: str | None = None,
    trade_date: str | None = None,
    emit: EventSink | None = None,
) -> WorkflowResult:
    config = load_storage_config(config_path)
    selected_id = workflow_id or f"stock_analysis_{symbol.replace('.', '_')}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    state_path = _workflow_root(config) / selected_id / "state.json"
    state = _load_or_create_state(state_path, selected_id, "stock_analysis")
    if state.get("status") == "paused":
        state["status"] = "pending"
    result = _run_step(state_path, state, "analyze_stock", emit, lambda: analyze_stock(config_path, symbol=symbol, factor_run_id=factor_run_id, trade_date=trade_date))
    state["status"] = "completed" if result.ok else "failed"
    state["updated_at"] = _now()
    if getattr(result, "artifact_path", None):
        state["artifacts"]["stock_analysis"] = str(result.artifact_path)
    _write_state(state_path, state)
    return WorkflowResult(result.ok, result.message, selected_id, state_path)


def _run_step(state_path: Path, state: dict[str, Any], name: str, emit: EventSink | None, fn):
    if state.get("status") == "paused":
        raise RuntimeError(f"workflow paused before step: {name}")
    if state.get("steps", {}).get(name, {}).get("status") == "completed":
        _event(emit, state, "step_skipped", f"already completed: {name}")
        return StepResult(True, f"step already completed: {name}")
    state["status"] = "running"
    state["current_step"] = name
    state["steps"].setdefault(name, {"status": "pending"})
    state["steps"][name].update({"status": "running", "started_at": _now()})
    state["updated_at"] = _now()
    _write_state(state_path, state)
    _event(emit, state, "step_started", name)
    try:
        result = fn()
    except Exception as error:
        state["steps"][name].update({"status": "failed", "finished_at": _now(), "message": str(error)})
        state["errors"].append(str(error))
        _write_state(state_path, state)
        _event(emit, state, "step_failed", str(error))
        raise
    state["steps"][name].update({"status": "completed" if result.ok else "failed", "finished_at": _now(), "message": result.message})
    if not result.ok:
        state["errors"].append(result.message)
    state["updated_at"] = _now()
    _write_state(state_path, state)
    _event(emit, state, "step_completed" if result.ok else "step_failed", result.message)
    return result


def _fail(workflow_id: str, state_path: Path, message: str, state: dict[str, Any], emit: EventSink | None) -> WorkflowResult:
    state["status"] = "failed"
    state["updated_at"] = _now()
    _write_state(state_path, state)
    _event(emit, state, "workflow_failed", message)
    return WorkflowResult(False, message, workflow_id, state_path)


def _workflow_root(config) -> Path:
    return config.reports_root / "desktop_workflows"


def _load_or_create_state(path: Path, workflow_id: str, workflow_type: str) -> dict[str, Any]:
    if path.exists():
        return _load_state(path)
    return {
        "workflow_id": workflow_id,
        "type": workflow_type,
        "status": "pending",
        "current_step": None,
        "steps": {},
        "artifacts": {},
        "warnings": [],
        "errors": [],
        "created_at": _now(),
        "updated_at": _now(),
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "workflow_id": path.parent.name, "steps": {}, "artifacts": {}, "warnings": [], "errors": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _event(emit: EventSink | None, state: dict[str, Any], event: str, message: str) -> None:
    if emit:
        emit({"event": event, "workflow_id": state.get("workflow_id"), "status": state.get("status"), "step": state.get("current_step"), "message": message, "timestamp": _now()})


def _latest_snapshot_id(sqlite_path: Path) -> str | None:
    import sqlite3

    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute("SELECT snapshot_id FROM snapshot_manifests ORDER BY created_at DESC, snapshot_id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
