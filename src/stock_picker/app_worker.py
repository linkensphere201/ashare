"""Local app worker client and runner."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_picker.analysis import analyze_stock
from stock_picker.config import load_storage_config
from stock_picker.publish import build_report_artifact


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class WorkerConfig:
    app_base_url: str
    worker_id: str
    worker_token_env: str
    poll_interval_seconds: float
    default_factor_run_id: str | None = None
    default_trade_date: str | None = None


def load_worker_config(path: Path) -> WorkerConfig:
    if not path.exists():
        return WorkerConfig("http://127.0.0.1:3000", "local-worker", "STOCK_APP_WORKER_TOKEN", 15.0)
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        data = _simple_yaml(path)
    return WorkerConfig(
        app_base_url=str(data.get("app_base_url", "http://127.0.0.1:3000")).rstrip("/"),
        worker_id=str(data.get("worker_id", "local-worker")),
        worker_token_env=str(data.get("worker_token_env", "STOCK_APP_WORKER_TOKEN")),
        poll_interval_seconds=float(data.get("poll_interval_seconds", 15)),
        default_factor_run_id=data.get("default_factor_run_id"),
        default_trade_date=data.get("default_trade_date"),
    )


def run_worker_once(config_path: Path, worker_config_path: Path, mock_task_path: Path | None = None) -> WorkerResult:
    storage_config = load_storage_config(config_path)
    worker_config = load_worker_config(worker_config_path)
    try:
        task = _load_mock_task(mock_task_path) if mock_task_path else _claim_task(worker_config)
    except Exception as error:
        return WorkerResult(False, f"worker claim failed: {error}")
    if task is None:
        return WorkerResult(True, "worker queue empty")
    request_id = str(task.get("id") or task.get("analysis_request_id"))
    request_type = str(task.get("request_type") or task.get("type"))
    try:
        if request_type == "stock_analysis":
            symbol = str(task.get("symbol"))
            factor_run_id = str(task.get("factor_run_id") or worker_config.default_factor_run_id or "")
            if not factor_run_id:
                raise ValueError("stock_analysis task requires factor_run_id or worker default_factor_run_id")
            result = analyze_stock(config_path, symbol=symbol, factor_run_id=factor_run_id, trade_date=task.get("report_date") or worker_config.default_trade_date)
            payload = result.artifact
        elif request_type == "report_update":
            factor_run_id = str(task.get("factor_run_id") or worker_config.default_factor_run_id or "")
            if not factor_run_id:
                raise ValueError("report_update task requires factor_run_id or worker default_factor_run_id in first version")
            result = build_report_artifact(config_path, factor_run_id=factor_run_id, trade_date=task.get("report_date") or worker_config.default_trade_date)
            payload = result.artifact
        else:
            raise ValueError(f"unsupported request_type: {request_type}")
        if not result.ok:
            raise ValueError(result.message)
        response = {
            "status": "completed",
            "result_artifact_json": payload,
            "warnings_json": [],
            "source_snapshot_id": (payload or {}).get("publish_metadata", {}).get("source_snapshot_id") or (payload or {}).get("analysis_metadata", {}).get("source_snapshot_id"),
            "source_strategy_version": (payload or {}).get("publish_metadata", {}).get("source_strategy_version") or (payload or {}).get("analysis_metadata", {}).get("source_strategy_version"),
            "generated_at": (payload or {}).get("publish_metadata", {}).get("generated_at") or (payload or {}).get("analysis_metadata", {}).get("generated_at"),
        }
        _submit_result(worker_config, request_id, response, mock_task_path)
        return WorkerResult(True, f"worker completed task: {request_id}")
    except Exception as error:
        _submit_result(worker_config, request_id, {"status": "failed", "error_message": str(error), "warnings_json": []}, mock_task_path)
        return WorkerResult(False, f"worker failed task {request_id}: {error}")


def run_worker_loop(config_path: Path, worker_config_path: Path, max_iterations: int | None = None) -> WorkerResult:
    worker_config = load_worker_config(worker_config_path)
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        result = run_worker_once(config_path, worker_config_path)
        if not result.ok:
            return result
        iterations += 1
        time.sleep(worker_config.poll_interval_seconds)
    return WorkerResult(True, f"worker loop completed iterations: {iterations}")


def _claim_task(config: WorkerConfig) -> dict[str, Any] | None:
    response = _request_json(
        config,
        "/api/worker/analysis-requests/claim",
        {"worker_id": config.worker_id, "capabilities": {"request_types": ["report_update", "stock_analysis"], "artifact_schema_versions": ["stock_app_publish_v001", "stock_app_stock_analysis_v001"]}},
    )
    return response.get("analysis_request")


def _submit_result(config: WorkerConfig, request_id: str, payload: dict[str, Any], mock_task_path: Path | None = None) -> None:
    if mock_task_path:
        result_path = mock_task_path.with_suffix(".result.json")
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return
    _request_json(config, f"/api/worker/analysis-requests/{request_id}/result", payload)


def _request_json(config: WorkerConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get(config.worker_token_env)
    if not token:
        raise ValueError(f"missing required environment variable: {config.worker_token_env}")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        config.app_base_url + path,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"app worker API failed: HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"app worker API network error: {error.reason}") from error


def _load_mock_task(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload.get("analysis_request", payload)


def _simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data
