"""Local app worker client and runner."""

from __future__ import annotations

import json
import hashlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stock_picker.analysis import analyze_stock
from stock_picker.config import load_storage_config
from stock_picker.publish import build_daily_bundle


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
        if request_type != "stock_analysis":
            raise ValueError(f"unsupported request_type: {request_type}")
        symbol = str(task.get("symbol"))
        factor_run_id = str(task.get("factor_run_id") or worker_config.default_factor_run_id or "")
        if not factor_run_id:
            raise ValueError("stock_analysis task requires factor_run_id or worker default_factor_run_id")
        result = analyze_stock(config_path, symbol=symbol, factor_run_id=factor_run_id, trade_date=task.get("report_date") or worker_config.default_trade_date)
        payload = result.artifact
        if not result.ok:
            raise ValueError(result.message)
        response = _stock_analysis_upload_payload(worker_config, request_id, payload)
        _upload_result(worker_config, response, _mock_result_path(mock_task_path))
        return WorkerResult(True, f"worker completed task: {request_id}")
    except Exception as error:
        response = {
            "result_type": "stock_analysis",
            "job_id": request_id,
            "worker_id": worker_config.worker_id,
            "status": "failed",
            "error_message": str(error),
            "warnings_json": [],
        }
        _upload_result(worker_config, response, _mock_result_path(mock_task_path))
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


def run_daily_check(
    config_path: Path,
    worker_config_path: Path,
    factor_run_id: str | None = None,
    trade_date: str | None = None,
    previous_bundle_path: Path | None = None,
    previous_candidate_pool_path: Path | None = None,
    top: int = 10,
    force: bool = False,
    mock_upload: bool = False,
    mock_upload_path: Path | None = None,
) -> WorkerResult:
    config = load_storage_config(config_path)
    worker_config = load_worker_config(worker_config_path)
    selected_factor_run = factor_run_id or worker_config.default_factor_run_id
    if not selected_factor_run:
        return WorkerResult(False, "daily check requires factor_run_id or worker default_factor_run_id")
    result = build_daily_bundle(
        config_path,
        factor_run_id=selected_factor_run,
        trade_date=trade_date or worker_config.default_trade_date,
        previous_bundle_path=previous_bundle_path,
        previous_candidate_pool_path=previous_candidate_pool_path,
        top=top,
    )
    if not result.ok or not result.artifact:
        return WorkerResult(False, result.message)

    payload = _daily_bundle_upload_payload(worker_config, result.artifact)
    selected_date = str(payload.get("trade_date"))
    artifact_hash = str(payload.get("artifact_hash"))
    state_path = _daily_upload_state_path(config)
    state = _load_json(state_path)
    previous = state.get("daily_bundle_uploads", {}).get(selected_date, {})
    if not force and previous.get("artifact_hash") == artifact_hash:
        return WorkerResult(True, f"daily bundle already uploaded: {selected_date} {artifact_hash}")

    upload_path = mock_upload_path or config.reports_root / "app_worker" / f"daily_bundle_upload_{selected_date.replace('-', '')}.json"
    response = _upload_result(worker_config, payload, upload_path if mock_upload else None)
    state.setdefault("daily_bundle_uploads", {})[selected_date] = {
        "artifact_hash": artifact_hash,
        "uploaded_at": _generated_at(result.artifact),
        "remote_result_id": response.get("result_id") or response.get("bundle_id"),
        "mock_upload_path": str(upload_path) if mock_upload else None,
    }
    _write_json(state_path, state)
    return WorkerResult(True, f"daily bundle uploaded: {selected_date} {artifact_hash}")


def _claim_task(config: WorkerConfig) -> dict[str, Any] | None:
    response = _request_json(
        config,
        "/api/worker/analysis-requests/claim",
        {"worker_id": config.worker_id, "capabilities": {"request_types": ["stock_analysis"], "artifact_schema_versions": ["stock_app_stock_analysis_v001"]}},
    )
    return response.get("analysis_request")


def _upload_result(config: WorkerConfig, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        return {"status": "mock_uploaded", "result_id": mock_output_path.stem}
    return _request_json(config, "/api/worker/results/upload", payload)


def _request_json(config: WorkerConfig, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get(config.worker_token_env)
    if not token:
        raise ValueError(f"missing required environment variable: {config.worker_token_env}")
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
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


def _mock_result_path(mock_task_path: Path | None) -> Path | None:
    return mock_task_path.with_suffix(".result.json") if mock_task_path else None


def _stock_analysis_upload_payload(config: WorkerConfig, job_id: str, artifact: dict[str, Any] | None) -> dict[str, Any]:
    artifact = artifact or {}
    return {
        "result_type": "stock_analysis",
        "job_id": job_id,
        "worker_id": config.worker_id,
        "status": "completed",
        "schema_version": artifact.get("analysis_metadata", {}).get("schema_version"),
        "artifact_hash": _artifact_hash(artifact),
        "artifact_json": artifact,
        "warnings_json": [],
        "source_snapshot_id": _source_snapshot_id(artifact),
        "source_strategy_version": _source_strategy_version(artifact),
        "generated_at": _generated_at(artifact),
    }


def _daily_bundle_upload_payload(config: WorkerConfig, artifact: dict[str, Any]) -> dict[str, Any]:
    metadata = artifact.get("bundle_metadata", {})
    return {
        "result_type": "daily_bundle",
        "worker_id": config.worker_id,
        "status": "completed",
        "trade_date": metadata.get("trade_date"),
        "schema_version": metadata.get("schema_version"),
        "artifact_hash": metadata.get("bundle_hash") or _artifact_hash(artifact),
        "artifact_json": artifact,
        "warnings_json": [],
        "source_snapshot_id": _source_snapshot_id(artifact),
        "source_strategy_version": _source_strategy_version(artifact),
        "generated_at": metadata.get("generated_at"),
    }


def _artifact_hash(artifact: dict[str, Any]) -> str:
    encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _daily_upload_state_path(config) -> Path:
    return config.reports_root / "app_worker" / "daily_upload_state.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _source_snapshot_id(payload: dict[str, Any] | None) -> str | None:
    payload = payload or {}
    return (
        payload.get("bundle_metadata", {}).get("source_snapshot_id")
        or payload.get("analysis_metadata", {}).get("source_snapshot_id")
    )


def _source_strategy_version(payload: dict[str, Any] | None) -> str | None:
    payload = payload or {}
    return (
        payload.get("candidate_pool", {}).get("candidate_pool_metadata", {}).get("strategy_version")
        or payload.get("analysis_metadata", {}).get("source_strategy_version")
    )


def _generated_at(payload: dict[str, Any] | None) -> str | None:
    payload = payload or {}
    return payload.get("bundle_metadata", {}).get("generated_at") or payload.get("analysis_metadata", {}).get("generated_at")


def _simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data
