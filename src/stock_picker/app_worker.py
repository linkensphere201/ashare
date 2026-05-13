"""Local app worker client and runner."""

from __future__ import annotations

import json
import hashlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
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
    publisher_token_env: str
    poll_interval_seconds: float
    holding_price_poll_interval_seconds: float
    default_factor_run_id: str | None = None
    default_trade_date: str | None = None


def load_worker_config(path: Path) -> WorkerConfig:
    if not path.exists():
        return WorkerConfig("http://127.0.0.1:3000", "local-worker", "STOCK_APP_WORKER_TOKEN", "STOCK_APP_PUBLISHER_TOKEN", 15.0, 300.0)
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        data = _simple_yaml(path)
    return WorkerConfig(
        app_base_url=str(data.get("app_base_url", "http://127.0.0.1:3000")).rstrip("/"),
        worker_id=str(data.get("worker_id", "local-worker")),
        worker_token_env=str(data.get("worker_token_env", "STOCK_APP_WORKER_TOKEN")),
        publisher_token_env=str(data.get("publisher_token_env", "STOCK_APP_PUBLISHER_TOKEN")),
        poll_interval_seconds=float(data.get("poll_interval_seconds", 15)),
        holding_price_poll_interval_seconds=float(data.get("holding_price_poll_interval_seconds", 300)),
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
    request_type = str(task.get("request_type") or task.get("requestType") or task.get("type"))
    try:
        if request_type != "stock_analysis":
            raise ValueError(f"unsupported request_type: {request_type}")
        symbol = str(task.get("symbol"))
        factor_run_id = str(task.get("factor_run_id") or worker_config.default_factor_run_id or "")
        if not factor_run_id:
            raise ValueError("stock_analysis task requires factor_run_id or worker default_factor_run_id")
        result = analyze_stock(config_path, symbol=symbol, factor_run_id=factor_run_id, trade_date=task.get("report_date") or task.get("reportDate") or worker_config.default_trade_date)
        payload = result.artifact
        if not result.ok:
            raise ValueError(result.message)
        response = _stock_analysis_upload_payload(payload)
        _upload_stock_analysis_result(worker_config, request_id, response, _mock_result_path(mock_task_path))
        return WorkerResult(True, f"worker completed task: {request_id}")
    except Exception as error:
        response = {
            "status": "failed",
            "error_message": str(error),
            "warnings_json": [],
        }
        _upload_stock_analysis_result(worker_config, request_id, response, _mock_result_path(mock_task_path))
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
    if len(result.artifact.get("candidate_pool", {}).get("top_stocks", [])) != 10:
        return WorkerResult(False, "daily bundle upload requires exactly 10 candidate_pool top_stocks")

    payload = result.artifact
    metadata = payload.get("bundle_metadata", {})
    selected_date = str(metadata.get("trade_date"))
    artifact_hash = str(metadata.get("bundle_hash") or _artifact_hash(payload))
    state_path = _daily_upload_state_path(config)
    state = _load_json(state_path)
    previous = state.get("daily_bundle_uploads", {}).get(selected_date, {})
    if not force and previous.get("artifact_hash") == artifact_hash:
        return WorkerResult(True, f"daily bundle already uploaded: {selected_date} {artifact_hash}")

    upload_path = mock_upload_path or config.reports_root / "app_worker" / f"daily_bundle_upload_{selected_date.replace('-', '')}.json"
    try:
        response = _upload_daily_bundle(worker_config, payload, upload_path if mock_upload else None)
    except Exception as error:
        return WorkerResult(False, f"daily bundle upload failed: {error}")
    state.setdefault("daily_bundle_uploads", {})[selected_date] = {
        "artifact_hash": artifact_hash,
        "uploaded_at": _generated_at(result.artifact),
        "remote_result_id": response.get("daily_bundle_id") or response.get("bundle_id"),
        "mock_upload_path": str(upload_path) if mock_upload else None,
    }
    _write_json(state_path, state)
    return WorkerResult(True, f"daily bundle uploaded: {selected_date} {artifact_hash}")


def run_holding_price_refresh(
    worker_config_path: Path,
    trade_date: str | None = None,
    token_env: str = "TUSHARE_TOKEN",
    mock_watchlist_path: Path | None = None,
    mock_upload_path: Path | None = None,
    limit: int | None = None,
) -> WorkerResult:
    worker_config = load_worker_config(worker_config_path)
    try:
        watchlist = _load_holding_watchlist(worker_config, mock_watchlist_path)
        symbols = list(watchlist.get("symbols") or [])
        if limit is not None:
            symbols = symbols[:limit]
        if not symbols:
            return WorkerResult(True, "holding price watchlist empty")
        token = os.environ.get(token_env)
        if not token:
            return WorkerResult(False, f"missing required environment variable: {token_env}")
        prices: list[dict[str, Any]] = []
        skipped: list[str] = []
        for item in symbols:
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            quote = _fetch_latest_tushare_quote(token, symbol, trade_date)
            if quote is None:
                skipped.append(symbol)
                continue
            prices.append({"symbol": symbol, "name": item.get("name"), **quote})
        if not prices:
            return WorkerResult(False, "holding price refresh found no quote rows")
        response = _upload_holding_prices(worker_config, {"prices": prices}, mock_upload_path)
        suffix = f", skipped={len(skipped)}" if skipped else ""
        return WorkerResult(True, f"holding prices uploaded: {len(prices)}{suffix}; status={response.get('status', 'ok')}")
    except Exception as error:
        return WorkerResult(False, f"holding price refresh failed: {error}")


def run_holding_price_loop(
    worker_config_path: Path,
    trade_date: str | None = None,
    token_env: str = "TUSHARE_TOKEN",
    mock_watchlist_path: Path | None = None,
    mock_upload_path: Path | None = None,
    limit: int | None = None,
    max_iterations: int | None = None,
) -> WorkerResult:
    worker_config = load_worker_config(worker_config_path)
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        result = run_holding_price_refresh(
            worker_config_path=worker_config_path,
            trade_date=trade_date,
            token_env=token_env,
            mock_watchlist_path=mock_watchlist_path,
            mock_upload_path=mock_upload_path,
            limit=limit,
        )
        if not result.ok:
            return result
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(worker_config.holding_price_poll_interval_seconds)
    return WorkerResult(True, f"holding price loop completed iterations: {iterations}")


def _claim_task(config: WorkerConfig) -> dict[str, Any] | None:
    response = _post_json(
        config,
        "/api/worker/analysis-requests/claim",
        {"worker_id": config.worker_id, "capabilities": {"request_types": ["stock_analysis"], "artifact_schema_versions": ["stock_app_stock_analysis_v001"]}},
        token_kind="worker",
    )
    return response.get("analysis_request")


def _upload_stock_analysis_result(config: WorkerConfig, request_id: str, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        return {"status": "mock_uploaded", "analysis_request_id": request_id}
    return _post_json(config, f"/api/worker/analysis-requests/{request_id}/result", payload, token_kind="worker")


def _upload_daily_bundle(config: WorkerConfig, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        metadata = payload.get("bundle_metadata", {})
        return {"status": "mock_uploaded", "bundle_id": metadata.get("bundle_id")}
    return _post_json(config, "/api/publish/daily-bundles", payload, token_kind="publisher")


def _upload_holding_prices(config: WorkerConfig, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        return {"status": "mock_uploaded"}
    return _post_json(config, "/api/worker/holding-prices", payload, token_kind="worker")


def _post_json(config: WorkerConfig, path: str, payload: dict[str, Any], token_kind: str) -> dict[str, Any]:
    token = _token(config, token_kind)
    if not token:
        raise ValueError(f"missing required environment variable: {_token_env_name(config, token_kind)}")
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    request = urllib.request.Request(
        config.app_base_url + path,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    return _open_json(request)


def _get_json(config: WorkerConfig, path: str, token_kind: str) -> dict[str, Any]:
    token = _token(config, token_kind)
    if not token:
        raise ValueError(f"missing required environment variable: {_token_env_name(config, token_kind)}")
    request = urllib.request.Request(
        config.app_base_url + path,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    return _open_json(request)


def _open_json(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"app worker API failed: HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"app worker API network error: {error.reason}") from error


def _token(config: WorkerConfig, token_kind: str) -> str | None:
    primary = _token_env_name(config, token_kind)
    fallback = "PUBLISHER_TOKEN" if token_kind == "publisher" else "WORKER_TOKEN"
    return os.environ.get(primary) or os.environ.get(fallback)


def _token_env_name(config: WorkerConfig, token_kind: str) -> str:
    return config.publisher_token_env if token_kind == "publisher" else config.worker_token_env


def _load_holding_watchlist(config: WorkerConfig, mock_watchlist_path: Path | None) -> dict[str, Any]:
    if mock_watchlist_path:
        return json.loads(mock_watchlist_path.read_text(encoding="utf-8-sig"))
    return _get_json(config, "/api/worker/holding-prices/watchlist", token_kind="worker")


def _load_mock_task(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload.get("analysis_request", payload)


def _mock_result_path(mock_task_path: Path | None) -> Path | None:
    return mock_task_path.with_suffix(".result.json") if mock_task_path else None


def _stock_analysis_upload_payload(artifact: dict[str, Any] | None) -> dict[str, Any]:
    artifact = artifact or {}
    return {
        "status": "completed",
        "result_artifact_json": artifact,
        "warnings_json": [],
        "source_snapshot_id": _source_snapshot_id(artifact),
        "source_strategy_version": _source_strategy_version(artifact),
        "generated_at": _generated_at(artifact),
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


def _fetch_latest_tushare_quote(token: str, symbol: str, trade_date: str | None) -> dict[str, Any] | None:
    try:
        import tushare as ts  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("tushare package is required for holding price refresh") from error
    pro = ts.pro_api(token)
    params = {"ts_code": symbol}
    if trade_date:
        params["trade_date"] = _compact_date(trade_date)
    pandas_frame = pro.daily(**params)
    rows = pandas_frame.to_dict("records") if hasattr(pandas_frame, "to_dict") else list(pandas_frame)
    if not rows:
        return None
    row = sorted(rows, key=lambda item: str(item.get("trade_date") or ""))[-1]
    quote_date = _display_date(str(row.get("trade_date") or trade_date or ""))
    return {
        "last_price": _float(row.get("close")),
        "change_percent": _float(row.get("pct_chg", row.get("pct_change"))),
        "quote_time": f"{quote_date}T15:00:00+08:00" if quote_date else datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "tushare.daily",
    }


def _compact_date(value: str | None) -> str | None:
    return value.replace("-", "") if value else None


def _display_date(value: str) -> str:
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data
