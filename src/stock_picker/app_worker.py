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
from stock_picker.workflow import sync_report_workflow

DEFAULT_NO_PROXY = "127.0.0.1,localhost,101.34.212.101"


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class WorkerConfig:
    app_base_url: str
    worker_id: str
    worker_token_env: str
    tushare_token_env: str
    poll_interval_seconds: float
    holding_price_poll_interval_seconds: float
    daily_bundle_check_interval_seconds: float
    daily_bundle_upload_archive_max: int
    earliest_daily_publish_time: str
    analysis_claim_path: str
    analysis_result_path_template: str
    daily_bundle_publish_path: str
    holding_watchlist_path: str
    holding_prices_path: str
    stock_analysis_enabled: bool = True
    daily_bundle_enabled: bool = True
    holding_price_enabled: bool = True
    local_env_path: Path | None = None
    default_factor_run_id: str | None = None
    default_trade_date: str | None = None


def load_worker_config(path: Path) -> WorkerConfig:
    defaults = _worker_defaults()
    if not path.exists():
        return WorkerConfig(**defaults, local_env_path=path.parent / ".env")
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:
        data = _simple_yaml(path)
    tasks = data.get("tasks", {}) if isinstance(data.get("tasks"), dict) else {}
    api_paths = data.get("api_paths", {}) if isinstance(data.get("api_paths"), dict) else {}
    stock_task = tasks.get("stock_analysis", {}) if isinstance(tasks.get("stock_analysis"), dict) else {}
    daily_task = tasks.get("daily_bundle", {}) if isinstance(tasks.get("daily_bundle"), dict) else {}
    holding_task = tasks.get("holding_prices", {}) if isinstance(tasks.get("holding_prices"), dict) else {}
    return WorkerConfig(
        app_base_url=str(data.get("app_base_url", defaults["app_base_url"])).rstrip("/"),
        worker_id=str(data.get("worker_id", defaults["worker_id"])),
        worker_token_env=str(data.get("worker_token_env", defaults["worker_token_env"])),
        tushare_token_env=str(data.get("tushare_token_env", defaults["tushare_token_env"])),
        poll_interval_seconds=float(stock_task.get("poll_interval_seconds", data.get("poll_interval_seconds", defaults["poll_interval_seconds"]))),
        holding_price_poll_interval_seconds=float(holding_task.get("poll_interval_seconds", data.get("holding_price_poll_interval_seconds", defaults["holding_price_poll_interval_seconds"]))),
        daily_bundle_check_interval_seconds=float(daily_task.get("check_interval_seconds", data.get("daily_bundle_check_interval_seconds", defaults["daily_bundle_check_interval_seconds"]))),
        daily_bundle_upload_archive_max=max(0, int(daily_task.get("upload_archive_max", data.get("daily_bundle_upload_archive_max", defaults["daily_bundle_upload_archive_max"])))),
        earliest_daily_publish_time=str(daily_task.get("earliest_publish_time", data.get("earliest_daily_publish_time", defaults["earliest_daily_publish_time"]))),
        analysis_claim_path=str(api_paths.get("analysis_claim", data.get("analysis_claim_path", defaults["analysis_claim_path"]))),
        analysis_result_path_template=str(api_paths.get("analysis_result", data.get("analysis_result_path_template", defaults["analysis_result_path_template"]))),
        daily_bundle_publish_path=str(api_paths.get("daily_bundle_publish", data.get("daily_bundle_publish_path", defaults["daily_bundle_publish_path"]))),
        holding_watchlist_path=str(api_paths.get("holding_watchlist", data.get("holding_watchlist_path", defaults["holding_watchlist_path"]))),
        holding_prices_path=str(api_paths.get("holding_prices", data.get("holding_prices_path", defaults["holding_prices_path"]))),
        stock_analysis_enabled=_bool(stock_task.get("enabled", data.get("stock_analysis_enabled", True))),
        daily_bundle_enabled=_bool(daily_task.get("enabled", data.get("daily_bundle_enabled", True))),
        holding_price_enabled=_bool(holding_task.get("enabled", data.get("holding_price_enabled", True))),
        local_env_path=path.parent / ".env",
        default_factor_run_id=data.get("default_factor_run_id"),
        default_trade_date=data.get("default_trade_date"),
    )


def run_worker_once(config_path: Path, worker_config_path: Path, mock_task_path: Path | None = None) -> WorkerResult:
    worker_config = load_worker_config(worker_config_path)
    if not worker_config.stock_analysis_enabled:
        return WorkerResult(True, "stock analysis worker disabled")
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
        factor_run_id = _resolve_factor_run_id(config_path, task.get("factor_run_id"), worker_config)
        if not factor_run_id:
            raise ValueError("stock_analysis task requires a local Candidate 002 factor run")
        result = analyze_stock(config_path, symbol=symbol, factor_run_id=factor_run_id, trade_date=task.get("report_date") or task.get("reportDate") or worker_config.default_trade_date)
        payload = result.artifact
        if not result.ok:
            raise ValueError(result.message)
        response = _stock_analysis_upload_payload(payload)
        _upload_stock_analysis_result(worker_config, request_id, response, _mock_result_path(mock_task_path))
        summary = _stock_analysis_summary(payload)
        suffix = f"; {summary}" if summary else ""
        return WorkerResult(True, f"worker completed task: {request_id}{suffix}")
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


def run_manual_stock_analysis(
    config_path: Path,
    worker_config_path: Path,
    symbol: str,
    trade_date: str | None = None,
    output_path: Path | None = None,
    json_events: bool = False,
) -> WorkerResult:
    worker_config = load_worker_config(worker_config_path)
    selected_symbol = symbol.strip().upper()
    _emit_worker_event(json_events, "stock_analysis", "running", "validate_symbol", 1, 4, f"validating symbol: {selected_symbol}")
    if not selected_symbol:
        _emit_worker_event(json_events, "stock_analysis", "failed", "validate_symbol", 1, 4, "symbol is required", error="symbol is required")
        return WorkerResult(False, "stock analysis requires a symbol")
    _emit_worker_event(json_events, "stock_analysis", "running", "resolve_factor_run", 2, 4, "resolving latest Candidate 002 factor run")
    factor_run_id = _resolve_factor_run_id(config_path, None, worker_config, allow_config_default=False)
    if not factor_run_id:
        message = "stock analysis requires a local Candidate 002 factor run"
        _emit_worker_event(json_events, "stock_analysis", "failed", "resolve_factor_run", 2, 4, message, error=message)
        return WorkerResult(False, message)
    _emit_worker_event(json_events, "stock_analysis", "running", "analyze_stock", 3, 4, f"analyzing {selected_symbol} with {factor_run_id}")
    result = analyze_stock(
        config_path,
        symbol=selected_symbol,
        factor_run_id=factor_run_id,
        trade_date=trade_date or worker_config.default_trade_date,
        output_path=output_path,
    )
    if not result.ok:
        _emit_worker_event(json_events, "stock_analysis", "failed", "analyze_stock", 3, 4, result.message, error=result.message)
        return WorkerResult(False, result.message)
    summary = _stock_analysis_summary(result.artifact)
    message = f"{result.message}; {summary}" if summary else result.message
    _emit_worker_event(json_events, "stock_analysis", "completed", "write_summary", 4, 4, message)
    return WorkerResult(True, message)


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
    json_events: bool = False,
    auto_pipeline: bool = False,
) -> WorkerResult:
    return DailyBundleJob(
        config_path=config_path,
        worker_config_path=worker_config_path,
        factor_run_id=factor_run_id,
        trade_date=trade_date,
        previous_bundle_path=previous_bundle_path,
        previous_candidate_pool_path=previous_candidate_pool_path,
        top=top,
        force=force,
        mock_upload=mock_upload,
        mock_upload_path=mock_upload_path,
        json_events=json_events,
        auto_pipeline=auto_pipeline,
    ).run()


class DailyBundleJob:
    def __init__(
        self,
        *,
        config_path: Path,
        worker_config_path: Path,
        factor_run_id: str | None,
        trade_date: str | None,
        previous_bundle_path: Path | None,
        previous_candidate_pool_path: Path | None,
        top: int,
        force: bool,
        mock_upload: bool,
        mock_upload_path: Path | None,
        json_events: bool,
        auto_pipeline: bool,
    ) -> None:
        self.config_path = config_path
        self.worker_config_path = worker_config_path
        self.factor_run_id = factor_run_id
        self.trade_date = trade_date
        self.previous_bundle_path = previous_bundle_path
        self.previous_candidate_pool_path = previous_candidate_pool_path
        self.top = top
        self.force = force
        self.mock_upload = mock_upload
        self.mock_upload_path = mock_upload_path
        self.json_events = json_events
        self.auto_pipeline = auto_pipeline
        self.config = load_storage_config(config_path)
        self.worker_config = load_worker_config(worker_config_path)
        self.state_path = _daily_upload_state_path(self.config)
        self.state = _load_json(self.state_path)

    def run(self) -> WorkerResult:
        if not self.worker_config.daily_bundle_enabled:
            return WorkerResult(True, "daily bundle worker disabled")
        self._event("started", "resolve_factor_run", 1, 4, "daily bundle check started")
        pipeline_result = self._ensure_auto_pipeline()
        if pipeline_result is not None:
            return pipeline_result
        selected_factor_run = self._resolve_factor_run()
        if not selected_factor_run:
            self._event("failed", "resolve_factor_run", 1, 4, "missing local Candidate 002 factor run")
            return WorkerResult(False, "daily check requires a local Candidate 002 factor run")
        bundle_result = self._build_bundle(selected_factor_run)
        if not bundle_result.ok or not bundle_result.artifact:
            self._event("failed", "build_bundle", 2, 4, bundle_result.message)
            return WorkerResult(False, bundle_result.message)
        if len(bundle_result.artifact.get("candidate_pool", {}).get("top_stocks", [])) != 10:
            self._event("failed", "validate_bundle", 3, 4, "candidate_pool top_stocks is not 10")
            return WorkerResult(False, "daily bundle upload requires exactly 10 candidate_pool top_stocks")
        built = self._persist_bundle_built(bundle_result)
        skip_result = self._dedupe_upload_decision(built)
        if skip_result is not None:
            return skip_result
        return self._upload_and_persist(bundle_result, built)

    def _ensure_auto_pipeline(self) -> WorkerResult | None:
        if not self.auto_pipeline or self.factor_run_id:
            return None
        workflow_id = _active_daily_workflow_id(self.state)
        if workflow_id:
            self._event("running", "sync_latest", 1, 4, f"resuming daily workflow: {workflow_id}")
        else:
            workflow_id = f"daily_bundle_auto_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            _set_active_daily_workflow(self.state, workflow_id, "running")
            self._save_state()
        self._event("running", "sync_latest", 1, 4, "syncing latest data and computing Candidate 002")
        try:
            pipeline = sync_report_workflow(
                config_path=self.config_path,
                workflow_id=workflow_id,
                dry_run=False,
                confirmed=True,
                start_date=None,
                end_date=self.trade_date or self.worker_config.default_trade_date,
                top=10,
                emit=lambda event: _print_json_event(event) if self.json_events else None,
            )
        except Exception as error:
            self.state = _load_json(self.state_path)
            _set_active_daily_workflow(self.state, workflow_id, "failed", str(error))
            self._save_state()
            return WorkerResult(False, str(error))
        if not pipeline.ok:
            self.state = _load_json(self.state_path)
            _set_active_daily_workflow(self.state, workflow_id, "failed", pipeline.message)
            self._save_state()
            return WorkerResult(False, pipeline.message)
        self.state = _load_json(self.state_path)
        _set_active_daily_workflow(self.state, workflow_id, "pipeline_completed")
        self._save_state()
        return None

    def _resolve_factor_run(self) -> str | None:
        return _resolve_factor_run_id(self.config_path, self.factor_run_id, self.worker_config, allow_config_default=not self.auto_pipeline)

    def _build_bundle(self, selected_factor_run: str):
        self._event("running", "build_bundle", 2, 4, f"building bundle from {selected_factor_run}")
        return build_daily_bundle(
            self.config_path,
            factor_run_id=selected_factor_run,
            trade_date=self.trade_date or self.worker_config.default_trade_date,
            previous_bundle_path=self.previous_bundle_path,
            previous_candidate_pool_path=self.previous_candidate_pool_path,
            top=self.top,
        )

    def _persist_bundle_built(self, result) -> dict[str, Any]:
        payload = result.artifact
        metadata = payload.get("bundle_metadata", {})
        selected_date = str(metadata.get("trade_date"))
        artifact_hash = str(metadata.get("bundle_hash") or _artifact_hash(payload))
        bundle_path = str(result.artifact_path) if result.artifact_path else None
        active_workflow_id = _active_daily_workflow_id(self.state)
        _set_latest_daily_bundle_state(
            self.state,
            status="bundle_built",
            trade_date=selected_date,
            artifact_hash=artifact_hash,
            bundle_path=bundle_path,
            workflow_id=active_workflow_id,
        )
        self._save_state()
        return {
            "payload": payload,
            "selected_date": selected_date,
            "artifact_hash": artifact_hash,
            "bundle_path": bundle_path,
            "active_workflow_id": active_workflow_id,
        }

    def _dedupe_upload_decision(self, built: dict[str, Any]) -> WorkerResult | None:
        previous = self.state.get("daily_bundle_uploads", {}).get(built["selected_date"], {})
        if self.force or previous.get("artifact_hash") != built["artifact_hash"]:
            return None
        _set_latest_daily_bundle_state(
            self.state,
            status="skipped_already_uploaded",
            trade_date=built["selected_date"],
            artifact_hash=built["artifact_hash"],
            bundle_path=built["bundle_path"],
            workflow_id=built["active_workflow_id"],
            archive_path=previous.get("archive_path"),
            remote_result_id=previous.get("remote_result_id"),
            upload_response=previous.get("upload_response") if isinstance(previous.get("upload_response"), dict) else None,
        )
        _clear_active_daily_workflow(self.state)
        self._save_state()
        self._event("skipped", "upload_bundle", 4, 4, f"already uploaded: {built['selected_date']}")
        return WorkerResult(True, f"daily bundle already uploaded: {built['selected_date']} {built['artifact_hash']}")

    def _upload_and_persist(self, result, built: dict[str, Any]) -> WorkerResult:
        upload_path = self.mock_upload_path or self.config.reports_root / "app_worker" / f"daily_bundle_upload_{built['selected_date'].replace('-', '')}.json"
        self._event("running", "upload_bundle", 4, 4, "uploading daily bundle")
        self._event("running", "upload_bundle", 4, 4, f"upload target: {self.worker_config.app_base_url}{self.worker_config.daily_bundle_publish_path}; no_proxy={_expected_no_proxy()}; proxy=disabled")
        if built["active_workflow_id"]:
            _set_active_daily_workflow(self.state, built["active_workflow_id"], "uploading")
        _set_latest_daily_bundle_state(
            self.state,
            status="uploading",
            trade_date=built["selected_date"],
            artifact_hash=built["artifact_hash"],
            bundle_path=built["bundle_path"],
            workflow_id=built["active_workflow_id"],
        )
        self._save_state()
        try:
            response = _upload_daily_bundle_with_retry(
                self.worker_config,
                built["payload"],
                upload_path if self.mock_upload else None,
                json_events=self.json_events,
            )
        except Exception as error:
            self._persist_upload_failure(built, error)
            return WorkerResult(False, f"daily bundle upload failed: {error}")
        return self._persist_upload_success(result, built, response, upload_path)

    def _persist_upload_failure(self, built: dict[str, Any], error: Exception) -> None:
        self.state = _load_json(self.state_path)
        if built["active_workflow_id"]:
            _set_active_daily_workflow(self.state, built["active_workflow_id"], "upload_failed", str(error))
        _set_latest_daily_bundle_state(
            self.state,
            status="upload_failed",
            trade_date=built["selected_date"],
            artifact_hash=built["artifact_hash"],
            bundle_path=built["bundle_path"],
            workflow_id=built["active_workflow_id"],
            error=str(error),
        )
        self._save_state()
        self._event("failed", "upload_bundle", 4, 4, str(error), error=str(error))

    def _persist_upload_success(self, result, built: dict[str, Any], response: dict[str, Any], upload_path: Path) -> WorkerResult:
        self.state = _load_json(self.state_path)
        archived_path = _archive_uploaded_daily_bundle(self.config, built["payload"], built["artifact_hash"], self.worker_config.daily_bundle_upload_archive_max)
        self.state.setdefault("daily_bundle_uploads", {})[built["selected_date"]] = {
            "artifact_hash": built["artifact_hash"],
            "uploaded_at": _generated_at(result.artifact),
            "remote_result_id": response.get("daily_bundle_id") or response.get("bundle_id"),
            "mock_upload_path": str(upload_path) if self.mock_upload else None,
            "archive_path": str(archived_path) if archived_path else None,
            "upload_response": _upload_response_summary(response),
        }
        _set_latest_daily_bundle_state(
            self.state,
            status="uploaded",
            trade_date=built["selected_date"],
            artifact_hash=built["artifact_hash"],
            bundle_path=built["bundle_path"],
            workflow_id=built["active_workflow_id"],
            archive_path=str(archived_path) if archived_path else None,
            remote_result_id=response.get("daily_bundle_id") or response.get("bundle_id"),
            upload_response=_upload_response_summary(response),
        )
        _clear_active_daily_workflow(self.state)
        self._save_state()
        response_text = json.dumps(_upload_response_summary(response), ensure_ascii=False, sort_keys=True)
        self._event("completed", "upload_bundle", 4, 4, f"daily bundle uploaded: {built['selected_date']}; upload_response={response_text}")
        return WorkerResult(True, f"daily bundle uploaded: {built['selected_date']} {built['artifact_hash']}; upload_response={response_text}")

    def _save_state(self) -> None:
        _write_json(self.state_path, self.state)

    def _event(self, status: str, step: str, step_index: int, step_total: int, message: str, *, error: str | None = None) -> None:
        _emit_worker_event(self.json_events, "daily_bundle", status, step, step_index, step_total, message, error=error)


def run_holding_price_refresh(
    worker_config_path: Path,
    trade_date: str | None = None,
    token_env: str | None = None,
    mock_watchlist_path: Path | None = None,
    mock_upload_path: Path | None = None,
    limit: int | None = None,
    json_events: bool = False,
) -> WorkerResult:
    _ensure_no_proxy()
    worker_config = load_worker_config(worker_config_path)
    if not worker_config.holding_price_enabled:
        return WorkerResult(True, "holding price worker disabled")
    selected_token_env = token_env or worker_config.tushare_token_env
    try:
        _emit_worker_event(json_events, "holding_analysis", "running", "load_watchlist", 1, 3, "loading holding watchlist")
        watchlist = _load_holding_watchlist(worker_config, mock_watchlist_path)
        symbols = list(watchlist.get("symbols") or [])
        if limit is not None:
            symbols = symbols[:limit]
        if not symbols:
            _emit_worker_event(json_events, "holding_analysis", "completed", "load_watchlist", 1, 3, "holding price watchlist empty")
            return WorkerResult(True, "holding price watchlist empty")
        token = _env_value(selected_token_env, worker_config.local_env_path)
        if not token:
            message = f"missing required environment variable: {selected_token_env}"
            _emit_worker_event(json_events, "holding_analysis", "failed", "load_watchlist", 1, 3, message, error=message)
            return WorkerResult(False, message)
        _emit_worker_event(json_events, "holding_analysis", "running", "refresh_quotes", 2, 3, f"refreshing prices for {len(symbols)} holdings")
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
            message = "holding price refresh found no quote rows"
            _emit_worker_event(json_events, "holding_analysis", "failed", "refresh_quotes", 2, 3, message, error=message)
            return WorkerResult(False, message)
        _emit_worker_event(json_events, "holding_analysis", "running", "upload_prices", 3, 3, f"uploading {len(prices)} holding prices")
        response = _upload_holding_prices(worker_config, {"prices": prices}, mock_upload_path)
        suffix = f", skipped={len(skipped)}" if skipped else ""
        summary = _holding_prices_summary(prices)
        message = f"holding prices uploaded: {len(prices)}{suffix}; status={response.get('status', 'ok')}; prices={summary}"
        _emit_worker_event(json_events, "holding_analysis", "completed", "upload_prices", 3, 3, message)
        return WorkerResult(True, message)
    except Exception as error:
        _emit_worker_event(json_events, "holding_analysis", "failed", "upload_prices", 3, 3, str(error), error=str(error))
        return WorkerResult(False, f"holding price refresh failed: {error}")


def run_holding_price_loop(
    worker_config_path: Path,
    trade_date: str | None = None,
    token_env: str | None = None,
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
            json_events=False,
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
        config.analysis_claim_path,
        {"worker_id": config.worker_id, "capabilities": {"request_types": ["stock_analysis"], "artifact_schema_versions": ["stock_app_stock_analysis_v001"]}},
        token_kind="worker",
    )
    return response.get("analysis_request")


def _upload_stock_analysis_result(config: WorkerConfig, request_id: str, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        return {"status": "mock_uploaded", "analysis_request_id": request_id}
    return _post_json(config, config.analysis_result_path_template.replace("{requestId}", request_id), payload, token_kind="worker")


def _upload_daily_bundle(config: WorkerConfig, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        metadata = payload.get("bundle_metadata", {})
        return {"status": "mock_uploaded", "bundle_id": metadata.get("bundle_id")}
    return _post_json(config, config.daily_bundle_publish_path, payload, token_kind="worker")


def _upload_daily_bundle_with_retry(
    config: WorkerConfig,
    payload: dict[str, Any],
    mock_output_path: Path | None = None,
    *,
    json_events: bool = False,
    max_retries: int = 3,
    retry_interval_seconds: float = 5,
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            _emit_worker_event(
                json_events,
                "daily_bundle",
                "running",
                "upload_bundle",
                4,
                4,
                f"upload attempt {attempt + 1}/{max_retries + 1}",
            )
            return _upload_daily_bundle(config, payload, mock_output_path)
        except Exception as error:
            if attempt >= max_retries or not _is_retryable_upload_error(error):
                raise
            attempt += 1
            _emit_worker_event(
                json_events,
                "daily_bundle",
                "retrying",
                "upload_bundle",
                4,
                4,
                f"upload failed, retrying {attempt}/{max_retries} in {retry_interval_seconds:g}s: {error}",
                error=str(error),
            )
            time.sleep(retry_interval_seconds)


def _is_retryable_upload_error(error: Exception) -> bool:
    if isinstance(error, (urllib.error.URLError, ConnectionError, TimeoutError, OSError)):
        return True
    message = str(error)
    return (
        "HTTP 5" in message
        or "WinError 10054" in message
        or "network error" in message.lower()
        or "timed out" in message.lower()
        or "temporarily unavailable" in message.lower()
    )


def _upload_holding_prices(config: WorkerConfig, payload: dict[str, Any], mock_output_path: Path | None = None) -> dict[str, Any]:
    if mock_output_path:
        _write_json(mock_output_path, payload)
        return {"status": "mock_uploaded"}
    return _post_json(config, config.holding_prices_path, payload, token_kind="worker")


def _post_json(config: WorkerConfig, path: str, payload: dict[str, Any], token_kind: str) -> dict[str, Any]:
    _ensure_no_proxy()
    token = _token(config, token_kind)
    if not token:
        raise ValueError(f"missing required environment variable: {_token_env_name(config, token_kind)}")
    url = config.app_base_url + path
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    return _open_json(request, url)


def _get_json(config: WorkerConfig, path: str, token_kind: str) -> dict[str, Any]:
    _ensure_no_proxy()
    token = _token(config, token_kind)
    if not token:
        raise ValueError(f"missing required environment variable: {_token_env_name(config, token_kind)}")
    url = config.app_base_url + path
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )
    return _open_json(request, url)


def _open_json(request: urllib.request.Request, url: str) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        message = f"app worker API failed: HTTP {error.code}; url={url}"
        if body:
            message = f"{message}: {body}"
        raise RuntimeError(message) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"app worker API network error: {error.reason}; url={url}") from error


def _ensure_no_proxy() -> None:
    merged = _expected_no_proxy()
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def _expected_no_proxy() -> str:
    return _merge_no_proxy(os.environ.get("NO_PROXY"), os.environ.get("no_proxy"), DEFAULT_NO_PROXY)


def _merge_no_proxy(*values: str | None) -> str:
    entries: list[str] = []
    for value in values:
        if not value:
            continue
        for item in value.split(","):
            trimmed = item.strip()
            if trimmed and trimmed not in entries:
                entries.append(trimmed)
    return ",".join(entries)


def _token(config: WorkerConfig, token_kind: str) -> str | None:
    primary = _token_env_name(config, token_kind)
    return _env_value(primary, config.local_env_path)


def _token_env_name(config: WorkerConfig, token_kind: str) -> str:
    return config.worker_token_env


def _load_holding_watchlist(config: WorkerConfig, mock_watchlist_path: Path | None) -> dict[str, Any]:
    if mock_watchlist_path:
        return json.loads(mock_watchlist_path.read_text(encoding="utf-8-sig"))
    return _get_json(config, config.holding_watchlist_path, token_kind="worker")


def _resolve_factor_run_id(config_path: Path, explicit_factor_run_id: Any, worker_config: WorkerConfig, allow_config_default: bool = True) -> str | None:
    if explicit_factor_run_id:
        return str(explicit_factor_run_id)
    if allow_config_default and worker_config.default_factor_run_id:
        return str(worker_config.default_factor_run_id)
    config = load_storage_config(config_path)
    run_root = config.reports_root / "factor_exploration"
    if not run_root.exists():
        return None
    candidates: list[tuple[str, str]] = []
    for metadata_path in run_root.glob("*/factor_run_metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not (metadata_path.parent / "stock_factors_daily.csv").exists():
            continue
        factor_version = str(metadata.get("factor_version") or "")
        if factor_version and "flow_momentum_quality" not in factor_version and "candidate_002" not in factor_version:
            continue
        run_id = str(metadata.get("factor_run_id") or metadata_path.parent.name)
        sort_key = str(metadata.get("end_date") or metadata.get("created_at") or metadata_path.parent.stat().st_mtime)
        candidates.append((sort_key, run_id))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


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


def _archive_uploaded_daily_bundle(config, payload: dict[str, Any], artifact_hash: str, max_count: int) -> Path | None:
    if max_count <= 0:
        return None
    metadata = payload.get("bundle_metadata", {})
    trade_date = str(metadata.get("trade_date") or "unknown").replace("-", "")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = config.reports_root / "app_worker" / "uploaded_bundles"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"daily_bundle_{trade_date}_{timestamp}_{artifact_hash[:12]}.json"
    counter = 2
    while path.exists():
        path = archive_dir / f"daily_bundle_{trade_date}_{timestamp}_{artifact_hash[:12]}_{counter}.json"
        counter += 1
    _write_json(path, payload)
    _prune_daily_bundle_archives(archive_dir, max_count)
    return path


def _prune_daily_bundle_archives(archive_dir: Path, max_count: int) -> None:
    archives = sorted(archive_dir.glob("daily_bundle_*.json"), key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    for path in archives[max_count:]:
        path.unlink()


def _active_daily_workflow_id(state: dict[str, Any]) -> str | None:
    active = state.get("active_daily_workflow")
    if not isinstance(active, dict):
        return None
    workflow_id = str(active.get("workflow_id") or "")
    return workflow_id or None


def _set_active_daily_workflow(state: dict[str, Any], workflow_id: str, status: str, error: str | None = None) -> None:
    active = state.get("active_daily_workflow") if isinstance(state.get("active_daily_workflow"), dict) else {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    state["active_daily_workflow"] = {
        "workflow_id": workflow_id,
        "status": status,
        "created_at": active.get("created_at") or now,
        "updated_at": now,
    }
    if error:
        state["active_daily_workflow"]["error"] = error


def _clear_active_daily_workflow(state: dict[str, Any]) -> None:
    state.pop("active_daily_workflow", None)


def _set_latest_daily_bundle_state(
    state: dict[str, Any],
    *,
    status: str,
    trade_date: str,
    artifact_hash: str,
    bundle_path: str | None,
    workflow_id: str | None,
    error: str | None = None,
    archive_path: str | None = None,
    remote_result_id: str | None = None,
    upload_response: dict[str, Any] | None = None,
) -> None:
    latest = {
        "status": status,
        "trade_date": trade_date,
        "artifact_hash": artifact_hash,
        "bundle_path": bundle_path,
        "workflow_id": workflow_id,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if error:
        latest["error"] = error
    if archive_path:
        latest["archive_path"] = archive_path
    if remote_result_id:
        latest["remote_result_id"] = remote_result_id
    if upload_response:
        latest["upload_response"] = upload_response
    state["latest_daily_bundle"] = latest


def _upload_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "bundle_id",
        "daily_bundle_id",
        "status",
        "validation_errors",
        "message",
        "error",
    ]
    return {key: response.get(key) for key in keys if key in response}


def _stock_analysis_summary(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    stock = payload.get("stock", {}) if isinstance(payload.get("stock"), dict) else {}
    candidate = payload.get("candidate_status", {}) if isinstance(payload.get("candidate_status"), dict) else {}
    notes = payload.get("risk_notes") if isinstance(payload.get("risk_notes"), list) else []
    parts = [
        f"symbol={stock.get('symbol') or payload.get('analysis_metadata', {}).get('symbol')}",
        f"name={stock.get('stock_name') or '-'}",
        f"in_candidate_pool={candidate.get('in_latest_candidate_pool')}",
    ]
    if candidate.get("total_score") is not None:
        parts.append(f"score={_float(candidate.get('total_score')):.3f}")
    if notes:
        parts.append(f"risk_notes={'; '.join(str(item) for item in notes[:3])}")
    return ", ".join(part for part in parts if part and not part.endswith("=None"))


def _holding_prices_summary(prices: list[dict[str, Any]], limit: int = 8) -> str:
    entries = []
    for item in prices[:limit]:
        entries.append(
            f"{item.get('symbol')} {item.get('name') or '-'} "
            f"last={item.get('last_price')} change={item.get('change_percent')}% quote_time={item.get('quote_time')}"
        )
    if len(prices) > limit:
        entries.append(f"... +{len(prices) - limit} more")
    return " | ".join(entries)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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


def _worker_defaults() -> dict[str, Any]:
    return {
        "app_base_url": "http://127.0.0.1:3000",
        "worker_id": "local-worker",
        "worker_token_env": "APP_API_TOKEN",
        "tushare_token_env": "TUSHARE_TOKEN",
        "poll_interval_seconds": 15.0,
        "holding_price_poll_interval_seconds": 300.0,
        "daily_bundle_check_interval_seconds": 900.0,
        "daily_bundle_upload_archive_max": 20,
        "earliest_daily_publish_time": "16:30",
        "analysis_claim_path": "/api/worker/analysis-requests/claim",
        "analysis_result_path_template": "/api/worker/analysis-requests/{requestId}/result",
        "daily_bundle_publish_path": "/api/publish/daily-bundles",
        "holding_watchlist_path": "/api/worker/holding-prices/watchlist",
        "holding_prices_path": "/api/worker/holding-prices",
    }


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _env_value(name: str, env_path: Path | None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    for path in [env_path, Path(".env")]:
        if not path or not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("\"'")
    return None


def _emit_worker_event(json_events: bool, task_type: str, status: str, step: str, step_index: int, step_total: int, message: str, error: str | None = None) -> None:
    if not json_events:
        return
    _print_json_event(
        {
            "event": "worker_task_event",
            "task_type": task_type,
            "status": status,
            "step": step,
            "step_index": step_index,
            "step_total": step_total,
            "message": message,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )


def _print_json_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
