"""Shared provider run task engine."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol


class ProviderErrorReason(str, Enum):
    RATE_LIMIT = "rate_limit"
    PERMISSION_DENIED = "permission_denied"
    AUTH_MISSING = "auth_missing"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    EMPTY_RESULT = "empty_result"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_TASK = "invalid_task"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderTaskError:
    reason: ProviderErrorReason
    message: str
    provider_message: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class TaskExecutionResult:
    ok: bool
    raw_batch_id: str | None = None
    raw_path: Path | None = None
    row_count: int = 0
    error: ProviderTaskError | None = None


@dataclass(frozen=True)
class ProviderRunSpec:
    run_id: str
    source: str
    run_type: str
    start_date: str | None
    end_date: str | None
    as_of_date: str
    max_tasks: int
    requests_per_minute: float
    retry: int
    retry_wait_seconds: float
    backoff_multiplier: float
    progress_every_tasks: int = 0


@dataclass(frozen=True)
class ProviderTaskSpec:
    task_id: str
    run_id: str
    source: str
    dataset_name: str
    task_type: str
    trade_date: str | None = None
    symbol_start_offset: int | None = None
    symbol_end_offset: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class ProviderRunResult:
    ok: bool
    message: str
    status: str
    completed_this_invocation: int
    row_count: int
    last_raw_batch_id: str | None = None
    error: ProviderTaskError | None = None


class ProviderTaskAdapter(Protocol):
    def plan_tasks(self, run_spec: ProviderRunSpec) -> list[ProviderTaskSpec]:
        ...

    def execute_task(self, task: dict[str, object]) -> TaskExecutionResult:
        ...


def execute_provider_run(
    sqlite_path: Path,
    run_spec: ProviderRunSpec,
    adapter: ProviderTaskAdapter,
    progress_callback: Callable[[str], None] | None = None,
) -> ProviderRunResult:
    if run_spec.max_tasks <= 0:
        return ProviderRunResult(False, "provider run requires a positive max_tasks", "failed", 0, 0)
    if run_spec.requests_per_minute < 0:
        return ProviderRunResult(False, "provider run requires a non-negative requests_per_minute", "failed", 0, 0)
    if run_spec.retry < 0:
        return ProviderRunResult(False, "provider run requires a non-negative retry", "failed", 0, 0)
    if run_spec.retry_wait_seconds < 0:
        return ProviderRunResult(False, "provider run requires a non-negative retry_wait_seconds", "failed", 0, 0)
    if run_spec.backoff_multiplier < 1:
        return ProviderRunResult(False, "provider run requires backoff_multiplier >= 1", "failed", 0, 0)
    if run_spec.progress_every_tasks < 0:
        return ProviderRunResult(False, "provider run requires a non-negative progress interval", "failed", 0, 0)

    task_specs = adapter.plan_tasks(run_spec)
    _ensure_run(sqlite_path, run_spec, len(task_specs))
    _ensure_tasks(sqlite_path, task_specs)

    completed = 0
    last_result: TaskExecutionResult | None = None
    last_task: dict[str, object] | None = None
    last_request_at: float | None = None
    started_at = time.monotonic()

    for _ in range(run_spec.max_tasks):
        task = _next_task(sqlite_path, run_spec.run_id)
        if task is None:
            _mark_run_completed(sqlite_path, run_spec.run_id)
            break
        last_task = task
        result, last_request_at = _execute_task_with_retry(
            sqlite_path=sqlite_path,
            run_spec=run_spec,
            adapter=adapter,
            task=task,
            last_request_at=last_request_at,
        )
        if not result.ok:
            stats = _run_stats(sqlite_path, run_spec.run_id)
            reason = result.error.reason.value if result.error else "unknown"
            message = f"provider run failed: {run_spec.run_id} task={task['task_id']} reason={reason}"
            _mark_run_failed(sqlite_path, run_spec.run_id, result.error)
            return ProviderRunResult(False, message, "failed", completed, stats["rows"], result.raw_batch_id, result.error)

        completed += 1
        last_result = result
        _update_run_summary(sqlite_path, run_spec.run_id, "running")
        if progress_callback and run_spec.progress_every_tasks and completed % run_spec.progress_every_tasks == 0:
            progress_callback(
                _progress_message(
                    sqlite_path=sqlite_path,
                    run_spec=run_spec,
                    completed_this_invocation=completed,
                    elapsed_seconds=time.monotonic() - started_at,
                    last_task=task,
                    last_result=result,
                )
            )

    stats = _run_stats(sqlite_path, run_spec.run_id)
    status = "completed" if stats["remaining"] == 0 else "running"
    if status == "completed":
        _mark_run_completed(sqlite_path, run_spec.run_id)
    _update_run_summary(sqlite_path, run_spec.run_id, status)
    recent = _recent_raw_batches(sqlite_path, run_spec.run_id)
    message = (
        f"provider run: {run_spec.run_id} status={status} "
        f"tasks_success={stats['success']}/{stats['total']} "
        f"tasks_failed={stats['failed']} tasks_remaining={stats['remaining']} "
        f"rows={stats['rows']} tasks_this_run={completed} "
        f"recent_raw_batches={', '.join(recent)}"
    )
    return ProviderRunResult(
        True,
        message,
        status,
        completed,
        stats["rows"],
        last_result.raw_batch_id if last_result else None,
    )


def classify_tushare_error(error: Exception) -> ProviderTaskError:
    message = str(error)
    lowered = message.lower()
    if any(marker in lowered for marker in ("每分钟", "频次", "频率", "频率超限", "访问接口", "访问该接口", "最多访问", "超过", "limit", "rate")):
        return ProviderTaskError(
            ProviderErrorReason.RATE_LIMIT,
            "provider rate limit",
            provider_message=message,
            retryable=True,
        )
    if any(marker in lowered for marker in ("权限", "没有权限", "积分", "2002", "permission")):
        return ProviderTaskError(
            ProviderErrorReason.PERMISSION_DENIED,
            "provider permission denied",
            provider_message=message,
            retryable=False,
        )
    return ProviderTaskError(
        ProviderErrorReason.UNKNOWN,
        "provider task failed",
        provider_message=message,
        retryable=False,
    )


def _execute_task_with_retry(
    sqlite_path: Path,
    run_spec: ProviderRunSpec,
    adapter: ProviderTaskAdapter,
    task: dict[str, object],
    last_request_at: float | None,
) -> tuple[TaskExecutionResult, float | None]:
    max_attempts = run_spec.retry + 1
    for attempt in range(1, max_attempts + 1):
        _mark_task_running(sqlite_path, str(task["task_id"]))
        last_request_at = _throttle(run_spec.requests_per_minute, last_request_at)
        result = adapter.execute_task(task)
        if result.ok:
            _mark_task_success(sqlite_path, str(task["task_id"]), attempt, str(result.raw_batch_id), result.row_count)
            return result, last_request_at
        error = result.error or ProviderTaskError(ProviderErrorReason.UNKNOWN, "provider task failed", retryable=False)
        if not error.retryable or attempt >= max_attempts:
            _mark_task_failed(sqlite_path, str(task["task_id"]), attempt, error)
            return result, last_request_at
        wait_seconds = run_spec.retry_wait_seconds * (run_spec.backoff_multiplier ** (attempt - 1))
        time.sleep(wait_seconds)
    error = ProviderTaskError(ProviderErrorReason.UNKNOWN, "provider task failed unexpectedly")
    return TaskExecutionResult(False, error=error), last_request_at


def _ensure_run(sqlite_path: Path, run_spec: ProviderRunSpec, total_tasks: int) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        existing = connection.execute("SELECT run_id FROM provider_runs WHERE run_id = ?", (run_spec.run_id,)).fetchone()
        if existing:
            return
        connection.execute(
            """
            INSERT INTO provider_runs (
              run_id, source, dataset_name, start_date, end_date, as_of_date, status,
              total_symbols, next_offset, batch_size, requested_symbols, symbols_with_rows,
              failed_symbols, row_count, raw_batch_ids, failure_json, created_at, updated_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0, 0, 0, 0, 0, 0, '[]', '[]', ?, ?, ?)
            """,
            (
                run_spec.run_id,
                run_spec.source,
                run_spec.run_type,
                run_spec.start_date,
                run_spec.end_date,
                run_spec.as_of_date,
                total_tasks,
                now,
                now,
                "provider task engine",
            ),
        )


def _ensure_tasks(sqlite_path: Path, task_specs: list[ProviderTaskSpec]) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows = [
        (
            task.task_id,
            task.run_id,
            task.source,
            task.dataset_name,
            task.task_type,
            task.trade_date,
            task.symbol_start_offset,
            task.symbol_end_offset,
            task.start_date,
            task.end_date,
            json.dumps(task.payload or {}, ensure_ascii=False, sort_keys=True),
            now,
            now,
        )
        for task in task_specs
    ]
    with sqlite3.connect(sqlite_path) as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO provider_run_tasks (
              task_id, run_id, source, dataset_name, task_type, trade_date,
              symbol_start_offset, symbol_end_offset, start_date, end_date, payload_json,
              status, attempts, raw_batch_id, row_count, error_reason, error_message,
              provider_message, retryable, created_at, updated_at, started_at, finished_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, 0, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL, 'provider task engine')
            """,
            rows,
        )


def _next_task(sqlite_path: Path, run_id: str) -> dict[str, object] | None:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
              task_id, run_id, source, dataset_name, task_type, trade_date,
              symbol_start_offset, symbol_end_offset, start_date, end_date, payload_json, attempts
            FROM provider_run_tasks
            WHERE run_id = ? AND status IN ('pending', 'running', 'failed')
            ORDER BY COALESCE(trade_date, start_date, ''), dataset_name, COALESCE(symbol_start_offset, -1), task_id
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    keys = [
        "task_id",
        "run_id",
        "source",
        "dataset_name",
        "task_type",
        "trade_date",
        "symbol_start_offset",
        "symbol_end_offset",
        "start_date",
        "end_date",
        "payload_json",
        "attempts",
    ]
    return dict(zip(keys, row, strict=True))


def _mark_task_running(sqlite_path: Path, task_id: str) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'running', updated_at = ?, started_at = COALESCE(started_at, ?)
            WHERE task_id = ?
            """,
            (now, now, task_id),
        )


def _mark_task_success(sqlite_path: Path, task_id: str, attempts: int, batch_id: str, row_count: int) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'success',
                attempts = ?,
                raw_batch_id = ?,
                row_count = ?,
                error_reason = NULL,
                error_message = NULL,
                provider_message = NULL,
                retryable = NULL,
                updated_at = ?,
                finished_at = ?
            WHERE task_id = ?
            """,
            (attempts, batch_id, row_count, now, now, task_id),
        )


def _mark_task_failed(sqlite_path: Path, task_id: str, attempts: int, error: ProviderTaskError) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_run_tasks
            SET status = 'failed',
                attempts = ?,
                error_reason = ?,
                error_message = ?,
                provider_message = ?,
                retryable = ?,
                updated_at = ?,
                finished_at = ?
            WHERE task_id = ?
            """,
            (
                attempts,
                error.reason.value,
                error.message,
                error.provider_message,
                int(error.retryable),
                now,
                now,
                task_id,
            ),
        )


def _run_stats(sqlite_path: Path, run_id: str) -> dict[str, int]:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute(
            """
            SELECT
              COUNT(*),
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
              SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END),
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END),
              SUM(row_count)
            FROM provider_run_tasks
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    return {
        "total": int(row[0] or 0),
        "success": int(row[1] or 0),
        "failed": int(row[2] or 0),
        "remaining": int(row[3] or 0),
        "rows": int(row[4] or 0),
    }


def _update_run_summary(sqlite_path: Path, run_id: str, status: str) -> None:
    stats = _run_stats(sqlite_path, run_id)
    raw_batch_ids = _all_raw_batches(sqlite_path, run_id)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE provider_runs
            SET status = ?,
                next_offset = ?,
                requested_symbols = ?,
                symbols_with_rows = ?,
                failed_symbols = ?,
                row_count = ?,
                raw_batch_ids = ?,
                updated_at = ?
            WHERE run_id = ?
            """,
            (
                status,
                stats["success"],
                stats["success"],
                stats["success"],
                stats["failed"],
                stats["rows"],
                json.dumps(raw_batch_ids),
                now,
                run_id,
            ),
        )


def _mark_run_completed(sqlite_path: Path, run_id: str) -> None:
    _update_run_summary(sqlite_path, run_id, "completed")


def _mark_run_failed(sqlite_path: Path, run_id: str, error: ProviderTaskError | None) -> None:
    with sqlite3.connect(sqlite_path) as connection:
        row = connection.execute("SELECT failure_json FROM provider_runs WHERE run_id = ?", (run_id,)).fetchone()
        failures = json.loads(str(row[0])) if row and row[0] else []
        if error:
            failures.append(
                {
                    "reason": error.reason.value,
                    "message": error.message,
                    "provider_message": error.provider_message,
                    "retryable": error.retryable,
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        connection.execute(
            """
            UPDATE provider_runs
            SET status = 'failed',
                failed_symbols = (SELECT COUNT(*) FROM provider_run_tasks WHERE run_id = ? AND status = 'failed'),
                row_count = (SELECT COALESCE(SUM(row_count), 0) FROM provider_run_tasks WHERE run_id = ?),
                raw_batch_ids = ?,
                failure_json = ?,
                updated_at = ?
            WHERE run_id = ?
            """,
            (
                run_id,
                run_id,
                json.dumps(_all_raw_batches(sqlite_path, run_id)),
                json.dumps(failures, ensure_ascii=False),
                datetime.now(UTC).isoformat(timespec="seconds"),
                run_id,
            ),
        )


def _all_raw_batches(sqlite_path: Path, run_id: str) -> list[str]:
    with sqlite3.connect(sqlite_path) as connection:
        rows = connection.execute(
            """
            SELECT raw_batch_id
            FROM provider_run_tasks
            WHERE run_id = ? AND raw_batch_id IS NOT NULL
            ORDER BY COALESCE(trade_date, start_date, ''), dataset_name, COALESCE(symbol_start_offset, -1), task_id
            """,
            (run_id,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _recent_raw_batches(sqlite_path: Path, run_id: str) -> list[str]:
    return _all_raw_batches(sqlite_path, run_id)[-5:]


def _progress_message(
    sqlite_path: Path,
    run_spec: ProviderRunSpec,
    completed_this_invocation: int,
    elapsed_seconds: float,
    last_task: dict[str, object],
    last_result: TaskExecutionResult,
) -> str:
    stats = _run_stats(sqlite_path, run_spec.run_id)
    rate = completed_this_invocation / elapsed_seconds * 60 if elapsed_seconds > 0 else 0.0
    symbol_range = ""
    if last_task.get("symbol_start_offset") is not None and last_task.get("symbol_end_offset") is not None:
        symbol_range = f" symbols={last_task['symbol_start_offset']}-{last_task['symbol_end_offset']}"
    date_part = last_task.get("trade_date") or f"{last_task.get('start_date')}-{last_task.get('end_date')}"
    return (
        f"progress run_id={run_spec.run_id} "
        f"this_invocation={completed_this_invocation}/{run_spec.max_tasks} "
        f"total_success={stats['success']}/{stats['total']} "
        f"failed={stats['failed']} remaining={stats['remaining']} "
        f"rows={stats['rows']} "
        f"last={last_task['dataset_name']}/{date_part}{symbol_range} "
        f"last_rows={last_result.row_count} "
        f"rate={rate:.1f}_tasks_per_min"
    )


def _throttle(requests_per_minute: float, last_request_at: float | None) -> float:
    if requests_per_minute <= 0:
        return time.monotonic()
    interval = 60.0 / requests_per_minute
    now = time.monotonic()
    if last_request_at is not None:
        elapsed = now - last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
    return time.monotonic()
