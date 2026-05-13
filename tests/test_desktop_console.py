from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sys

import polars as pl
import pytest

from stock_picker.cli import main as cli_main
from stock_picker.analysis import analyze_stock
from stock_picker.app_worker import run_daily_check, run_holding_price_refresh, run_worker_once
from stock_picker.curated import promote_raw_batch
from stock_picker.provider import probe_provider_api, write_raw_batch
from stock_picker.publish import build_candidate_pool, build_daily_bundle, build_market_status, validate_daily_bundle
from stock_picker.storage import init_storage, register_schemas
from stock_picker.workflow import pause_workflow, stock_analysis_workflow, sync_report_workflow, workflow_status


def test_daily_bundle_and_stock_analysis(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    _write_market_status_inputs(tmp_path)

    publish = build_daily_bundle(config_path, "factor_002_test", trade_date="2026-04-28", top=2)

    assert publish.ok
    assert publish.artifact_path is not None
    artifact = json.loads(publish.artifact_path.read_text(encoding="utf-8"))
    assert validate_daily_bundle(artifact) == []
    assert artifact["bundle_metadata"]["schema_version"] == "daily_publish_bundle_v001"
    assert artifact["market_status"]["market_status_metadata"]["schema_version"] == "market_status_v001"
    assert artifact["candidate_pool"]["candidate_pool_metadata"]["schema_version"] == "candidate_pool_v001"
    assert artifact["candidate_pool"]["top_stocks"][0]["symbol"] == "600519.SH"
    assert "bundle_hash" in artifact["bundle_metadata"]
    assert str(tmp_path) not in json.dumps(artifact, ensure_ascii=False)

    analysis = analyze_stock(config_path, "600519.SH", "factor_002_test", trade_date="2026-04-28")

    assert analysis.ok
    assert analysis.artifact_path is not None
    payload = json.loads(analysis.artifact_path.read_text(encoding="utf-8"))
    assert payload["analysis_metadata"]["schema_version"] == "stock_app_stock_analysis_v001"
    assert payload["candidate_status"]["in_latest_candidate_pool"] is True
    assert "买入" not in json.dumps(payload, ensure_ascii=False)


def test_stock_analysis_handles_missing_symbol(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")

    result = analyze_stock(config_path, "000000.SZ", "factor_002_test", trade_date="2026-04-28")

    assert result.ok
    assert result.artifact is not None
    assert result.artifact["candidate_status"]["exclusion_reason"] == "not_found_in_factor_run"


def test_daily_bundle_validation_rejects_missing_sections_and_unsafe_text() -> None:
    errors = validate_daily_bundle(
        {
            "bundle_metadata": {"schema_version": "daily_publish_bundle_v001"},
            "market_status": {"market_status_metadata": {"schema_version": "market_status_v001"}, "index_markets": {}, "sectors": {}, "structural_risks": [{"summary": "可以买入"}]},
            "candidate_pool": {"candidate_pool_metadata": {"schema_version": "candidate_pool_v001"}, "strategy": {}, "summary": {}, "top_stocks": [{"local_path": "C:/secret", "raw_payload": {"x": 1}}], "diff": {}},
        }
    )

    assert "prohibited wording: 买入" in errors
    assert "forbidden sensitive field: raw_payload" in errors
    assert "forbidden local absolute path" in errors


def test_candidate_pool_change_labels_from_previous_pool(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run_with_four_tradable(tmp_path, "factor_002_test")
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "candidate_pool_metadata": {"version_id": "candidate_pool_previous"},
                "top_stocks": [
                    {"symbol": "600519.SH", "rank": 3},
                    {"symbol": "300750.SZ", "rank": 2},
                    {"symbol": "688001.SH", "rank": 1},
                    {"symbol": "600000.SH", "rank": 4},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_candidate_pool(config_path, "factor_002_test", trade_date="2026-04-28", previous_candidate_pool_path=previous, top=4)

    assert result.ok
    assert result.artifact is not None
    changes = {item["symbol"]: item["change_status"] for item in result.artifact["top_stocks"]}
    assert changes == {"600519.SH": "upgraded", "300750.SZ": "retained", "688001.SH": "downgraded", "000002.SZ": "newly_added"}
    ranks = {item["symbol"]: (item["current_rank"], item.get("previous_rank")) for item in result.artifact["top_stocks"]}
    assert ranks["600519.SH"] == (1, 3)
    assert ranks["300750.SZ"] == (2, 2)
    assert result.artifact["diff"]["removed"][0]["symbol"] == "600000.SH"
    assert result.artifact["diff"]["removed"][0]["previous_rank"] == 4


def test_candidate_pool_rejects_invalid_inputs_and_empty_factor_run(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok

    assert not build_candidate_pool(config_path, "missing", top=1).ok
    assert not build_candidate_pool(config_path, "missing", top=0).ok

    _write_factor_run(tmp_path, "empty_factor")
    run_dir = tmp_path / "data" / "reports" / "factor_exploration" / "empty_factor"
    (run_dir / "stock_factors_daily.csv").write_text("symbol,trade_date,total_score,tradable_flag\n", encoding="utf-8")

    result = build_candidate_pool(config_path, "empty_factor", top=1)

    assert not result.ok
    assert "factor run has no rows" in result.message


def test_market_status_identifies_index_and_sector_strength(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_market_status_inputs(tmp_path)

    result = build_market_status(config_path, trade_date="2026-04-28")

    assert result.ok
    assert result.artifact is not None
    payload = result.artifact
    assert payload["market_status_metadata"]["schema_version"] == "market_status_v001"
    assert payload["index_markets"]["strongest"]["name"] == "中证500"
    assert payload["index_markets"]["strongest"]["return_1d"] == 1.4
    assert payload["index_markets"]["weakest"]["name"] == "创业板指"
    assert payload["index_markets"]["weakest"]["return_1d"] == -1.1
    assert payload["sectors"]["strongest"] == ["通信", "机械设备", "电子"]
    assert payload["sectors"]["weakest"] == ["房地产", "医药生物", "食品饮料"]
    assert payload["sectors"]["summary_text"] != "强势方向集中在通信、机械设备、电子，弱势方向集中在房地产、医药生物、食品饮料。"
    assert {risk["risk_type"] for risk in payload["structural_risks"]}


def test_market_status_handles_missing_industry_data_as_partial(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_index_daily_current(tmp_path)

    result = build_market_status(config_path, trade_date="2026-04-28")

    assert result.ok
    assert result.artifact is not None
    assert result.artifact["market_status_metadata"]["status"] == "partial"
    assert any(risk["risk_type"] == "data_insufficient" for risk in result.artifact["structural_risks"])


def test_promote_shenwan_industry_raw_batches(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _copy_curated_schemas(tmp_path, ["industry_classification.yaml", "industry_daily.yaml"])
    assert register_schemas(config_path).ok
    write_raw_batch(
        config_path,
        "tushare",
        "index_classify",
        pl.DataFrame({"index_code": ["801770.SI"], "industry_name": ["通信"], "level": ["L1"], "src": ["SW2021"]}),
        "2026-04-28",
    )
    write_raw_batch(
        config_path,
        "tushare",
        "sw_daily",
        pl.DataFrame({"ts_code": ["801770.SI"], "name": ["通信"], "trade_date": ["20260428"], "close": [1234.5], "pct_change": [2.4], "vol": [100.0], "amount": [200.0]}),
        "2026-04-28",
    )

    classify = promote_raw_batch(config_path, "tushare", "index_classify", "2026-04-28")
    daily = promote_raw_batch(config_path, "tushare", "sw_daily", "2026-04-28")

    assert classify.ok
    assert daily.ok
    classification = pl.read_parquet(tmp_path / "data" / "curated" / "current" / "industry_classification" / "part-000.parquet")
    industry_daily = pl.read_parquet(tmp_path / "data" / "curated" / "current" / "industry_daily" / "part-000.parquet")
    assert classification.to_dicts()[0]["industry_name"] == "通信"
    assert industry_daily.to_dicts()[0]["industry_name"] == "通信"


def test_provider_probe_supports_shenwan_industry_apis(monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "tushare", _FakeTushare)

    classify = probe_provider_api("tushare", "index_classify")
    daily = probe_provider_api("tushare", "sw_daily", ts_code="801770.SI", trade_date="20260428")

    assert classify.ok
    assert daily.ok
    assert "missing_expected_fields: none" in classify.message
    assert "missing_expected_fields: none" in daily.message


def test_stock_analysis_rejects_missing_and_empty_factor_run(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok

    assert not analyze_stock(config_path, "600519.SH", "missing").ok

    _write_factor_run(tmp_path, "empty_factor")
    run_dir = tmp_path / "data" / "reports" / "factor_exploration" / "empty_factor"
    (run_dir / "stock_factors_daily.csv").write_text("symbol,trade_date,total_score,tradable_flag\n", encoding="utf-8")

    result = analyze_stock(config_path, "600519.SH", "empty_factor")

    assert not result.ok
    assert "factor run has no rows" in result.message


def test_stock_analysis_surfaces_risk_notes_and_avoids_sensitive_output(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")

    result = analyze_stock(config_path, "300001.SZ", "factor_002_test", trade_date="2026-04-28")

    assert result.ok
    assert result.artifact is not None
    text = json.dumps(result.artifact, ensure_ascii=False)
    assert "low_liquidity" in result.artifact["risk_notes"]
    assert "Recent volatility or drawdown score is weak." in result.artifact["risk_notes"]
    assert str(tmp_path) not in text
    assert "买入" not in text


def test_worker_run_once_processes_mock_stock_analysis(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("default_factor_run_id: factor_002_test\n", encoding="utf-8")
    mock_task = tmp_path / "task.json"
    mock_task.write_text(
        json.dumps(
            {
                "analysis_request": {
                    "id": "request_001",
                    "request_type": "stock_analysis",
                    "symbol": "600519.SH",
                    "report_date": "2026-04-28",
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_worker_once(config_path, worker_config, mock_task)

    assert result.ok
    response = json.loads(mock_task.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["result_artifact_json"]["analysis_metadata"]["schema_version"] == "stock_app_stock_analysis_v001"
    assert response["result_artifact_json"]["stock"]["symbol"] == "600519.SH"


def test_worker_run_once_processes_bom_stock_analysis_mock_task(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("default_factor_run_id: factor_002_test\n", encoding="utf-8")
    mock_task = tmp_path / "task.json"
    mock_task.write_text(
        "\ufeff" + json.dumps({"analysis_request": {"id": "request_002", "request_type": "stock_analysis", "symbol": "600519.SH", "report_date": "2026-04-28"}}),
        encoding="utf-8",
    )

    result = run_worker_once(config_path, worker_config, mock_task)

    assert result.ok
    response = json.loads(mock_task.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["result_artifact_json"]["stock"]["symbol"] == "600519.SH"


def test_worker_run_once_accepts_stock_app_camel_case_task(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("default_factor_run_id: factor_002_test\n", encoding="utf-8")
    mock_task = tmp_path / "task.json"
    mock_task.write_text(
        json.dumps({"analysis_request": {"id": "request_002b", "requestType": "stock_analysis", "symbol": "600519.SH", "reportDate": "2026-04-28"}}),
        encoding="utf-8",
    )

    result = run_worker_once(config_path, worker_config, mock_task)

    assert result.ok
    response = json.loads(mock_task.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert response["status"] == "completed"
    assert response["result_artifact_json"]["stock"]["symbol"] == "600519.SH"


def test_daily_check_mock_uploads_bundle_and_skips_duplicate_hash(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run_with_ten_tradable(tmp_path, "factor_002_test")
    _write_market_status_inputs(tmp_path)
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("worker_id: local-test-worker\ndefault_factor_run_id: factor_002_test\n", encoding="utf-8")
    upload_path = tmp_path / "daily-upload.json"

    first = run_daily_check(config_path, worker_config, trade_date="2026-04-28", mock_upload=True, mock_upload_path=upload_path, top=10)
    second = run_daily_check(config_path, worker_config, trade_date="2026-04-28", mock_upload=True, mock_upload_path=upload_path, top=10)

    assert first.ok
    assert second.ok
    assert "already uploaded" in second.message
    response = json.loads(upload_path.read_text(encoding="utf-8"))
    assert response["bundle_metadata"]["schema_version"] == "daily_publish_bundle_v001"
    assert response["candidate_pool"]["candidate_pool_metadata"]["schema_version"] == "candidate_pool_v001"
    assert len(response["candidate_pool"]["top_stocks"]) == 10
    state = json.loads((tmp_path / "data" / "reports" / "app_worker" / "daily_upload_state.json").read_text(encoding="utf-8"))
    assert state["daily_bundle_uploads"]["2026-04-28"]["artifact_hash"] == response["bundle_metadata"]["bundle_hash"]


def test_daily_check_rejects_upload_when_candidate_pool_is_not_top_10(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    _write_market_status_inputs(tmp_path)
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("default_factor_run_id: factor_002_test\n", encoding="utf-8")

    result = run_daily_check(config_path, worker_config, trade_date="2026-04-28", mock_upload=True, top=2)

    assert not result.ok
    assert "exactly 10" in result.message


def test_daily_check_requires_factor_run_id(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("", encoding="utf-8")

    result = run_daily_check(config_path, worker_config, mock_upload=True)

    assert not result.ok
    assert "requires factor_run_id" in result.message


def test_worker_run_once_writes_failed_mock_results(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("", encoding="utf-8")
    mock_task = tmp_path / "task.json"
    mock_task.write_text(json.dumps({"analysis_request": {"id": "request_003", "request_type": "unknown"}}), encoding="utf-8")

    result = run_worker_once(config_path, worker_config, mock_task)

    assert not result.ok
    response = json.loads(mock_task.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert response["status"] == "failed"
    assert "unsupported request_type" in response["error_message"]


def test_worker_run_once_requires_factor_run_id_for_mock_task(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("", encoding="utf-8")
    mock_task = tmp_path / "task.json"
    mock_task.write_text(json.dumps({"analysis_request": {"id": "request_004", "request_type": "stock_analysis", "symbol": "600519.SH"}}), encoding="utf-8")

    result = run_worker_once(config_path, worker_config, mock_task)

    assert not result.ok
    response = json.loads(mock_task.with_suffix(".result.json").read_text(encoding="utf-8"))
    assert "requires factor_run_id" in response["error_message"]


def test_worker_run_once_reports_missing_http_token(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("worker_token_env: TEST_STOCK_WORKER_TOKEN\n", encoding="utf-8")
    monkeypatch.delenv("TEST_STOCK_WORKER_TOKEN", raising=False)

    result = run_worker_once(config_path, worker_config)

    assert not result.ok
    assert "missing required environment variable: TEST_STOCK_WORKER_TOKEN" in result.message


def test_daily_check_reports_missing_publisher_token(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run_with_ten_tradable(tmp_path, "factor_002_test")
    _write_market_status_inputs(tmp_path)
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.write_text("publisher_token_env: TEST_STOCK_PUBLISHER_TOKEN\ndefault_factor_run_id: factor_002_test\n", encoding="utf-8")
    monkeypatch.delenv("TEST_STOCK_PUBLISHER_TOKEN", raising=False)
    monkeypatch.delenv("PUBLISHER_TOKEN", raising=False)

    result = run_daily_check(config_path, worker_config, trade_date="2026-04-28")

    assert not result.ok
    assert "missing required environment variable: TEST_STOCK_PUBLISHER_TOKEN" in result.message


def test_holding_price_refresh_mock_watchlist_and_upload(tmp_path: Path, monkeypatch) -> None:
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.parent.mkdir(parents=True)
    worker_config.write_text("worker_id: local-test-worker\n", encoding="utf-8")
    watchlist = tmp_path / "watchlist.json"
    watchlist.write_text(json.dumps({"symbols": [{"symbol": "600519.SH", "name": "Kweichow Moutai"}]}), encoding="utf-8")
    upload_path = tmp_path / "holding-prices.json"
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "tushare", _FakeTushare)

    result = run_holding_price_refresh(worker_config, trade_date="2026-04-28", mock_watchlist_path=watchlist, mock_upload_path=upload_path)

    assert result.ok
    payload = json.loads(upload_path.read_text(encoding="utf-8"))
    assert payload["prices"][0]["symbol"] == "600519.SH"
    assert payload["prices"][0]["last_price"] == 1688.0
    assert payload["prices"][0]["change_percent"] == 1.23
    assert payload["prices"][0]["quote_time"] == "2026-04-28T15:00:00+08:00"
    assert payload["prices"][0]["source"] == "tushare.daily"


def test_holding_price_refresh_handles_empty_watchlist_and_missing_token(tmp_path: Path, monkeypatch) -> None:
    worker_config = tmp_path / "config" / "app-worker.yaml"
    worker_config.parent.mkdir(parents=True)
    worker_config.write_text("", encoding="utf-8")
    watchlist = tmp_path / "watchlist.json"
    watchlist.write_text(json.dumps({"symbols": []}), encoding="utf-8")

    empty = run_holding_price_refresh(worker_config, mock_watchlist_path=watchlist)
    assert empty.ok
    assert "watchlist empty" in empty.message

    watchlist.write_text(json.dumps({"symbols": [{"symbol": "600519.SH"}]}), encoding="utf-8")
    monkeypatch.delenv("MISSING_TUSHARE_TOKEN", raising=False)
    missing = run_holding_price_refresh(worker_config, token_env="MISSING_TUSHARE_TOKEN", mock_watchlist_path=watchlist)
    assert not missing.ok
    assert "missing required environment variable: MISSING_TUSHARE_TOKEN" in missing.message


def test_workflow_status_pause_and_stock_analysis(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")

    result = stock_analysis_workflow(config_path, symbol="600519.SH", factor_run_id="factor_002_test", workflow_id="wf_stock", trade_date="2026-04-28")

    assert result.ok
    status = workflow_status(config_path, "wf_stock")
    assert status.ok
    assert '"status": "completed"' in status.message
    paused = pause_workflow(config_path, "wf_stock")
    assert paused.ok
    assert '"status": "paused"' in workflow_status(config_path, "wf_stock").message


def test_workflow_resume_skips_completed_steps_and_emits_events(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_factor_run(tmp_path, "factor_002_test")
    events: list[dict[str, object]] = []

    first = stock_analysis_workflow(config_path, symbol="600519.SH", factor_run_id="factor_002_test", workflow_id="wf_resume", trade_date="2026-04-28", emit=events.append)
    second = stock_analysis_workflow(config_path, symbol="600519.SH", factor_run_id="factor_002_test", workflow_id="wf_resume", trade_date="2026-04-28", emit=events.append)

    assert first.ok
    assert second.ok
    event_names = [event["event"] for event in events]
    assert "step_started" in event_names
    assert "step_completed" in event_names
    assert "step_skipped" in event_names


def test_workflow_failed_step_records_state(tmp_path: Path) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok

    result = stock_analysis_workflow(config_path, symbol="600519.SH", factor_run_id="missing", workflow_id="wf_failed", trade_date="2026-04-28")

    assert not result.ok
    status = json.loads(workflow_status(config_path, "wf_failed").message)
    assert status["status"] == "failed"
    assert status["steps"]["analyze_stock"]["status"] == "failed"
    assert status["errors"]


def test_sync_report_workflow_dry_run_requires_confirmation(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_storage_config(tmp_path)
    assert init_storage(config_path).ok
    _write_trading_calendar_current(tmp_path)
    _write_daily_prices_current(tmp_path, [date(2026, 4, 28)])
    _write_capital_flow_current(
        tmp_path,
        [{"symbol": "600519.SH", "trade_date": date(2026, 4, 28), "main_net_inflow_rate": 1.0, "close_profit_ratio": 90.0, "data_method": "tushare_cyq_perf"}],
    )
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "tushare", _FakeTushare)
    events: list[dict[str, object]] = []

    result = sync_report_workflow(config_path, workflow_id="wf_sync_dry_run", dry_run=True, end_date="2026-05-01", emit=events.append)

    assert result.ok
    assert "dry_run: no data fetched or promoted" in result.message
    state = json.loads(workflow_status(config_path, "wf_sync_dry_run").message)
    assert state["requires_confirmation"] is True
    assert state["steps"]["sync_dry_run"]["status"] == "completed"
    assert [event["event"] for event in events] == ["workflow_started", "step_started", "step_completed"]


def test_new_cli_help_commands_are_available(capsys) -> None:
    commands = [
        ["publish", "build-market-status", "--help"],
        ["publish", "build-candidate-pool", "--help"],
        ["publish", "build-daily-bundle", "--help"],
        ["analysis", "stock", "--help"],
        ["workflow", "sync-report", "--help"],
        ["app-worker", "run-once", "--help"],
        ["app-worker", "daily-check", "--help"],
        ["app-worker", "refresh-holding-prices", "--help"],
    ]

    for command in commands:
        with pytest.raises(SystemExit) as error:
            cli_main(command)
        assert error.value.code == 0

    output = capsys.readouterr().out
    assert "build-market-status" in output
    assert "build-candidate-pool" in output
    assert "build-daily-bundle" in output
    assert "stock-picker analysis stock" in output
    assert "stock-picker workflow sync-report" in output
    assert "stock-picker app-worker run-once" in output
    assert "daily-check" in output
    assert "refresh-holding-prices" in output


def _write_storage_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "storage.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                "  data_root: ./data",
                "  raw_root: ${storage.data_root}/raw",
                "  curated_root: ${storage.data_root}/curated",
                "  current_curated_root: ${storage.curated_root}/current",
                "  frozen_curated_root: ${storage.curated_root}/frozen",
                "  reports_root: ${storage.data_root}/reports",
                "  backtests_root: ${storage.data_root}/backtests",
                "  metadata_sqlite_path: ${storage.data_root}/metadata.sqlite",
                "  schema_root: ./schemas",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_factor_run(tmp_path: Path, factor_run_id: str) -> None:
    run_dir = tmp_path / "data" / "reports" / "factor_exploration" / factor_run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "factor_run_id": factor_run_id,
        "snapshot_id": "snapshot_20260428_001",
        "factor_version": "flow_momentum_quality_v001",
        "start_date": "2026-04-27",
        "end_date": "2026-04-28",
        "created_at": "2026-04-28T16:00:00+00:00",
        "dataset_paths": {},
    }
    (run_dir / "factor_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "000001.SZ", "300001.SZ"],
            "trade_date": [date(2026, 4, 28), date(2026, 4, 28), date(2026, 4, 28)],
            "name": ["Kweichow Moutai", "Ping An Bank", "Tech Sample"],
            "market_segment": ["sh_main", "sz_main", "chinext"],
            "industry": ["Food", "Bank", "Tech"],
            "mom_20": [0.1, 0.05, -0.02],
            "mom_60": [0.2, 0.04, 0.01],
            "ma_strength": [0.03, 0.02, -0.01],
            "flow_5d": [3.0, 2.0, 1.0],
            "flow_20d": [6.0, 1.5, 0.5],
            "flow_persistence": [0.8, 0.7, 0.4],
            "winner_rate": [82.0, 75.0, 60.0],
            "winner_rate_change_5d": [2.0, 1.0, -1.0],
            "vol_20": [0.02, 0.03, 0.05],
            "max_drawdown_60": [-0.05, -0.08, -0.2],
            "avg_amount_20": [90000.0, 80000.0, 10000.0],
            "momentum_score": [0.9, 0.7, 0.2],
            "flow_score": [0.9, 0.75, 0.3],
            "chip_score": [0.95, 0.8, 0.4],
            "risk_score": [0.8, 0.7, 0.2],
            "liquidity_score": [0.9, 0.8, 0.1],
            "total_score": [0.89, 0.74, 0.24],
            "tradable_flag": [True, True, False],
            "exclusion_reason": ["none", "none", "low_liquidity"],
            "factor_version": ["flow_momentum_quality_v001"] * 3,
        }
    ).write_csv(run_dir / "stock_factors_daily.csv")


def _write_factor_run_with_four_tradable(tmp_path: Path, factor_run_id: str) -> None:
    run_dir = tmp_path / "data" / "reports" / "factor_exploration" / factor_run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "factor_run_id": factor_run_id,
        "snapshot_id": "snapshot_20260428_001",
        "factor_version": "flow_momentum_quality_v001",
        "start_date": "2026-04-27",
        "end_date": "2026-04-28",
        "created_at": "2026-04-28T16:00:00+00:00",
        "dataset_paths": {},
    }
    (run_dir / "factor_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pl.DataFrame(
        {
            "symbol": ["600519.SH", "300750.SZ", "688001.SH", "000002.SZ"],
            "trade_date": [date(2026, 4, 28)] * 4,
            "name": ["Kweichow Moutai", "CATL", "Star Sample", "Vanke A"],
            "market_segment": ["sh_main", "chinext", "star", "sz_main"],
            "industry": ["食品饮料", "电力设备", "电子", "房地产"],
            "mom_20": [0.1, 0.08, 0.04, 0.02],
            "mom_60": [0.2, 0.1, 0.08, 0.01],
            "ma_strength": [0.03, 0.025, 0.02, 0.01],
            "flow_5d": [3.0, 2.0, 1.0, 0.5],
            "flow_20d": [6.0, 3.0, 2.0, 1.0],
            "flow_persistence": [0.8, 0.7, 0.6, 0.5],
            "winner_rate": [82.0, 78.0, 72.0, 65.0],
            "winner_rate_change_5d": [2.0, 1.5, 1.0, 0.5],
            "vol_20": [0.02, 0.025, 0.03, 0.035],
            "max_drawdown_60": [-0.05, -0.07, -0.09, -0.11],
            "avg_amount_20": [90000.0, 85000.0, 70000.0, 60000.0],
            "momentum_score": [0.9, 0.8, 0.65, 0.5],
            "flow_score": [0.9, 0.78, 0.62, 0.5],
            "chip_score": [0.95, 0.82, 0.7, 0.55],
            "risk_score": [0.8, 0.75, 0.7, 0.6],
            "liquidity_score": [0.9, 0.85, 0.7, 0.65],
            "total_score": [0.89, 0.8, 0.7, 0.58],
            "tradable_flag": [True, True, True, True],
            "exclusion_reason": ["none", "none", "none", "none"],
            "factor_version": ["flow_momentum_quality_v001"] * 4,
        }
    ).write_csv(run_dir / "stock_factors_daily.csv")


def _write_factor_run_with_ten_tradable(tmp_path: Path, factor_run_id: str) -> None:
    run_dir = tmp_path / "data" / "reports" / "factor_exploration" / factor_run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "factor_run_id": factor_run_id,
        "snapshot_id": "snapshot_20260428_001",
        "factor_version": "flow_momentum_quality_v001",
        "start_date": "2026-04-27",
        "end_date": "2026-04-28",
        "created_at": "2026-04-28T16:00:00+00:00",
        "dataset_paths": {},
    }
    (run_dir / "factor_run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    symbols = [f"6005{index:02d}.SH" for index in range(10)]
    pl.DataFrame(
        {
            "symbol": symbols,
            "trade_date": [date(2026, 4, 28)] * 10,
            "name": [f"Sample {index}" for index in range(10)],
            "market_segment": ["sh_main"] * 10,
            "industry": ["通信", "机械设备", "电子", "银行", "汽车", "煤炭", "计算机", "食品饮料", "医药生物", "公用事业"],
            "mom_20": [0.1] * 10,
            "mom_60": [0.2] * 10,
            "ma_strength": [0.03] * 10,
            "flow_5d": [3.0] * 10,
            "flow_20d": [6.0] * 10,
            "flow_persistence": [0.8] * 10,
            "winner_rate": [82.0] * 10,
            "winner_rate_change_5d": [2.0] * 10,
            "vol_20": [0.02] * 10,
            "max_drawdown_60": [-0.05] * 10,
            "avg_amount_20": [90000.0] * 10,
            "momentum_score": [0.9] * 10,
            "flow_score": [0.9] * 10,
            "chip_score": [0.95] * 10,
            "risk_score": [0.8] * 10,
            "liquidity_score": [0.9] * 10,
            "total_score": [1.0 - index * 0.01 for index in range(10)],
            "tradable_flag": [True] * 10,
            "exclusion_reason": ["none"] * 10,
            "factor_version": ["flow_momentum_quality_v001"] * 10,
        }
    ).write_csv(run_dir / "stock_factors_daily.csv")


def _write_market_status_inputs(tmp_path: Path) -> None:
    _write_index_daily_current(tmp_path)
    industry_path = tmp_path / "data" / "curated" / "current" / "industry_daily" / "part-000.parquet"
    industry_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "index_code": ["801770.SI", "801890.SI", "801080.SI", "801180.SI", "801150.SI", "801120.SI"],
            "industry_name": ["通信", "机械设备", "电子", "房地产", "医药生物", "食品饮料"],
            "trade_date": [date(2026, 4, 28)] * 6,
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "pct_change": [2.6, 2.1, 1.8, -2.4, -1.9, -1.4],
        }
    ).write_parquet(industry_path)


def _write_index_daily_current(tmp_path: Path) -> None:
    daily_path = tmp_path / "data" / "curated" / "current" / "daily_prices" / "part-000.parquet"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    symbols = ["000300.SH", "000905.SH", "000852.SH", "399006.SZ", "000688.SH", "899050.BJ"]
    rows = []
    for day_index, trade_date in enumerate([date(2026, 4, 24), date(2026, 4, 27), date(2026, 4, 28)]):
        for symbol in symbols:
            latest_pct = {"000300.SH": 0.2, "000905.SH": 1.4, "000852.SH": 0.6, "399006.SZ": -1.1, "000688.SH": -0.7, "899050.BJ": -0.2}[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "asset_type": "index",
                    "close": 100.0 + day_index + (latest_pct if trade_date == date(2026, 4, 28) else 0.0),
                    "pct_change": latest_pct if trade_date == date(2026, 4, 28) else 0.1,
                }
            )
    pl.DataFrame(rows).write_parquet(daily_path)


def _copy_curated_schemas(tmp_path: Path, names: list[str]) -> None:
    target = tmp_path / "schemas" / "curated"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = Path("schemas") / "curated" / name
        (target / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_trading_calendar_current(tmp_path: Path) -> None:
    calendar_path = tmp_path / "data" / "curated" / "current" / "trading_calendar" / "part-000.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "calendar_id": ["cn_a_share"],
            "trade_date": [date(2026, 4, 28)],
            "is_trading_day": [True],
        }
    ).write_parquet(calendar_path)


def _write_daily_prices_current(tmp_path: Path, trade_dates: list[date]) -> None:
    daily_path = tmp_path / "data" / "curated" / "current" / "daily_prices" / "part-000.parquet"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": ["600519.SH"] * len(trade_dates),
            "trade_date": trade_dates,
            "asset_type": ["stock"] * len(trade_dates),
            "close": [10.0] * len(trade_dates),
        }
    ).write_parquet(daily_path)


def _write_capital_flow_current(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    flow_path = tmp_path / "data" / "curated" / "current" / "capital_flow_or_chip" / "part-000.parquet"
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(flow_path)


class _FakeTushare:
    @staticmethod
    def pro_api(token):
        assert token == "test-token"
        return _FakePro()


class _FakePro:
    def trade_cal(self, exchange=None, start_date=None, end_date=None):
        return pl.DataFrame(
            {
                "exchange": ["SSE", "SSE", "SSE"],
                "cal_date": ["20260429", "20260430", "20260501"],
                "is_open": [1, 1, 0],
                "pretrade_date": ["20260428", "20260429", "20260430"],
            }
        ).to_pandas()

    def index_classify(self, level=None, src=None):
        assert level == "L1"
        assert src == "SW2021"
        return pl.DataFrame({"index_code": ["801770.SI"], "industry_name": ["通信"], "level": ["L1"]}).to_pandas()

    def sw_daily(self, ts_code=None, trade_date=None, start_date=None, end_date=None):
        assert ts_code == "801770.SI"
        assert trade_date == "20260428"
        return pl.DataFrame({"ts_code": ["801770.SI"], "trade_date": ["20260428"], "close": [1234.5], "pct_change": [2.4]}).to_pandas()

    def daily(self, ts_code=None, trade_date=None):
        assert ts_code == "600519.SH"
        assert trade_date == "20260428"
        return pl.DataFrame({"ts_code": ["600519.SH"], "trade_date": ["20260428"], "close": [1688.0], "pct_chg": [1.23]}).to_pandas()
