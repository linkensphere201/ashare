from __future__ import annotations

from pathlib import Path

from stock_picker.app_worker import run_daily_check, run_holding_price_loop, run_holding_price_refresh, run_manual_stock_analysis, run_worker_loop, run_worker_once


def execute(args, context):
    if args.app_worker_command == "run-once":
        return run_worker_once(context.config_path, Path(args.worker_config), Path(args.mock_task) if args.mock_task else None)
    if args.app_worker_command == "run":
        return run_worker_loop(context.config_path, Path(args.worker_config), args.max_iterations)
    if args.app_worker_command == "analyze-stock":
        return run_manual_stock_analysis(context.config_path, Path(args.worker_config), args.symbol, args.trade_date, Path(args.output) if args.output else None, args.json_events)
    if args.app_worker_command == "daily-check":
        return run_daily_check(
            context.config_path,
            worker_config_path=Path(args.worker_config),
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            previous_candidate_pool_path=Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            previous_bundle_path=Path(args.previous_bundle) if args.previous_bundle else None,
            top=args.top,
            force=args.force,
            mock_upload=args.mock_upload,
            mock_upload_path=Path(args.mock_upload_path) if args.mock_upload_path else None,
            json_events=args.json_events,
            auto_pipeline=args.auto_pipeline,
        )
    if args.app_worker_command == "refresh-holding-prices":
        if args.loop:
            return run_holding_price_loop(
                Path(args.worker_config),
                args.trade_date,
                args.token_env,
                Path(args.mock_watchlist) if args.mock_watchlist else None,
                Path(args.mock_upload_path) if args.mock_upload_path else None,
                args.limit,
                args.max_iterations,
            )
        return run_holding_price_refresh(
            Path(args.worker_config),
            args.trade_date,
            args.token_env,
            Path(args.mock_watchlist) if args.mock_watchlist else None,
            Path(args.mock_upload_path) if args.mock_upload_path else None,
            args.limit,
            args.json_events,
        )
    return None
