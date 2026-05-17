from __future__ import annotations

from pathlib import Path

from stock_picker.publish import build_candidate_pool, build_daily_bundle, build_market_status


def execute(args, context):
    if args.publish_command == "build-market-status":
        return build_market_status(context.config_path, args.trade_date, Path(args.output) if args.output else None)
    if args.publish_command == "build-candidate-pool":
        return build_candidate_pool(
            context.config_path,
            args.factor_run_id,
            args.trade_date,
            Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            Path(args.previous_bundle) if args.previous_bundle else None,
            Path(args.output) if args.output else None,
            args.top,
        )
    if args.publish_command == "build-daily-bundle":
        return build_daily_bundle(
            config_path=context.config_path,
            factor_run_id=args.factor_run_id,
            trade_date=args.trade_date,
            previous_candidate_pool_path=Path(args.previous_candidate_pool) if args.previous_candidate_pool else None,
            previous_bundle_path=Path(args.previous_bundle) if args.previous_bundle else None,
            output_path=Path(args.output) if args.output else None,
            top=args.top,
        )
    return None
