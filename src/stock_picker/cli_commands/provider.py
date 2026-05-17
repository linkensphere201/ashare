from __future__ import annotations

from stock_picker.provider import fetch_cyq_perf_batch, fetch_provider_raw, probe_provider_api, run_cyq_perf_batches, run_market_daily
from stock_picker.sync import sync_latest


def execute(args, context, progress_logger_factory):
    if args.provider_command == "fetch":
        return fetch_provider_raw(
            config_path=context.config_path,
            source=args.source,
            dataset=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            ts_code=args.ts_code,
            trade_date=args.trade_date,
            token_env=args.token_env,
        )
    if args.provider_command == "fetch-cyq-perf-batch":
        return fetch_cyq_perf_batch(
            config_path=context.config_path,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            symbols=args.symbol,
            limit=args.limit,
            offset=args.offset,
            requests_per_minute=args.requests_per_minute,
            token_env=args.token_env,
        )
    if args.provider_command == "run-cyq-perf-batches":
        logger = progress_logger_factory()
        return run_cyq_perf_batches(
            config_path=context.config_path,
            source=args.source,
            run_id=args.run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            requests_per_minute=args.requests_per_minute,
            token_env=args.token_env,
            progress_every_batches=args.progress_every_batches,
            progress_callback=logger.info,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
        )
    if args.provider_command == "run-market-daily":
        logger = progress_logger_factory()
        return run_market_daily(
            config_path=context.config_path,
            source=args.source,
            run_id=args.run_id,
            datasets=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            max_tasks=args.max_tasks,
            requests_per_minute=args.requests_per_minute,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
            symbol_batch_size=args.symbol_batch_size,
            token_env=args.token_env,
            progress_every_tasks=args.progress_every_tasks,
            progress_callback=logger.info,
        )
    if args.provider_command == "probe":
        return probe_provider_api(args.source, args.api, args.ts_code, args.trade_date, args.start_date, args.end_date, args.token_env)
    if args.provider_command == "sync-latest":
        logger = progress_logger_factory()
        return sync_latest(
            config_path=context.config_path,
            source=args.source,
            datasets=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            calendar_lookback_days=args.calendar_lookback_days,
            dry_run=args.dry_run,
            run_id_prefix=args.run_id_prefix,
            max_market_tasks=args.max_market_tasks,
            requests_per_minute=args.requests_per_minute,
            symbol_batch_size=args.symbol_batch_size,
            max_cyq_batches=args.max_cyq_batches,
            cyq_batch_size=args.cyq_batch_size,
            cyq_requests_per_minute=args.cyq_requests_per_minute,
            retry=args.retry,
            retry_wait_seconds=args.retry_wait_seconds,
            backoff_multiplier=args.backoff_multiplier,
            create_snapshot_after=args.create_snapshot,
            token_env=args.token_env,
            benchmark_symbols=args.benchmark_symbol,
            progress_every_tasks=args.progress_every_tasks,
            progress_every_batches=args.progress_every_batches,
            progress_callback=logger.info,
        )
    return None
