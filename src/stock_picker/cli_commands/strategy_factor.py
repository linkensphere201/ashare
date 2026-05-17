from __future__ import annotations

from stock_picker.factor_exploration import backtest_candidate_002, compute_daily_factors, evaluate_factor_run, rank_candidate_002
from stock_picker.factor_research import research_candidate_001
from stock_picker.reports import show_report
from stock_picker.strategy import backtest_candidate_001, rank_candidate_001


def execute(args, context):
    if args.command == "strategy" and args.strategy_command == "rank-candidate-001":
        return rank_candidate_001(context.config_path, args.snapshot_id, args.top)
    if args.command == "strategy" and args.strategy_command == "backtest-candidate-001":
        return backtest_candidate_001(context.config_path, args.snapshot_id, args.holding_days, args.top, args.benchmark_symbol or "000852.SH")
    if args.command == "strategy" and args.strategy_command == "rank-candidate-002":
        return rank_candidate_002(context.config_path, args.factor_run_id, args.trade_date, args.top)
    if args.command == "strategy" and args.strategy_command == "backtest-candidate-002":
        return backtest_candidate_002(context.config_path, args.factor_run_id, args.top, args.rebalance, args.benchmark_symbol or "000852.SH")
    if args.command == "factor" and args.factor_command == "research-candidate-001":
        return research_candidate_001(context.config_path, args.snapshot_id, args.holding_days, args.top, args.report_id)
    if args.command == "factor" and args.factor_command == "compute-daily":
        return compute_daily_factors(context.config_path, args.snapshot_id, args.start_date, args.end_date, args.run_id)
    if args.command == "factor" and args.factor_command == "evaluate":
        return evaluate_factor_run(context.config_path, args.factor_run_id, args.forward_days, args.groups)
    if args.command == "reports" and args.reports_command == "show-report":
        return show_report(context.config_path, args.report_id, args.limit)
    return None
