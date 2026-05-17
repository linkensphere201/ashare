from __future__ import annotations

from pathlib import Path

from stock_picker.analysis import analyze_stock


def execute(args, context):
    if args.analysis_command == "stock":
        return analyze_stock(context.config_path, args.symbol, args.factor_run_id, args.trade_date, Path(args.output) if args.output else None)
    return None
