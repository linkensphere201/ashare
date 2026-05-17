from __future__ import annotations

from stock_picker.workflow import pause_workflow, stock_analysis_workflow, sync_report_workflow, workflow_status


def execute(args, context, json_event_printer):
    if args.workflow_command == "sync-report":
        return sync_report_workflow(
            context.config_path,
            workflow_id=args.workflow_id,
            dry_run=args.dry_run or not args.confirm,
            confirmed=args.confirm,
            factor_run_id=args.factor_run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            top=args.top,
            emit=json_event_printer if args.json_events else None,
        )
    if args.workflow_command == "stock-analysis":
        return stock_analysis_workflow(context.config_path, args.symbol, args.factor_run_id, args.workflow_id, args.trade_date, json_event_printer if args.json_events else None)
    if args.workflow_command == "status":
        return workflow_status(context.config_path, args.workflow_id)
    if args.workflow_command == "pause":
        return pause_workflow(context.config_path, args.workflow_id)
    return None
