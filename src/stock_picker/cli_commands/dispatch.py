from __future__ import annotations

import argparse
from collections.abc import Callable

from stock_picker.cli_commands import app_worker, analysis, provider, publish, storage, strategy_factor, workflow


def execute_command(args: argparse.Namespace, context, *, progress_logger_factory: Callable, json_event_printer: Callable):
    if args.command == "storage":
        return storage.execute(args, context)
    if args.command == "provider":
        return provider.execute(args, context, progress_logger_factory)
    if args.command in {"strategy", "factor", "reports"}:
        return strategy_factor.execute(args, context)
    if args.command == "publish":
        return publish.execute(args, context)
    if args.command == "analysis":
        return analysis.execute(args, context)
    if args.command == "workflow":
        return workflow.execute(args, context, json_event_printer)
    if args.command == "app-worker":
        return app_worker.execute(args, context)
    return None
