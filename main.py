"""Unified CLI entrypoint for speech merging workflows."""

import argparse
import sys

from experiments.train_task import main as train_task_main


def parse_args() -> argparse.Namespace:
    """Parse high-level command."""
    parser = argparse.ArgumentParser(description="Speech merging pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run task training.")
    train_parser.add_argument("--task", default="asr", help="Task name to run.")
    train_parser.add_argument("--config", default=None, help="Config filename override.")

    return parser.parse_args()


def main() -> None:
    """Dispatch commands."""
    args = parse_args()
    if args.command == "train":
        argv = ["train_task.py"]
        if args.task:
            argv.extend(["--task", args.task])
        if args.config:
            argv.extend(["--config", args.config])
        sys.argv = argv
        train_task_main()
    else:
        raise NotImplementedError(f"Command '{args.command}' not supported.")


if __name__ == "__main__":
    main()
