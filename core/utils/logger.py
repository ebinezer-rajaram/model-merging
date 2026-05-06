"""Logging helpers with optional colored output."""

import logging
import sys
from pathlib import Path
from typing import IO, Optional


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, *args, stream: Optional[IO[str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stream = stream

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno)
        stream = self.stream if self.stream is not None else sys.stderr
        if color and stream.isatty():
            return f"{color}{message}{self.RESET}"
        return message


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Create a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_ColorFormatter("%(message)s", stream=sys.stdout))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger
