"""Centralized logging configuration.

Called once by each CLI entry point. Replaces the pattern where
multiple modules call basicConfig() independently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure logging for the pipeline.

    Sets up a stream handler (stdout) and optionally a file handler.
    Should be called once at CLI entry point startup.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers (prevents duplicate output on re-init)
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
