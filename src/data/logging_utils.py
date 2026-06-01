"""Shared telemetry helpers for the data pipeline.

CLAUDE.md (Logging Protocols) requires explicit telemetry across the data
pipeline -- vertex-degradation scales, NaN monitoring, etc. Centralising the
logger config here keeps that policy consistent across modules.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler attached once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
