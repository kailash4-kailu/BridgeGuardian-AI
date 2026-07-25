"""
BridgeGuardian AI — Structured JSON Logging Configuration
Configures structured JSON logging containing ISO timestamps, log levels, request trace IDs, and context parameters.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Formats Python log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if passed via extra={}
        if hasattr(record, "trace_id"):
            log_object["trace_id"] = getattr(record, "trace_id")
        if hasattr(record, "user_id"):
            log_object["user_id"] = getattr(record, "user_id")

        return json.dumps(log_object)


def setup_json_logging(log_level: int = logging.INFO) -> None:
    """Configures root logger to output structured JSON to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(handler)
