"""Structured JSON log formatter for Cloud Run + Cloud Logging.

Cloud Logging auto-detects structured JSON on stdout when the payload contains
a ``severity`` field (maps to Cloud Logging's severity enum) and a ``message``
field.  The ``httpRequest`` field is auto-parsed by the log router.

Usage (uvicorn --log-config deploy/log-config.json):
  The formatter is referenced as ``fantasy_coach.logging_config.CloudRunJsonFormatter``
  in the log-config JSON file.  uvicorn passes access-log records to it so
  Cloud Logging receives structured httpRequest objects instead of raw text lines.

Reference: https://cloud.google.com/logging/docs/structured-logging
"""

from __future__ import annotations

import json
import logging
import time

_SEVERITY_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class CloudRunJsonFormatter(logging.Formatter):
    """Emit one JSON object per log record, compatible with Cloud Logging.

    Fields emitted:
      severity   — Cloud Logging severity string (DEBUG / INFO / WARNING / ERROR / CRITICAL)
      message    — formatted log message
      timestamp  — ISO 8601 UTC with fractional seconds
      logger     — logger name
      httpRequest — populated for uvicorn.access records (auto-parsed by Cloud Logging)
    """

    def format(self, record: logging.LogRecord) -> str:
        severity = _SEVERITY_MAP.get(record.levelno, "DEFAULT")
        payload: dict = {
            "severity": severity,
            "message": record.getMessage(),
            "logger": record.name,
            "timestamp": self._iso_timestamp(record.created),
        }

        # uvicorn access log records carry HTTP request metadata as attributes.
        # Map them to the Cloud Logging httpRequest field so the log router
        # can parse them without regex.
        if record.name == "uvicorn.access" and hasattr(record, "args") and record.args:
            payload["httpRequest"] = self._extract_http_request(record)

        # Pass through any extra fields set via logger.info(..., extra={...}).
        for key, val in record.__dict__.items():
            if key not in _SKIP_ATTRS and not key.startswith("_"):
                payload.setdefault(key, val)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)

    @staticmethod
    def _iso_timestamp(created: float) -> str:
        ts = time.gmtime(created)
        frac = created - int(created)
        return (
            f"{ts.tm_year:04d}-{ts.tm_mon:02d}-{ts.tm_mday:02d}T"
            f"{ts.tm_hour:02d}:{ts.tm_min:02d}:{ts.tm_sec:02d}"
            f".{int(frac * 1_000_000):06d}Z"
        )

    @staticmethod
    def _extract_http_request(record: logging.LogRecord) -> dict:
        # uvicorn.access args: (client_addr, method, path_with_query, http_version, status_code)
        args = record.args
        if not isinstance(args, tuple) or len(args) < 5:
            return {}
        _client, method, path, http_version, status = args
        return {
            "requestMethod": method,
            "requestUrl": path,
            "status": status,
            "protocol": f"HTTP/{http_version}",
        }


# Fields from LogRecord that are internal bookkeeping — don't re-emit.
_SKIP_ATTRS = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
