"""Tests for CloudRunJsonFormatter."""

from __future__ import annotations

import json
import logging

from fantasy_coach.logging_config import CloudRunJsonFormatter


def _make_record(
    msg: str,
    level: int = logging.INFO,
    logger_name: str = "test",
    exc_info: bool = False,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=logger_name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=(ValueError, ValueError("oops"), None) if exc_info else None,
    )
    return record


def test_output_is_valid_json() -> None:
    fmt = CloudRunJsonFormatter()
    record = _make_record("hello world")
    raw = fmt.format(record)
    parsed = json.loads(raw)
    assert parsed["message"] == "hello world"


def test_severity_mapping() -> None:
    fmt = CloudRunJsonFormatter()
    cases = [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRITICAL"),
    ]
    for level, expected in cases:
        record = _make_record("x", level=level)
        parsed = json.loads(fmt.format(record))
        assert parsed["severity"] == expected


def test_timestamp_format() -> None:
    fmt = CloudRunJsonFormatter()
    record = _make_record("ts test")
    parsed = json.loads(fmt.format(record))
    ts = parsed["timestamp"]
    assert ts.endswith("Z")
    assert "T" in ts


def test_logger_name_included() -> None:
    fmt = CloudRunJsonFormatter()
    record = _make_record("hi", logger_name="fantasy_coach.app")
    parsed = json.loads(fmt.format(record))
    assert parsed["logger"] == "fantasy_coach.app"


def test_exception_included() -> None:
    fmt = CloudRunJsonFormatter()
    record = _make_record("boom", exc_info=True)
    parsed = json.loads(fmt.format(record))
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_http_request_extracted_for_uvicorn_access() -> None:
    fmt = CloudRunJsonFormatter()
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname="access.py",
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1234", "GET", "/predictions?season=2026", "1.1", 200),
        exc_info=None,
    )
    parsed = json.loads(fmt.format(record))
    assert "httpRequest" in parsed
    http = parsed["httpRequest"]
    assert http["requestMethod"] == "GET"
    assert http["requestUrl"] == "/predictions?season=2026"
    assert http["status"] == 200
    assert http["protocol"] == "HTTP/1.1"


def test_non_access_record_has_no_http_request() -> None:
    fmt = CloudRunJsonFormatter()
    record = _make_record("app log", logger_name="fantasy_coach.predictions")
    parsed = json.loads(fmt.format(record))
    assert "httpRequest" not in parsed
