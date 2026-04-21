"""Tests for GeminiClient — no live Vertex AI calls made."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from fantasy_coach.commentary.client import GeminiClient, GeminiResponse

PROJECT = "test-project"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-001"
ENDPOINT = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1"
    f"/projects/{PROJECT}/locations/{LOCATION}"
    f"/publishers/google/models/{MODEL}:generateContent"
)

_FAKE_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Panthers should win by 10."}],
                "role": "model",
            }
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 42,
        "candidatesTokenCount": 8,
        "totalTokenCount": 50,
    },
}


@pytest.fixture(autouse=True)
def patch_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any real credential fetch in every test."""
    monkeypatch.setattr(
        GeminiClient,
        "_auth_token",
        lambda self: "fake-bearer-token",
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real sleeps between retries."""
    import fantasy_coach.commentary.client as _mod

    monkeypatch.setattr(_mod.time, "sleep", lambda _: None)


def _make_client(**kwargs: object) -> GeminiClient:
    return GeminiClient(PROJECT, location=LOCATION, model=MODEL, **kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@respx.mock
def test_generate_returns_parsed_response() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    result = _make_client().generate("Who will win?")

    assert isinstance(result, GeminiResponse)
    assert result.text == "Panthers should win by 10."
    assert result.input_tokens == 42
    assert result.output_tokens == 8
    assert result.total_tokens == 50


@respx.mock
def test_generate_sends_correct_body() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    _make_client().generate("Analyse the match.", system="You are an NRL expert.")

    body = json.loads(route.calls.last.request.content)
    assert body["contents"][0]["parts"][0]["text"] == "Analyse the match."
    assert body["systemInstruction"]["parts"][0]["text"] == "You are an NRL expert."
    assert body["generationConfig"]["maxOutputTokens"] == 512


@respx.mock
def test_generate_omits_system_instruction_when_empty() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    _make_client().generate("Who wins?")

    body = json.loads(route.calls.last.request.content)
    assert "systemInstruction" not in body


@respx.mock
def test_generate_uses_bearer_token() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    _make_client().generate("Test.")

    assert route.calls.last.request.headers["Authorization"] == "Bearer fake-bearer-token"


@respx.mock
def test_generate_custom_max_output_tokens() -> None:
    route = respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    _make_client().generate("Test.", max_output_tokens=128)

    body = json.loads(route.calls.last.request.content)
    assert body["generationConfig"]["maxOutputTokens"] == 128


# ---------------------------------------------------------------------------
# Token usage logging
# ---------------------------------------------------------------------------


@respx.mock
def test_generate_logs_token_usage(caplog: pytest.LogCaptureFixture) -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_FAKE_RESPONSE))

    with caplog.at_level(logging.INFO, logger="fantasy_coach.commentary.client"):
        _make_client().generate("Test.")

    assert "gemini_tokens" in caplog.text
    assert "input=42" in caplog.text
    assert "output=8" in caplog.text
    assert "total=50" in caplog.text


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@respx.mock
def test_generate_retries_on_429_then_succeeds() -> None:
    call_count = 0

    def side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=_FAKE_RESPONSE)

    respx.post(ENDPOINT).mock(side_effect=side_effect)

    result = _make_client(max_retries=3).generate("Test.")

    assert result.text == "Panthers should win by 10."
    assert call_count == 2


@respx.mock
def test_generate_retries_on_503() -> None:
    call_count = 0

    def side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=_FAKE_RESPONSE)

    respx.post(ENDPOINT).mock(side_effect=side_effect)

    result = _make_client(max_retries=3).generate("Test.")
    assert result.text == "Panthers should win by 10."
    assert call_count == 3


@respx.mock
def test_generate_raises_after_all_retries_exhausted() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(503))

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        _make_client(max_retries=2).generate("Test.")


@respx.mock
def test_generate_raises_on_non_retryable_error() -> None:
    respx.post(ENDPOINT).mock(return_value=httpx.Response(400, json={"error": "bad request"}))

    with pytest.raises(httpx.HTTPStatusError):
        _make_client().generate("Test.")


# ---------------------------------------------------------------------------
# GeminiResponse helpers
# ---------------------------------------------------------------------------


def test_gemini_response_total_tokens() -> None:
    r = GeminiResponse(text="hi", input_tokens=10, output_tokens=5)
    assert r.total_tokens == 15
