"""Vertex AI Gemini Flash client.

Model pin: gemini-2.0-flash-001 (latest stable as of 2026-04; bump when Vertex retires it).
Auth:      Application Default Credentials / Workload Identity — no API keys stored here.
Retries:   exponential back-off on 429 / 5xx.
Tokens:    usage logged per call at INFO level for cost visibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Pin to a specific stable revision; update when the model is retired.
DEFAULT_MODEL = "gemini-2.0-flash-001"
DEFAULT_LOCATION = "us-central1"

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class GeminiResponse:
    """Parsed response from a single generateContent call."""

    text: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GeminiClient:
    """Thin wrapper around the Vertex AI generateContent REST endpoint.

    Uses Application Default Credentials (service account / Workload Identity)
    — never API keys.  Retries on 429 / 5xx with exponential back-off.

    Usage::

        client = GeminiClient(project="my-gcp-project")
        response = client.generate(
            "Which team will win Saturday's game?",
            system="You are a concise NRL analyst.",
        )
        print(response.text)
    """

    def __init__(
        self,
        project: str,
        *,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self._project = project
        self._location = location
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout
        self._endpoint = (
            f"https://{location}-aiplatform.googleapis.com/v1"
            f"/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent"
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_token(self) -> str:
        """Return a valid OAuth2 bearer token from Application Default Credentials."""
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default(scopes=_SCOPES)
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        max_output_tokens: int = 512,
    ) -> GeminiResponse:
        """Call generateContent and return the parsed response.

        Retries up to ``max_retries`` times on transient HTTP errors (429 / 5xx).
        Raises ``RuntimeError`` when all retries are exhausted.
        """
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        token = self._auth_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            if attempt:
                time.sleep(2**attempt)
            try:
                resp = httpx.post(
                    self._endpoint,
                    json=body,
                    headers=headers,
                    timeout=self._timeout,
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    logger.warning(
                        "gemini_retry attempt=%d status=%d", attempt + 1, resp.status_code
                    )
                    continue
                resp.raise_for_status()
                return self._parse(resp.json())
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning("gemini_timeout attempt=%d", attempt + 1)

        raise RuntimeError(
            f"Gemini generateContent failed after {self._max_retries} attempts"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(data: dict) -> GeminiResponse:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        input_tokens = int(usage.get("promptTokenCount", 0))
        output_tokens = int(usage.get("candidatesTokenCount", 0))
        logger.info(
            "gemini_tokens input=%d output=%d total=%d",
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
        )
        return GeminiResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
