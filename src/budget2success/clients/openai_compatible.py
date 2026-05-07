from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from budget2success.clients.base import GenerationRequest, GenerationResponse


@dataclass
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat completions client."""

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.api_key = self.api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY") or "EMPTY"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raw = self._post_chat_completion(request)
        return self._parse_response(raw, request)

    def _post_chat_completion(self, request: GenerationRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "max_tokens": int(request.max_tokens or self.max_tokens or 1024),
            "temperature": float(request.temperature if request.temperature is not None else (self.temperature or 0.0)),
        }
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible server returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible server request failed: {exc}") from exc

    def _parse_response(self, raw: dict[str, Any], request: GenerationRequest) -> GenerationResponse:
        text = ""
        finish_reason = None
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    text = "" if content is None else str(content)
                elif choice.get("text") is not None:
                    text = str(choice.get("text"))
                finish_reason = choice.get("finish_reason") or choice.get("stop_reason")

        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        return GenerationResponse(
            text=text,
            model=str(raw.get("model") or request.model or self.model or ""),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            completion_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            raw_response=raw,
        )
