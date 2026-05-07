from __future__ import annotations

import json
import os
from dataclasses import dataclass
import urllib.error
import urllib.request
from typing import Any

from budget2success.clients.base import GenerationRequest, GenerationResponse

DEFAULT_GATEWAY_BASE_URL = "http://localhost:8000/v1"
DEFAULT_CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"


@dataclass
class ChatGatewayClient:
    """Generic OpenAI-compatible chat-completions gateway client.

    Use this when experiments are routed through a non-OpenAI provider, an
    institutional gateway, or a local proxy that exposes `/chat/completions`.
    Provider-specific details should stay here rather than leaking into the
    benchmark protocol or paper.
    """

    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0
    organization: str | None = None
    endpoint: str = DEFAULT_CHAT_COMPLETIONS_ENDPOINT

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("BUDGET2SUCCESS_GATEWAY_API_KEY")
        self.base_url = (self.base_url or os.getenv("BUDGET2SUCCESS_GATEWAY_BASE_URL") or DEFAULT_GATEWAY_BASE_URL).rstrip("/")
        self.organization = self.organization or os.getenv("BUDGET2SUCCESS_GATEWAY_ORG")
        if not self.api_key:
            raise ValueError("Set BUDGET2SUCCESS_GATEWAY_API_KEY or pass api_key to ChatGatewayClient")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raw = self._call_chat_completion_api(request)
        return self._parse_response(raw, request)

    def _url(self) -> str:
        base_url = str(self.base_url or DEFAULT_GATEWAY_BASE_URL).rstrip("/")
        if base_url.endswith(DEFAULT_CHAT_COMPLETIONS_ENDPOINT):
            return base_url
        endpoint = self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
        return f"{base_url}{endpoint}"

    def _messages(self, request: GenerationRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "stream": False,
            "max_tokens": int(request.max_tokens),
        }
        top_p = request.metadata.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_http_request(self, request: GenerationRequest) -> urllib.request.Request:
        return urllib.request.Request(
            self._url(),
            data=json.dumps(self._payload(request)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

    def _call_chat_completion_api(self, request: GenerationRequest) -> dict[str, Any]:
        """Call the configured chat-completions endpoint."""
        http_request = self._build_http_request(request)
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat-completions gateway returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Chat-completions gateway request failed: {exc}") from exc

    def _parse_response(self, raw: dict[str, Any], request: GenerationRequest) -> GenerationResponse:
        text = ""
        finish_reason = None
        if "choices" in raw and raw["choices"]:
            choice = raw["choices"][0]
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                text = message.get("content", "") or ""
            elif isinstance(choice, dict):
                text = choice.get("text", "") or ""
            if isinstance(choice, dict):
                finish_reason = (
                    choice.get("finish_reason")
                    or choice.get("stop_reason")
                    or choice.get("finishReason")
                )
        elif "text" in raw:
            text = str(raw["text"])
        elif "content" in raw:
            text = str(raw["content"])

        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        return GenerationResponse(
            text=text,
            model=raw.get("model", request.model) if isinstance(raw, dict) else request.model,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            completion_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=completion_details.get("reasoning_tokens"),
            raw_response=raw,
        )
