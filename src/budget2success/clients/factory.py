from __future__ import annotations

from typing import Any

from .base import ModelClient
from .mock import MockClient
from .openai_compatible import OpenAICompatibleClient
from .chat_gateway import ChatGatewayClient


def build_client(config: dict[str, Any]) -> ModelClient:
    provider = config.get("provider", "mock")
    if provider == "mock":
        return MockClient()
    if provider in {"provider", "gateway", "chat_gateway", "provider_gateway", "openai_gateway"}:
        return ChatGatewayClient(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            endpoint=config.get("endpoint", "/chat/completions"),
            timeout_seconds=float(config.get("timeout_seconds") or config.get("timeout") or 120.0),
        )
    if provider in {"openai_compatible", "openai-compatible", "vllm", "local_vllm"}:
        return OpenAICompatibleClient(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=config.get("model"),
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature"),
            timeout_seconds=float(config.get("timeout_seconds") or config.get("timeout") or 120.0),
        )
    raise ValueError(f"Unknown provider: {provider}")
