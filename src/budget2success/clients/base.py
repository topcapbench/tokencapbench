from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationRequest:
    model: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    system: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw_response: dict[str, Any] | None = None


class ModelClient(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a request."""
        ...
