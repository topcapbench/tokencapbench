from __future__ import annotations

import json
import re
from dataclasses import dataclass

from budget2success.clients.base import GenerationRequest, GenerationResponse
from budget2success.utils.token_counting import approximate_token_count


@dataclass
class MockClient:
    """Deterministic client for smoke tests.

    The mock returns a valid forecast for forecast prompts and simple final
    answers for toy math/code tasks. It is intentionally not a model simulator;
    it exists so CI and local smoke tests do not require API access.
    """

    forecast_probability: float = 0.75

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        text = self._generate_text(request)
        return GenerationResponse(
            text=text,
            model=request.model,
            finish_reason="stop",
            prompt_tokens=approximate_token_count(request.prompt),
            completion_tokens=approximate_token_count(text),
            total_tokens=approximate_token_count(request.prompt) + approximate_token_count(text),
            raw_response={"provider": "mock"},
        )

    def _generate_text(self, request: GenerationRequest) -> str:
        prompt = request.prompt.lower()
        if "p_success_by_budget" in prompt or "forecast" in prompt and "json" in prompt:
            budgets = _extract_budgets(request.prompt) or [256, 512, 1024]
            probs = {str(b): min(0.99, self.forecast_probability + i * 0.05) for i, b in enumerate(budgets)}
            return json.dumps(
                {
                    "p_success_by_budget": probs,
                    "median_budget2success": budgets[len(budgets) // 2],
                    "p_failure_at_max_budget": 1.0 - probs[str(max(budgets))],
                    "predicted_visible_tokens_unconstrained": budgets[len(budgets) // 2],
                    "predicted_output_tokens_unconstrained": budgets[len(budgets) // 2],
                    "predicted_total_visible_tokens": budgets[len(budgets) // 2],
                    "predicted_unconstrained_output_tokens": budgets[len(budgets) // 2],
                    "predicted_min_tokens_for_success": budgets[len(budgets) // 2],
                    "short_rationale": "Mock forecast for smoke testing only.",
                }
            )
        if '"files"' in prompt and "complete replacement content" in prompt and "add_one" in prompt:
            return json.dumps({"files": [{"path": "add_one.py", "content": "def add_one(x):\n    return x + 1\n"}]})
        if "class csvparser" in prompt and "header" in prompt:
            if request.max_tokens < 512:
                return "class CSVParser:\n    def __init__(self, csv: str):\n        self.csv = csv\n"
            return (
                "class CSVParser:\n"
                "    def __init__(self, csv: str):\n"
                "        self.csv = csv\n\n"
                "    def contents(self) -> list[list[str]]:\n"
                "        lines = self.csv.split(\"\\n\")\n"
                "        output = []\n"
                "        for line in lines:\n"
                "            output.append(line.split(\",\"))\n"
                "        return output\n\n"
                "    def header(self) -> list[str]:\n"
                "        lines = self.csv.split(\"\\n\")\n"
                "        return lines[0].strip().split(\",\")\n"
            )
        if "compute 2+2" in prompt or "2+2" in prompt:
            return "4"
        if "return x + 1" in prompt or "add_one" in prompt:
            return "def add_one(x):\n    return x + 1\n"
        return "ANSWER: mock"


def _extract_budgets(prompt: str) -> list[int]:
    match = re.search(r"Token budgets:\s*\[([^\]]+)\]", prompt)
    if not match:
        return []
    budgets: list[int] = []
    for piece in match.group(1).split(","):
        piece = piece.strip()
        if piece.isdigit():
            budgets.append(int(piece))
    return budgets
