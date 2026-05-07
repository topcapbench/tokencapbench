from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from budget2success._pydantic_compat import BaseModel, ConfigDict, Field, field_validator, model_validator


BENCHMARK_SLUG = "budget2success"
Track = Literal["math", "coding", "coding_edit", "code_editing", "swe", "agentic"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _non_empty_optional(value: str | None, field_name: str) -> str | None:
    if value is None:
        return value
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return value


class VerificationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    SKIPPED = "skipped"


class TaskRecord(BaseModel):
    """Common task representation used by TokenCapBench adapters.

    Official benchmark-specific information should be preserved in metadata or
    external_eval. This lets the local harness keep provenance while delegating
    paper-grade verification to official benchmark code when available.
    """

    model_config = ConfigDict(validate_assignment=True)

    task_id: str
    track: Track
    prompt: str
    verifier: str
    answer: str | None = None
    source: str = "local"
    source_version: str | None = None
    external_id: str | None = None
    budget_grid: list[int] | None = None
    fresh_split: str | None = None
    verifier_policy: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    external_eval: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "prompt", "verifier", "source")
    @classmethod
    def required_strings_non_empty(cls, value: str, info: Any) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("source_version", "external_id", "fresh_split", "verifier_policy")
    @classmethod
    def optional_strings_non_empty(cls, value: str | None, info: Any) -> str | None:
        return _non_empty_optional(value, info.field_name)

    @field_validator("budget_grid")
    @classmethod
    def budget_grid_positive(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("budget_grid must be non-empty when provided")
        if any(b <= 0 for b in value):
            raise ValueError("all budget values must be positive")
        return sorted(set(value))

    @model_validator(mode="after")
    def fill_provenance_defaults(self) -> "TaskRecord":
        if self.source_version is None:
            self.source_version = self.source
        if self.external_id is None:
            self.external_id = self.task_id
        if not self.external_eval:
            self.external_eval = {"harness": self.verifier, "source": self.source}
        return self


class ForecastRecord(BaseModel):
    """Pre-execution budget-success forecast.

    The public JSONL contract uses ``forecast`` for the success-probability map.
    Older internal artifacts used ``p_success_by_budget`` and
    ``median_budget2success``. The parser accepts those names while prompts use
    clearer public wording.
    """

    model_config = ConfigDict(validate_assignment=True)

    benchmark_slug: str = BENCHMARK_SLUG
    run_id: str | None = None
    suite: str | None = None
    task_id: str | None = None
    model: str | None = None
    scaffold: str | None = None
    solver_scaffold: str | None = None
    budget_grid: list[int] | None = None
    p_success_by_budget: dict[str, float] | None = None
    forecast: dict[str, float] | None = None
    forecast_prompt_hash: str | None = None
    predicted_unconstrained_output_tokens: float | None = None
    predicted_min_tokens_for_success: float | None = None
    median_budget2success: float | None = None
    p_failure_at_max_budget: float | None = None
    short_rationale: str = ""
    forecast_extras: dict[str, Any] = Field(default_factory=dict)
    raw_text: str | None = None
    repeat_index: int | None = None
    fresh_split: str | None = None
    created_at_utc: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("benchmark_slug")
    @classmethod
    def benchmark_slug_expected(cls, value: str) -> str:
        value = value.strip().lower()
        if value != BENCHMARK_SLUG:
            raise ValueError(f"benchmark_slug must be {BENCHMARK_SLUG!r}")
        return value

    @field_validator("run_id", "suite", "task_id", "model", "scaffold", "solver_scaffold", "forecast_prompt_hash", "fresh_split")
    @classmethod
    def optional_strings_non_empty(cls, value: str | None, info: Any) -> str | None:
        return _non_empty_optional(value, info.field_name)

    @field_validator("budget_grid")
    @classmethod
    def forecast_budget_grid_positive(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("budget_grid must be non-empty when provided")
        if any(b <= 0 for b in value):
            raise ValueError("all budget values must be positive")
        return sorted(set(value))

    @field_validator("p_success_by_budget", "forecast", mode="before")
    @classmethod
    def coerce_probability_map(cls, value: Any) -> dict[str, float] | None:
        if value is None:
            return value
        if not isinstance(value, dict):
            raise ValueError("forecast probability map must be a JSON object")
        normalized: dict[str, float] = {}
        for raw_budget, raw_probability in value.items():
            try:
                budget = int(str(raw_budget).strip())
            except ValueError as exc:
                raise ValueError(f"Budget key is not int-like: {raw_budget!r}") from exc
            if budget <= 0:
                raise ValueError(f"Budget key must be positive: {raw_budget!r}")
            budget_key = str(budget)
            if budget_key in normalized:
                raise ValueError(f"Duplicate budget key after normalization: {budget_key}")
            if isinstance(raw_probability, str):
                text = raw_probability.strip()
                if text.endswith("%"):
                    probability = float(text[:-1].strip()) / 100.0
                else:
                    probability = float(text)
            else:
                probability = float(raw_probability)
            normalized[budget_key] = probability
        return dict(sorted(normalized.items(), key=lambda item: int(item[0])))

    @field_validator("p_success_by_budget", "forecast")
    @classmethod
    def probabilities_in_range(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("forecast probability map must be non-empty")
        for budget, p in value.items():
            int(budget)  # raises if not an int-like budget key
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Probability for budget {budget} is out of range: {p}")
        return value

    @field_validator("median_budget2success")
    @classmethod
    def median_positive(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("median_budget2success must be positive when provided")
        return value

    @field_validator("predicted_unconstrained_output_tokens", "predicted_min_tokens_for_success")
    @classmethod
    def optional_token_forecast_positive(cls, value: float | None, info: Any) -> float | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive when provided")
        return value

    @field_validator("p_failure_at_max_budget")
    @classmethod
    def failure_probability_in_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p_failure_at_max_budget is out of range: {value}")
        return value

    @model_validator(mode="after")
    def synchronize_fields(self) -> "ForecastRecord":
        if self.p_success_by_budget is None and self.forecast is not None:
            self.p_success_by_budget = dict(self.forecast)
        if self.forecast is None and self.p_success_by_budget is not None:
            self.forecast = dict(self.p_success_by_budget)
        if self.p_success_by_budget is None:
            raise ValueError("ForecastRecord requires forecast or p_success_by_budget")
        if self.solver_scaffold is None and self.scaffold is not None:
            self.solver_scaffold = self.scaffold
        if self.scaffold is None and self.solver_scaffold is not None:
            self.scaffold = self.solver_scaffold
        if self.p_failure_at_max_budget is None:
            max_budget = max(int(b) for b in self.p_success_by_budget)
            self.p_failure_at_max_budget = 1.0 - self.p_success_by_budget[str(max_budget)]
        return self

    def probabilities_as_int_keys(self) -> dict[int, float]:
        if self.p_success_by_budget is None:
            return {}
        return {int(k): float(v) for k, v in self.p_success_by_budget.items()}


class VerificationResult(BaseModel):
    status: VerificationStatus
    success: bool
    details: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, **details: Any) -> "VerificationResult":
        metadata = details.pop("metadata", None) or {}
        return cls(status=VerificationStatus.SUCCESS, success=True, details=details, metadata=metadata)

    @classmethod
    def fail(cls, **details: Any) -> "VerificationResult":
        metadata = details.pop("metadata", None) or {}
        return cls(status=VerificationStatus.FAILURE, success=False, details=details, metadata=metadata)

    @classmethod
    def error(cls, **details: Any) -> "VerificationResult":
        metadata = details.pop("metadata", None) or {}
        return cls(status=VerificationStatus.ERROR, success=False, details=details, metadata=metadata)


class BudgetRunRecord(BaseModel):
    """One fresh solver call under one imposed generated-token budget."""

    model_config = ConfigDict(validate_assignment=True)

    benchmark_slug: str = BENCHMARK_SLUG
    run_id: str | None = None
    suite: str | None = None
    task_id: str
    model: str
    scaffold: str | None = None
    solver_scaffold: str | None = None
    budget: int
    solution: str
    success: bool
    verification: VerificationResult
    verifier: str | None = None
    verifier_version: str | None = None
    finish_reason: str | None = None
    cap_hit: bool | None = None
    truncated: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_visible_tokens: int | None = None
    reasoning_tokens: int | None = None
    wall_time_seconds: float | None = None
    generation_wall_time_seconds: float | None = None
    verification_wall_time_seconds: float | None = None
    end_to_end_wall_time_seconds: float | None = None
    generation_wall_time_s: float | None = None
    verification_wall_time_s: float | None = None
    end_to_end_wall_time_s: float | None = None
    retry_count: int = 0
    provider_request_id_hash: str | None = None
    repeat_index: int | None = None
    fresh_split: str | None = None
    verifier_policy: str | None = None
    created_at_utc: str = Field(default_factory=_utc_now_iso)
    raw_response: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("benchmark_slug")
    @classmethod
    def run_benchmark_slug_expected(cls, value: str) -> str:
        value = value.strip().lower()
        if value != BENCHMARK_SLUG:
            raise ValueError(f"benchmark_slug must be {BENCHMARK_SLUG!r}")
        return value

    @field_validator("task_id", "model")
    @classmethod
    def run_required_strings_non_empty(cls, value: str, info: Any) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("run_id", "suite", "scaffold", "solver_scaffold", "verifier", "verifier_version", "finish_reason", "provider_request_id_hash", "fresh_split", "verifier_policy")
    @classmethod
    def run_optional_strings_non_empty(cls, value: str | None, info: Any) -> str | None:
        return _non_empty_optional(value, info.field_name)

    @field_validator("budget")
    @classmethod
    def budget_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("budget must be positive")
        return value

    @field_validator("repeat_index", "retry_count")
    @classmethod
    def non_negative_counts(cls, value: int | None, info: Any) -> int | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens", "total_visible_tokens", "reasoning_tokens")
    @classmethod
    def token_counts_non_negative(cls, value: int | None, info: Any) -> int | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @field_validator(
        "wall_time_seconds",
        "generation_wall_time_seconds",
        "verification_wall_time_seconds",
        "end_to_end_wall_time_seconds",
        "generation_wall_time_s",
        "verification_wall_time_s",
        "end_to_end_wall_time_s",
    )
    @classmethod
    def wall_time_non_negative(cls, value: float | None, info: Any) -> float | None:
        if value is None:
            return value
        if value < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return value

    @model_validator(mode="after")
    def synchronize_resource_aliases(self) -> "BudgetRunRecord":
        if self.solver_scaffold is None and self.scaffold is not None:
            self.solver_scaffold = self.scaffold
        if self.scaffold is None and self.solver_scaffold is not None:
            self.scaffold = self.solver_scaffold
        if self.total_visible_tokens is None and self.total_tokens is not None:
            self.total_visible_tokens = self.total_tokens
        if self.total_tokens is None and self.total_visible_tokens is not None:
            self.total_tokens = self.total_visible_tokens
        if self.generation_wall_time_s is None and self.generation_wall_time_seconds is not None:
            self.generation_wall_time_s = self.generation_wall_time_seconds
        if self.generation_wall_time_seconds is None and self.generation_wall_time_s is not None:
            self.generation_wall_time_seconds = self.generation_wall_time_s
        if self.verification_wall_time_s is None and self.verification_wall_time_seconds is not None:
            self.verification_wall_time_s = self.verification_wall_time_seconds
        if self.verification_wall_time_seconds is None and self.verification_wall_time_s is not None:
            self.verification_wall_time_seconds = self.verification_wall_time_s
        if self.end_to_end_wall_time_s is None and self.end_to_end_wall_time_seconds is not None:
            self.end_to_end_wall_time_s = self.end_to_end_wall_time_seconds
        if self.end_to_end_wall_time_seconds is None and self.end_to_end_wall_time_s is not None:
            self.end_to_end_wall_time_seconds = self.end_to_end_wall_time_s
        if self.cap_hit is None and self.finish_reason:
            self.cap_hit = self.finish_reason.strip().lower() in {"length", "max_tokens", "max_output_tokens", "token_limit"}
        if self.cap_hit is None and self.completion_tokens is not None:
            self.cap_hit = int(self.completion_tokens) >= int(self.budget)
        if self.truncated is None and self.cap_hit is not None:
            self.truncated = bool(self.cap_hit)
        return self


class ForecastErrorRecord(BaseModel):
    benchmark_slug: str = BENCHMARK_SLUG
    task_id: str
    model: str
    error: str
    raw_text: str | None = None
    created_at_utc: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    run_id: str
    task_file: str
    output_dir: str = "reports/runs"
    provider: str = "mock"
    model: str = "mock-model"
    scaffold: str = "direct"
    forecast_prompt: str = "prompts/forecast_prompt.md"
    solver_prompts: dict[str, str] = Field(default_factory=dict)
    budget_grid: dict[str, list[int]] = Field(default_factory=dict)
    temperature: float = 0.0
    max_forecast_tokens: int = 1200
    limit: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_file", "output_dir", "provider", "model", "scaffold", "forecast_prompt")
    @classmethod
    def config_required_strings_non_empty(cls, value: str, info: Any) -> str:
        value = value.strip()
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("max_forecast_tokens")
    @classmethod
    def max_forecast_tokens_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_forecast_tokens must be positive")
        return value

    @field_validator("limit")
    @classmethod
    def limit_positive(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("limit must be positive when provided")
        return value
