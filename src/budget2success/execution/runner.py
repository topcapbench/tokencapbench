from __future__ import annotations

import time
from dataclasses import dataclass

from budget2success.clients.base import GenerationRequest, ModelClient
from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import BudgetRunRecord, TaskRecord


CAP_HIT_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}


@dataclass
class BudgetRunResult:
    task_id: str
    model: str
    budget: int
    solution: str
    verification: object
    completion_tokens: int | None
    prompt_tokens: int | None
    total_tokens: int | None


def run_task_under_budget(
    client: ModelClient,
    verifier: Verifier,
    task: TaskRecord,
    model: str,
    prompt: str,
    budget: int,
    temperature: float = 0.0,
    scaffold: str | None = None,
) -> BudgetRunRecord:
    started = time.perf_counter()
    generation_started = time.perf_counter()
    response = client.generate(
        GenerationRequest(model=model, prompt=prompt, max_tokens=budget, temperature=temperature)
    )
    generation_elapsed = time.perf_counter() - generation_started
    verification_started = time.perf_counter()
    verification = verifier.verify(task, response.text)
    verification_elapsed = time.perf_counter() - verification_started
    end_to_end_elapsed = time.perf_counter() - started
    metadata = {
        "track": task.track,
        "source": task.source,
        "source_version": task.source_version,
        "external_id": task.external_id,
    }
    for key in ("label_source", "exclude_from_main_metrics", "official_harness_required"):
        if key in verification.metadata:
            metadata[key] = verification.metadata[key]
    return BudgetRunRecord(
        task_id=task.task_id,
        model=model,
        scaffold=scaffold,
        budget=budget,
        solution=response.text,
        verification=verification,
        success=verification.success,
        finish_reason=response.finish_reason,
        cap_hit=_is_cap_hit(response.finish_reason, response.completion_tokens, budget),
        truncated=_is_truncated(response.finish_reason, response.completion_tokens, budget),
        completion_tokens=response.completion_tokens,
        prompt_tokens=response.prompt_tokens,
        total_tokens=response.total_tokens,
        total_visible_tokens=response.total_tokens,
        reasoning_tokens=response.reasoning_tokens,
        wall_time_seconds=end_to_end_elapsed,
        generation_wall_time_seconds=generation_elapsed,
        verification_wall_time_seconds=verification_elapsed,
        end_to_end_wall_time_seconds=end_to_end_elapsed,
        raw_response=response.raw_response,
        metadata=metadata,
    )


def _is_cap_hit(finish_reason: str | None, completion_tokens: int | None, budget: int) -> bool:
    if finish_reason and finish_reason.strip().lower() in CAP_HIT_FINISH_REASONS:
        return True
    if completion_tokens is None:
        return False
    return int(completion_tokens) >= int(budget)


def _is_truncated(finish_reason: str | None, completion_tokens: int | None, budget: int) -> bool:
    # In this first-release harness, truncation evidence is conservative: a
    # provider length stop or exact budget fill is treated as cut-off evidence.
    # Downstream verifiers may add stronger parse-level truncation evidence.
    return _is_cap_hit(finish_reason, completion_tokens, budget)
