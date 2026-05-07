from __future__ import annotations

from budget2success.utils.token_counting import approximate_token_count


def prompt_length_curve(prompt: str, budget_grid: list[int]) -> dict[str, float]:
    """Toy prompt-length baseline.

    This is intentionally simple. Paper results should fit this baseline on a
    development split and evaluate on heldout tasks.
    """
    prompt_tokens = approximate_token_count(prompt)
    center = max(256, prompt_tokens * 2)
    curve: dict[str, float] = {}
    for b in sorted(budget_grid):
        curve[str(b)] = max(0.01, min(0.99, b / (b + center)))
    return curve
