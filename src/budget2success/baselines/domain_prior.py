from __future__ import annotations

DEFAULT_DOMAIN_PRIORS = {
    "math": 0.55,
    "coding": 0.35,
    "swe": 0.10,
    "agentic": 0.30,
}


def domain_curve(track: str, budget_grid: list[int], priors: dict[str, float] | None = None) -> dict[str, float]:
    priors = priors or DEFAULT_DOMAIN_PRIORS
    base = priors.get(track, 0.5)
    # Simple monotone baseline: larger budgets get a small bump.
    n = max(1, len(budget_grid) - 1)
    return {str(b): min(0.99, base + 0.05 * i / n) for i, b in enumerate(sorted(budget_grid))}
