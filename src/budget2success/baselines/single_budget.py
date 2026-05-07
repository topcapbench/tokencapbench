from __future__ import annotations


def single_budget_curve(predicted_budget: int, budget_grid: list[int]) -> dict[str, float]:
    return {str(b): 0.2 if b < predicted_budget else 0.8 for b in sorted(budget_grid)}
