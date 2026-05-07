from __future__ import annotations


def constant_curve(budget_grid: list[int], probability: float = 0.5) -> dict[str, float]:
    return {str(b): float(probability) for b in budget_grid}
