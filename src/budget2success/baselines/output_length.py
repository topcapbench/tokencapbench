from __future__ import annotations


def output_length_baseline_curve(predicted_output_tokens: int, budget_grid: list[int]) -> dict[str, float]:
    """Convert an output-length prediction into a crude success curve baseline."""
    return {str(b): 0.25 if b < predicted_output_tokens else 0.75 for b in sorted(budget_grid)}
