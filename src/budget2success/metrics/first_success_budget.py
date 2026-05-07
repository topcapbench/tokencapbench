from __future__ import annotations

import math
from statistics import mean

from budget2success.execution.token_budget import observed_budget2success


def observed_first_success_budget(outcomes: dict[int, bool]) -> int | None:
    """Smallest budget with verified success, or None for right-censored tasks."""
    return observed_budget2success(outcomes)


def observed_censored_at_budget(outcomes: dict[int, bool]) -> int | None:
    """Largest observed budget when a task never succeeds."""
    if not outcomes or observed_budget2success(outcomes) is not None:
        return None
    return max(outcomes)


def log_token_error(predicted: float | None, observed: int | None) -> float | None:
    """Absolute log error for budget2success. Returns None for censored/missing values."""
    if predicted is None or observed is None or predicted <= 0 or observed <= 0:
        return None
    return abs(math.log(predicted) - math.log(observed))


def solved_only_log_token_error(predicted: float | None, observed: int | None) -> float | None:
    return log_token_error(predicted, observed)


def signed_log_budget_error(predicted: float | None, observed: float | int | None) -> float | None:
    """log(predicted) - log(observed). Negative means underbudget."""
    if predicted is None or observed is None or predicted <= 0 or observed <= 0:
        return None
    return math.log(float(predicted)) - math.log(float(observed))


def absolute_log_budget_error(predicted: float | None, observed: float | int | None) -> float | None:
    """Absolute signed log budget error."""
    value = signed_log_budget_error(predicted, observed)
    return abs(value) if value is not None else None


def censored_lower_bound_error(predicted: float | None, censored_at_budget: int | None) -> float | None:
    """Lower-bound log error for right-censored tasks.

    If a task is unsolved at the largest observed budget, its true
    budget2success is greater than that budget. A prediction below the
    censoring budget is therefore wrong by at least log(censored/predicted).
    Predictions at or above the censoring budget have no measurable lower-bound
    error from the observed data alone.
    """
    if predicted is None or censored_at_budget is None or predicted <= 0 or censored_at_budget <= 0:
        return None
    return max(0.0, math.log(censored_at_budget) - math.log(predicted))


def censoring_rate(outcomes_by_task: list[dict[int, bool]] | dict[str, dict[int, bool]]) -> float | None:
    values = list(outcomes_by_task.values()) if isinstance(outcomes_by_task, dict) else list(outcomes_by_task)
    if not values:
        return None
    return sum(1 for outcomes in values if observed_budget2success(outcomes) is None) / len(values)


def max_budget_failure_rate(outcomes_by_task: list[dict[int, bool]] | dict[str, dict[int, bool]]) -> float | None:
    values = list(outcomes_by_task.values()) if isinstance(outcomes_by_task, dict) else list(outcomes_by_task)
    values = [outcomes for outcomes in values if outcomes]
    if not values:
        return None
    failures = 0
    for outcomes in values:
        failures += 0 if outcomes[max(outcomes)] else 1
    return failures / len(values)


def underbudgeted(predicted: float | None, observed: int | None) -> bool | None:
    if predicted is None or observed is None:
        return None
    return predicted < observed


def overbudgeted(predicted: float | None, observed: int | None) -> bool | None:
    if predicted is None or observed is None:
        return None
    return predicted > observed


def underbudget_rate(predicted_by_task: dict[str, float | None], observed_by_task: dict[str, float | int | None]) -> float | None:
    """Fraction of solved tasks where predicted < observed."""
    flags = [
        float(predicted) < float(observed)
        for task_id, observed in observed_by_task.items()
        if observed is not None
        for predicted in [predicted_by_task.get(task_id)]
        if predicted is not None and float(predicted) > 0 and float(observed) > 0
    ]
    return sum(flags) / len(flags) if flags else None


def overbudget_rate(predicted_by_task: dict[str, float | None], observed_by_task: dict[str, float | int | None]) -> float | None:
    """Fraction of solved tasks where predicted > observed."""
    flags = [
        float(predicted) > float(observed)
        for task_id, observed in observed_by_task.items()
        if observed is not None
        for predicted in [predicted_by_task.get(task_id)]
        if predicted is not None and float(predicted) > 0 and float(observed) > 0
    ]
    return sum(flags) / len(flags) if flags else None


def underbudget_shortfall_factor(predicted: float | None, observed: float | int | None) -> float | None:
    """observed / predicted for underbudgeted solved tasks."""
    if predicted is None or observed is None or predicted <= 0 or observed <= 0 or predicted >= observed:
        return None
    return float(observed) / float(predicted)


def overbudget_waste_factor(predicted: float | None, observed: float | int | None) -> float | None:
    """predicted / observed for overbudgeted solved tasks."""
    if predicted is None or observed is None or predicted <= 0 or observed <= 0 or predicted <= observed:
        return None
    return float(predicted) / float(observed)


def overbudget_ratio(predicted: float | None, observed: int | None) -> float | None:
    """Backward-compatible alias for predicted/observed.

    Final paper outputs use signed log error, underbudget rate, overbudget rate,
    shortfall factor, and waste factor instead.
    """
    if predicted is None or observed is None or observed <= 0:
        return None
    return predicted / observed


def mean_defined(values: list[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(defined) if defined else None


def pairwise_ranking_accuracy(predicted: dict[str, float | int | None], observed: dict[str, int | None]) -> float | None:
    """Pairwise accuracy for ranking tasks by budget2success.

    Censored tasks with missing observed values are skipped.
    """
    comparable = [(task_id, predicted.get(task_id), observed.get(task_id)) for task_id in predicted]
    comparable = [(t, p, o) for t, p, o in comparable if p is not None and o is not None]
    n = 0
    correct = 0
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            _, p_i, o_i = comparable[i]
            _, p_j, o_j = comparable[j]
            if o_i == o_j:
                continue
            n += 1
            pred_order = p_i < p_j
            obs_order = o_i < o_j
            if pred_order == obs_order:
                correct += 1
    if n == 0:
        return None
    return correct / n


def forecast_monotonicity_violation_rate(curves_by_task: dict[str, dict[int, float]]) -> float | None:
    if not curves_by_task:
        return None
    violations = 0
    for curve in curves_by_task.values():
        probabilities = [float(probability) for _, probability in sorted(curve.items())]
        if any(later < earlier - 1e-9 for earlier, later in zip(probabilities, probabilities[1:])):
            violations += 1
    return violations / len(curves_by_task)


def outcome_nonmonotonicity_rate(outcomes_by_task: dict[str, dict[int, bool]]) -> float | None:
    if not outcomes_by_task:
        return None
    violations = 0
    for outcomes in outcomes_by_task.values():
        seen_success = False
        for _, success in sorted(outcomes.items()):
            if seen_success and not success:
                violations += 1
                break
            seen_success = seen_success or bool(success)
    return violations / len(outcomes_by_task)


def task_budget_ranking_accuracy(
    predicted: dict[str, float | int | None],
    outcomes_by_task: dict[str, dict[int, bool]],
) -> float | None:
    observed = {
        task_id: observed_first_success_budget(outcomes)
        for task_id, outcomes in outcomes_by_task.items()
    }
    comparable = [
        (float(predicted[task_id]), int(observed_budget))
        for task_id, observed_budget in observed.items()
        if task_id in predicted and predicted[task_id] is not None and observed_budget is not None
    ]
    n = 0
    score = 0.0
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            pred_i, obs_i = comparable[i]
            pred_j, obs_j = comparable[j]
            if obs_i == obs_j:
                continue
            n += 1
            if pred_i == pred_j:
                score += 0.5
            elif (pred_i < pred_j) == (obs_i < obs_j):
                score += 1.0
    return score / n if n else None


def truncation_rate(outcome_rows: list[dict[str, object]]) -> float | None:
    values: list[bool] = []
    for row in outcome_rows:
        if row.get("truncated") is not None:
            values.append(bool(row["truncated"]))
            continue
        completion = row.get("completion_tokens")
        budget = row.get("budget")
        if completion is not None and budget is not None:
            values.append(int(completion) >= int(budget))
    return sum(values) / len(values) if values else None
