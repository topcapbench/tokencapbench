from __future__ import annotations

from collections import defaultdict
from typing import Any


ProbabilityCurve = dict[int, float]


def fit_constant_by_budget(calibration_outcomes: list[Any]) -> dict[int, float]:
    """Return empirical success probability for each budget on calibration tasks."""
    grouped: dict[int, list[bool]] = defaultdict(list)
    for row in calibration_outcomes:
        budget = _budget(row)
        if budget is not None:
            grouped[budget].append(_success(row))
    return {budget: _clamp(_mean(values)) for budget, values in sorted(grouped.items())}


def predict_constant_by_budget(
    task_ids: list[str],
    budget_grid: list[int],
    fitted: dict[int, float],
) -> dict[str, ProbabilityCurve]:
    """Return same fitted curve for every task."""
    return {task_id: _curve_from_budget_fit(budget_grid, fitted) for task_id in task_ids}


def fit_source_by_budget(calibration_outcomes: list[Any], task_metadata: dict[str, Any]) -> dict[str, dict[int, float]]:
    """Return empirical success probability by source and budget."""
    global_fit = fit_constant_by_budget(calibration_outcomes)
    grouped: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in calibration_outcomes:
        task_id = _task_id(row)
        budget = _budget(row)
        if task_id is None or budget is None:
            continue
        source = _source(task_metadata.get(task_id), row)
        grouped[source][budget].append(_success(row))
    result: dict[str, dict[int, float]] = {"__global__": global_fit}
    for source, by_budget in sorted(grouped.items()):
        curve = dict(global_fit)
        for budget, values in by_budget.items():
            curve[int(budget)] = _clamp(_mean(values))
        result[source] = curve
    return result


def predict_source_by_budget(
    task_ids: list[str],
    task_metadata: dict[str, Any],
    budget_grid_by_task: dict[str, list[int]],
    fitted: dict[str, dict[int, float]],
) -> dict[str, ProbabilityCurve]:
    """Apply source-budget priors, falling back to the global prior."""
    global_fit = fitted.get("__global__", {})
    curves: dict[str, ProbabilityCurve] = {}
    for task_id in task_ids:
        source = _source(task_metadata.get(task_id), None)
        source_fit = fitted.get(source) or global_fit
        curves[task_id] = _curve_from_budget_fit(budget_grid_by_task.get(task_id, []), source_fit or global_fit)
    return curves


def fit_prompt_length_bins(
    calibration_outcomes: list[Any],
    task_metadata: dict[str, Any],
    n_bins: int = 5,
) -> dict[str, Any]:
    """Fit a prompt-length bin prior by budget."""
    task_lengths = {
        task_id: _prompt_length(meta)
        for task_id, meta in task_metadata.items()
        if task_id in {_task_id(row) for row in calibration_outcomes}
    }
    lengths = sorted(task_lengths.values())
    global_fit = fit_constant_by_budget(calibration_outcomes)
    if not lengths:
        return {"n_bins": n_bins, "boundaries": [], "global": global_fit, "bins": {}}
    n_bins = max(1, min(int(n_bins), len(set(lengths)) or 1))
    boundaries = [_quantile(lengths, index / n_bins) for index in range(1, n_bins)]
    grouped: dict[int, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in calibration_outcomes:
        task_id = _task_id(row)
        budget = _budget(row)
        if task_id is None or budget is None:
            continue
        bin_index = _bin_index(task_lengths.get(task_id, 0), boundaries)
        grouped[bin_index][budget].append(_success(row))
    bins: dict[int, dict[int, float]] = {}
    for bin_index, by_budget in grouped.items():
        curve = dict(global_fit)
        for budget, values in by_budget.items():
            curve[int(budget)] = _clamp(_mean(values))
        bins[int(bin_index)] = curve
    return {"n_bins": n_bins, "boundaries": boundaries, "global": global_fit, "bins": bins}


def predict_prompt_length_bins(
    task_ids: list[str],
    task_metadata: dict[str, Any],
    budget_grid: list[int] | dict[str, list[int]],
    fitted: dict[str, Any],
) -> dict[str, ProbabilityCurve]:
    """Apply fitted prompt-length bin prior."""
    curves: dict[str, ProbabilityCurve] = {}
    boundaries = [float(value) for value in fitted.get("boundaries", [])]
    bins = {int(key): value for key, value in (fitted.get("bins") or {}).items()}
    global_fit = {int(key): float(value) for key, value in (fitted.get("global") or {}).items()}
    for task_id in task_ids:
        grid = budget_grid.get(task_id, []) if isinstance(budget_grid, dict) else budget_grid
        bin_index = _bin_index(_prompt_length(task_metadata.get(task_id)), boundaries)
        fit = bins.get(bin_index) or global_fit
        curves[task_id] = _curve_from_budget_fit([int(value) for value in grid], fit)
    return curves


def fit_histogram_recalibrator(
    calibration_forecasts: list[Any],
    calibration_outcomes: list[Any],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit a bin-based calibrator from raw forecast probabilities to empirical success."""
    labels = {(_task_id(row), _budget(row)): _success(row) for row in calibration_outcomes}
    examples: list[tuple[float, bool]] = []
    for row in calibration_forecasts:
        task_id = _task_id(row)
        for budget, probability in _probabilities(row).items():
            key = (task_id, int(budget))
            if key in labels:
                examples.append((float(probability), bool(labels[key])))
    global_probability = _clamp(_mean([label for _, label in examples])) if examples else 0.5
    n_bins = max(1, int(n_bins))
    bin_probs: dict[int, float] = {}
    grouped: dict[int, list[bool]] = defaultdict(list)
    for probability, success in examples:
        grouped[_probability_bin(probability, n_bins)].append(success)
    for bin_index in range(n_bins):
        values = grouped.get(bin_index, [])
        bin_probs[bin_index] = _clamp(_mean(values)) if values else global_probability
    return {
        "n_bins": n_bins,
        "global_probability": global_probability,
        "bin_probabilities": bin_probs,
    }


def apply_histogram_recalibrator(forecasts: list[Any], calibrator: dict[str, Any]) -> dict[str, ProbabilityCurve]:
    """Return recalibrated probability curves."""
    n_bins = int(calibrator.get("n_bins") or 10)
    global_probability = float(calibrator.get("global_probability", 0.5))
    raw_bins = calibrator.get("bin_probabilities") or {}
    bin_probabilities = {int(key): float(value) for key, value in raw_bins.items()}
    curves: dict[str, ProbabilityCurve] = {}
    for row in forecasts:
        task_id = _task_id(row)
        if task_id is None:
            continue
        curve: ProbabilityCurve = {}
        for budget, probability in _probabilities(row).items():
            curve[int(budget)] = _clamp(bin_probabilities.get(_probability_bin(probability, n_bins), global_probability))
        curves[task_id] = curve
    return curves


def _curve_from_budget_fit(budget_grid: list[int], fitted: dict[int, float]) -> ProbabilityCurve:
    if not budget_grid:
        return {}
    if not fitted:
        return {int(budget): 0.5 for budget in budget_grid}
    budgets = sorted(int(value) for value in fitted)
    curve: ProbabilityCurve = {}
    for budget in sorted(int(value) for value in budget_grid):
        if budget in fitted:
            curve[budget] = _clamp(fitted[budget])
            continue
        nearest = min(budgets, key=lambda candidate: abs(candidate - budget))
        curve[budget] = _clamp(fitted[nearest])
    return curve


def _task_id(row: Any) -> str | None:
    if row is None:
        return None
    if isinstance(row, dict):
        value = row.get("task_id")
    else:
        value = getattr(row, "task_id", None)
    return str(value) if value is not None else None


def _budget(row: Any) -> int | None:
    if row is None:
        return None
    value = row.get("budget") if isinstance(row, dict) else getattr(row, "budget", None)
    return int(value) if value is not None else None


def _success(row: Any) -> bool:
    value = row.get("success") if isinstance(row, dict) else getattr(row, "success", False)
    return bool(value)


def _probabilities(row: Any) -> dict[int, float]:
    value = row.get("p_success_by_budget") if isinstance(row, dict) else getattr(row, "p_success_by_budget", {})
    return {int(budget): float(probability) for budget, probability in (value or {}).items()}


def _source(meta: Any, row: Any | None) -> str:
    if meta is not None:
        if isinstance(meta, dict):
            value = meta.get("source")
        else:
            value = getattr(meta, "source", None)
        if value:
            return str(value)
    if isinstance(row, dict):
        row_meta = row.get("metadata") or {}
        return str(row_meta.get("source") or row.get("source") or "unknown")
    return "unknown"


def _prompt_length(meta: Any) -> int:
    if meta is None:
        return 0
    if isinstance(meta, dict):
        prompt = meta.get("prompt") or ""
        if "prompt_length" in meta:
            return int(meta["prompt_length"])
    else:
        prompt = getattr(meta, "prompt", "") or ""
    return len(str(prompt))


def _bin_index(value: int | float | None, boundaries: list[float]) -> int:
    number = float(value or 0)
    for index, boundary in enumerate(boundaries):
        if number <= boundary:
            return index
    return len(boundaries)


def _probability_bin(probability: float, n_bins: int) -> int:
    clipped = max(0.0, min(1.0, float(probability)))
    return min(n_bins - 1, int(clipped * n_bins))


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def _mean(values: list[bool]) -> float:
    return sum(1.0 for value in values if value) / len(values) if values else 0.5


def _clamp(value: float) -> float:
    return max(1e-4, min(1.0 - 1e-4, float(value)))
