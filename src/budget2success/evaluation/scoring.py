from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.metrics.regret import oracle_utility, selected_budget_from_forecast, utility
from budget2success.metrics.first_success_budget import (
    censored_lower_bound_error,
    max_budget_failure_rate,
    observed_censored_at_budget,
    log_token_error,
    overbudget_ratio,
    underbudgeted,
)


@dataclass(frozen=True)
class ScoreSummary:
    n: int
    brier: float
    ece: float
    solved_only_log_ttg_error: float | None
    censored_lower_bound_error: float | None
    censoring_rate: float | None
    max_budget_failure_rate: float | None
    underbudget_rate: float | None
    overbudget_ratio: float | None
    mean_regret: float | None

    @property
    def median_log_token_error(self) -> float | None:
        return self.solved_only_log_ttg_error

    @property
    def median_overbudget_ratio(self) -> float | None:
        return self.overbudget_ratio


def score_forecasts(
    forecast_records: list[dict[str, Any]],
    outcome_records: list[dict[str, Any]],
    token_cost: float = 0.0,
) -> ScoreSummary:
    outcomes_by_task: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in outcome_records:
        outcomes_by_task[str(row["task_id"])][int(row["budget"])] = bool(row["success"])

    probs: list[float] = []
    ys: list[bool] = []
    log_errors: list[float] = []
    lower_bound_errors: list[float] = []
    under_flags: list[bool] = []
    over_ratios: list[float] = []
    regrets: list[float] = []

    for forecast in forecast_records:
        task_id = str(forecast["task_id"])
        p_by_budget = {int(k): float(v) for k, v in forecast["p_success_by_budget"].items()}
        task_outcomes = outcomes_by_task.get(task_id, {})
        for budget, p in p_by_budget.items():
            if budget in task_outcomes:
                probs.append(p)
                ys.append(task_outcomes[budget])
        observed = _observed_budget2success(task_outcomes)
        censored_at = observed_censored_at_budget(task_outcomes)
        predicted = forecast.get("median_budget2success")
        err = log_token_error(float(predicted) if predicted is not None else None, observed)
        if err is not None:
            log_errors.append(err)
        lb_err = censored_lower_bound_error(float(predicted) if predicted is not None else None, censored_at)
        if lb_err is not None:
            lower_bound_errors.append(lb_err)
        flag = underbudgeted(float(predicted) if predicted is not None else None, observed)
        if flag is not None:
            under_flags.append(flag)
        ratio = overbudget_ratio(float(predicted) if predicted is not None else None, observed)
        if ratio is not None:
            over_ratios.append(ratio)
        if task_outcomes:
            selected = selected_budget_from_forecast(p_by_budget, reward=1.0, token_cost=token_cost)
            actual = utility(task_outcomes.get(selected, False), selected, reward=1.0, token_cost=token_cost)
            regrets.append(oracle_utility(task_outcomes, reward=1.0, token_cost=token_cost) - actual)

    censored_count = sum(1 for task_outcomes in outcomes_by_task.values() if _observed_budget2success(task_outcomes) is None)
    return ScoreSummary(
        n=len(forecast_records),
        brier=brier_score(probs, ys),
        ece=expected_calibration_error(probs, ys),
        solved_only_log_ttg_error=float(np.median(log_errors)) if log_errors else None,
        censored_lower_bound_error=float(np.mean(lower_bound_errors)) if lower_bound_errors else None,
        censoring_rate=censored_count / len(outcomes_by_task) if outcomes_by_task else None,
        max_budget_failure_rate=max_budget_failure_rate(outcomes_by_task),
        underbudget_rate=float(np.mean(under_flags)) if under_flags else None,
        overbudget_ratio=float(np.median(over_ratios)) if over_ratios else None,
        mean_regret=float(np.mean(regrets)) if regrets else None,
    )


def _observed_budget2success(outcomes: dict[int, bool]) -> int | None:
    for budget in sorted(outcomes):
        if outcomes[budget]:
            return budget
    return None
