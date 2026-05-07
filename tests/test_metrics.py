from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.metrics.regret import normalized_budget_regret, oracle_utility, selected_budget_from_forecast
from budget2success.metrics.first_success_budget import (
    absolute_log_budget_error,
    forecast_monotonicity_violation_rate,
    log_token_error,
    observed_budget2success,
    outcome_nonmonotonicity_rate,
    overbudget_rate,
    overbudget_waste_factor,
    pairwise_ranking_accuracy,
    signed_log_budget_error,
    task_budget_ranking_accuracy,
    truncation_rate,
    underbudget_rate,
    underbudget_shortfall_factor,
)


def test_brier_score():
    assert round(brier_score([0.0, 1.0], [False, True]), 6) == 0.0


def test_expected_calibration_error_perfect():
    assert expected_calibration_error([0.0, 1.0], [False, True]) == 0.0


def test_observed_budget2success():
    assert observed_budget2success({512: False, 1024: True}) == 1024
    assert observed_budget2success({512: False}) is None


def test_log_token_error():
    assert log_token_error(1000, 1000) == 0


def test_signed_log_budget_error_direction():
    assert signed_log_budget_error(1000, 2000) < 0
    assert signed_log_budget_error(4000, 1000) > 0
    assert absolute_log_budget_error(1000, 1000) == 0


def test_underbudget_and_overbudget_rates_and_factors():
    predicted = {"a": 1000, "b": 4000, "c": None, "d": 0}
    observed = {"a": 2000, "b": 1000, "c": 1000, "d": 1000}

    assert underbudget_rate(predicted, observed) == 0.5
    assert overbudget_rate(predicted, observed) == 0.5
    assert underbudget_shortfall_factor(1000, 2000) == 2.0
    assert overbudget_waste_factor(4000, 1000) == 4.0
    assert signed_log_budget_error(0, 1000) is None


def test_regret_budget_selection():
    chosen = selected_budget_from_forecast({512: 0.2, 1024: 0.9}, reward=1.0, token_cost=0.0)
    assert chosen == 1024
    assert oracle_utility({512: False, 1024: True}) == 1.0


def test_normalized_regret_zero_for_oracle_budget():
    assert normalized_budget_regret({512: False, 1024: True}, 1024) == 0.0


def test_normalized_regret_stable_for_simple_example():
    value = normalized_budget_regret({512: False, 1024: True}, 512)
    assert 0.0 <= value <= 1.0
    assert value == 1.0


def test_normalized_regret_zero_range_does_not_crash():
    value, metadata = normalized_budget_regret({512: True, 1024: True}, 512, return_metadata=True)
    assert value == 0.0
    assert metadata["zero_range"] is True


def test_pairwise_ranking_accuracy():
    acc = pairwise_ranking_accuracy({"a": 100, "b": 1000}, {"a": 128, "b": 2048})
    assert acc == 1.0


def test_diagnostic_metrics():
    assert forecast_monotonicity_violation_rate({"a": {64: 0.6, 128: 0.4}, "b": {64: 0.1, 128: 0.2}}) == 0.5
    assert outcome_nonmonotonicity_rate({"a": {64: True, 128: False}, "b": {64: False, 128: True}}) == 0.5
    assert task_budget_ranking_accuracy({"a": 64, "b": 128}, {"a": {64: True}, "b": {64: False, 128: True}}) == 1.0
    assert truncation_rate([{"completion_tokens": 64, "budget": 64}, {"truncated": False}]) == 0.5
