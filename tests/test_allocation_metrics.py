from budget2success.metrics.first_success_budget import (
    overbudget_rate,
    overbudget_waste_factor,
    signed_log_budget_error,
    underbudget_rate,
    underbudget_shortfall_factor,
)


def test_allocation_metric_contracts():
    assert signed_log_budget_error(1000, 2000) < 0
    assert signed_log_budget_error(4000, 1000) > 0
    assert underbudget_rate({"a": 1, "b": 3}, {"a": 2, "b": 2}) == 0.5
    assert overbudget_rate({"a": 1, "b": 3}, {"a": 2, "b": 2}) == 0.5
    assert underbudget_shortfall_factor(1, 2) == 2
    assert overbudget_waste_factor(3, 1) == 3
    assert signed_log_budget_error(None, 2) is None
