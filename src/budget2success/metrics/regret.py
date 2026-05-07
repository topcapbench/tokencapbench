from __future__ import annotations


def utility(success: bool, budget: int, reward: float = 1.0, token_cost: float = 0.0) -> float:
    return (reward if success else 0.0) - token_cost * budget


def oracle_utility(outcomes: dict[int, bool], reward: float = 1.0, token_cost: float = 0.0) -> float:
    if not outcomes:
        return 0.0
    return max(utility(success, budget, reward, token_cost) for budget, success in outcomes.items())


def selected_budget_from_forecast(
    p_success_by_budget: dict[int, float], reward: float = 1.0, token_cost: float = 0.0
) -> int:
    return max(
        p_success_by_budget,
        key=lambda budget: p_success_by_budget[budget] * reward - token_cost * budget,
    )


def normalized_budget_regret(
    outcomes: dict[int, bool],
    selected_budget: int,
    reward: float = 1.0,
    token_cost: float = 0.0,
    *,
    return_metadata: bool = False,
) -> float | tuple[float, dict[str, float | bool | str]]:
    """Return budget regret normalized by the observed oracle utility range.

    If all observed budgets have identical utility, the deployment decision has
    no observed utility range; in that zero-range case the normalized regret is
    defined as 0.0 and the metadata marks the normalizer used.
    """
    if not outcomes:
        metadata: dict[str, float | bool | str] = {"utility_range": 0.0, "normalizer": 0.0, "zero_range": True}
        return (0.0, metadata) if return_metadata else 0.0
    utilities = [utility(success, budget, reward, token_cost) for budget, success in outcomes.items()]
    oracle = max(utilities)
    selected = utility(outcomes.get(selected_budget, False), selected_budget, reward, token_cost)
    regret = max(0.0, oracle - selected)
    utility_range = max(utilities) - min(utilities)
    if utility_range <= 0:
        metadata = {"utility_range": float(utility_range), "normalizer": 0.0, "zero_range": True}
        return (0.0, metadata) if return_metadata else 0.0
    value = regret / utility_range
    metadata = {"utility_range": float(utility_range), "normalizer": float(utility_range), "zero_range": False}
    return (value, metadata) if return_metadata else value
