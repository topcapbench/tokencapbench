from scripts.audit_forecast_leakage import solver_prompt_contains_forecast


def test_solver_prompt_does_not_contain_forecast_probability_or_rationale():
    forecast = {
        "p_success_by_budget": {"64": 0.25, "128": 0.75},
        "short_rationale": "This rationale must stay out of the solver prompt.",
    }
    solver_prompt = "Solve the task. Task: Compute 2+2."
    assert not solver_prompt_contains_forecast(forecast, solver_prompt)


def test_solver_prompt_leakage_detector_flags_rationale():
    forecast = {
        "p_success_by_budget": {"64": 0.25, "128": 0.75},
        "short_rationale": "leaked rationale",
    }
    assert solver_prompt_contains_forecast(forecast, "Task text. leaked rationale")
