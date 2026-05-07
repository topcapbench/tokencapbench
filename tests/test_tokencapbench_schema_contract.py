from budget2success.forecasting.parse_forecast import parse_forecast_json
from budget2success.metrics.calibration import log_score
from budget2success.schemas.records import BudgetRunRecord, ForecastRecord, VerificationResult


def test_forecast_schema_accepts_public_contract_fields():
    row = ForecastRecord(
        run_id="r",
        suite="toy",
        task_id="t",
        model="m",
        solver_scaffold="direct",
        budget_grid=[64, 128],
        forecast={"64": 0.2, "128": 0.8},
        forecast_prompt_hash="hash",
    )
    assert row.benchmark_slug == "budget2success"
    assert row.p_success_by_budget == {"64": 0.2, "128": 0.8}
    assert row.scaffold == "direct"


def test_outcome_schema_accepts_public_resource_contract_fields():
    row = BudgetRunRecord(
        run_id="r",
        suite="toy",
        task_id="t",
        model="m",
        solver_scaffold="direct",
        budget=128,
        solution="answer",
        success=True,
        verification=VerificationResult.ok(),
        verifier="toy_verifier",
        verifier_version="v1",
        prompt_tokens=10,
        completion_tokens=12,
        total_visible_tokens=22,
        finish_reason="stop",
        cap_hit=False,
        truncated=False,
        retry_count=0,
        generation_wall_time_s=0.2,
        verification_wall_time_s=0.1,
        end_to_end_wall_time_s=0.3,
        provider_request_id_hash="abc",
    )
    assert row.total_tokens == 22
    assert row.generation_wall_time_seconds == 0.2
    assert row.scaffold == "direct"


def test_probability_clipping_records_repair_for_public_forecast_field():
    record = parse_forecast_json('{"forecast":{"64":-0.2,"128":"120%"}}')
    assert record.p_success_by_budget == {"64": 0.0, "128": 1.0}
    assert len(record.forecast_extras["probability_repairs"]) == 2


def test_log_score_clips_zero_and_one_probabilities():
    value = log_score([0.0, 1.0], [False, True])
    assert value >= 0.0
    assert value < 1e-6
