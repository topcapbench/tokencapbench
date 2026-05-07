from budget2success.schemas.records import BudgetRunRecord, ForecastRecord, TaskRecord, VerificationResult


def test_task_record_fills_provenance_defaults():
    task = TaskRecord(task_id="local_1", track="math", prompt="Compute.", verifier="numeric_exact", source="toy")
    assert task.source_version == "toy"
    assert task.external_id == "local_1"
    assert task.external_eval == {"harness": "numeric_exact", "source": "toy"}


def test_forecast_record_normalizes_budget_keys_and_probability_strings():
    forecast = ForecastRecord(p_success_by_budget={512: "75%", "256": "0.25"}, median_budget2success=512)
    assert forecast.p_success_by_budget == {"256": 0.25, "512": 0.75}
    assert forecast.p_failure_at_max_budget == 0.25


def test_budget_run_record_rejects_negative_token_counts():
    try:
        BudgetRunRecord(
            task_id="t",
            model="m",
            budget=128,
            solution="x",
            success=False,
            verification=VerificationResult.fail(),
            completion_tokens=-1,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_budget_run_record_accepts_finish_reason_and_truncation_flag():
    record = BudgetRunRecord(
        task_id="t",
        model="m",
        budget=128,
        solution="x",
        success=False,
        verification=VerificationResult.fail(),
        finish_reason="length",
        truncated=True,
    )

    assert record.finish_reason == "length"
    assert record.truncated is True


def test_budget_run_record_accepts_resource_timing_fields():
    record = BudgetRunRecord(
        task_id="t",
        model="m",
        budget=128,
        solution="x",
        success=True,
        verification=VerificationResult.ok(),
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        reasoning_tokens=5,
        wall_time_seconds=0.4,
        generation_wall_time_seconds=0.3,
        verification_wall_time_seconds=0.1,
        end_to_end_wall_time_seconds=0.4,
    )
    assert record.end_to_end_wall_time_seconds == 0.4
    assert record.generation_wall_time_seconds == 0.3
    assert record.verification_wall_time_seconds == 0.1
