from budget2success.forecasting.parse_forecast import parse_forecast_json, validate_forecast_budget_grid


def test_parse_forecast_plain_json():
    record = parse_forecast_json(
        '{"p_success_by_budget":{"256":0.1,"512":0.2},"median_budget2success":512,"p_failure_at_max_budget":0.8}'
    )
    assert record.p_success_by_budget["256"] == 0.1
    assert record.median_budget2success == 512


def test_parse_forecast_markdown_json():
    record = parse_forecast_json(
        '```json\n{"p_success_by_budget":{"256":0.1},"median_budget2success":256,"p_failure_at_max_budget":0.9}\n```'
    )
    assert record.p_success_by_budget["256"] == 0.1


def test_parse_forecast_accepts_public_median_success_budget_alias():
    record = parse_forecast_json('{"p_success_by_budget":{"256":0.1},"median_success_budget":256}')
    assert record.median_budget2success == 256


def test_parse_forecast_clips_bad_probability_and_records_repair():
    record = parse_forecast_json('{"p_success_by_budget":{"256":1.5},"p_failure_at_max_budget":0.0}')
    assert record.p_success_by_budget["256"] == 1.0
    repairs = record.forecast_extras["probability_repairs"]
    assert repairs[0]["budget"] == "256"
    assert repairs[0]["raw_probability"] == 1.5


def test_parse_forecast_uses_first_balanced_object_with_trailing_text():
    record = parse_forecast_json(
        'Notes with {not json}. {"p_success_by_budget":{"256":"10%","512":0.2},'
        '"median_budget2success":512,"p_failure_at_max_budget":0.8} trailing {ignored}'
    )
    assert record.p_success_by_budget == {"256": 0.1, "512": 0.2}


def test_validate_forecast_budget_grid_rejects_missing_budget():
    record = parse_forecast_json('{"p_success_by_budget":{"256":0.1},"median_budget2success":256}')
    try:
        validate_forecast_budget_grid(record, [256, 512])
    except ValueError as exc:
        assert "missing=[512]" in str(exc)
        return
    raise AssertionError("Expected ValueError")


def test_parse_forecast_preserves_extra_usage_fields():
    record = parse_forecast_json(
        '{"p_success_by_budget":{"256":0.1},"predicted_first_success_budget":256,'
        '"predicted_visible_tokens_unconstrained":800,'
        '"predicted_output_tokens_unconstrained":700,'
        '"predicted_total_visible_tokens":900,'
        '"predicted_unconstrained_output_tokens":700,'
        '"predicted_total_visible_tokens_to_solve":900,"confidence":0.5}'
    )

    assert record.predicted_unconstrained_output_tokens == 700
    assert record.forecast_extras["predicted_first_success_budget"] == 256
    assert record.forecast_extras["predicted_visible_tokens_unconstrained"] == 800
    assert record.forecast_extras["predicted_output_tokens_unconstrained"] == 700
    assert record.forecast_extras["predicted_total_visible_tokens"] == 900
    assert record.forecast_extras["predicted_total_visible_tokens_to_solve"] == 900
    assert record.forecast_extras["confidence"] == 0.5


def test_parse_forecast_accepts_optional_min_tokens_for_success():
    record = parse_forecast_json(
        '{"p_success_by_budget":{"256":0.1},"predicted_min_tokens_for_success":256,'
        '"predicted_unconstrained_output_tokens":700}'
    )

    assert record.predicted_min_tokens_for_success == 256
    assert record.predicted_unconstrained_output_tokens == 700
