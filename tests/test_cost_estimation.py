import json

import yaml

from scripts.estimate_experiment_cost import estimate_experiment_cost


def test_cost_estimation_uses_fake_pricing_and_config(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "source": "toy",
                "prompt": "Compute 2+2.",
                "answer": "4",
                "verifier": "numeric_exact",
                "budget_grid": [64, 128],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "suite.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "suite_name": "tiny",
                "task_file": str(task_file),
                "provider": "mock",
                "models": ["priced-model"],
                "forecast_prompt": "prompts/forecast_prompt.md",
                "solver_prompts": {"math": "prompts/solve_math_prompt.md"},
                "budget_grid": {"math": [64, 128]},
                "max_forecast_tokens": 100,
            }
        ),
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps({"pricing": {"priced-model": {"input_per_m": 1.0, "output_per_m": 2.0, "basis": "fake"}}}),
        encoding="utf-8",
    )

    estimate = estimate_experiment_cost(config, pricing_path=pricing, cap_usd=20)

    assert estimate["tasks"] == 1
    assert estimate["models"] == ["priced-model"]
    assert estimate["estimated_total_cost_usd"] > 0
    assert not estimate["exceeds_cap"]
    assert estimate["pricing_config_version"] == "2026-04-28"
    assert estimate["cost_mode"] == "active_artifact_estimate"
    assert estimate["reasoning_tokens_available"] is False


def test_cost_estimation_respects_forecast_only(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "source": "toy",
                "prompt": "Compute 2+2.",
                "answer": "4",
                "verifier": "numeric_exact",
                "budget_grid": [64, 128],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps({"pricing": {"priced-model": {"input_per_m": 1.0, "output_per_m": 2.0, "basis": "fake"}}}),
        encoding="utf-8",
    )

    base = {
        "suite_name": "tiny",
        "task_file": str(task_file),
        "provider": "mock",
        "models": ["priced-model"],
        "forecast_prompt": "prompts/forecast_prompt.md",
        "solver_prompts": {"math": "prompts/solve_math_prompt.md"},
        "budget_grid": {"math": [64, 128]},
        "max_forecast_tokens": 100,
    }
    full_config = tmp_path / "full.yaml"
    full_config.write_text(yaml.safe_dump(base), encoding="utf-8")
    forecast_config = tmp_path / "forecast.yaml"
    forecast_config.write_text(yaml.safe_dump({**base, "forecast_only": True}), encoding="utf-8")

    full = estimate_experiment_cost(full_config, pricing_path=pricing, cap_usd=20)
    forecast_only = estimate_experiment_cost(forecast_config, pricing_path=pricing, cap_usd=20)

    assert forecast_only["forecast_only"] is True
    assert forecast_only["estimated_total_cost_usd"] < full["estimated_total_cost_usd"]
    assert forecast_only["per_model"][0]["solver_call_cost_usd"] == 0.0
    assert forecast_only["per_model"][0]["by_source"][0]["solver_calls"] == 0


def test_cost_estimation_multiplies_prompt_variants(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "source": "toy",
                "prompt": "Compute 2+2.",
                "answer": "4",
                "verifier": "numeric_exact",
                "budget_grid": [64],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("Return JSON.", encoding="utf-8")
    prompt_b.write_text("Return JSON with calibrated probabilities.", encoding="utf-8")
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps({"pricing": {"priced-model": {"input_per_m": 1.0, "output_per_m": 2.0, "basis": "fake"}}}),
        encoding="utf-8",
    )
    base = {
        "suite_name": "tiny",
        "task_file": str(task_file),
        "provider": "mock",
        "models": ["priced-model"],
        "forecast_prompt": str(prompt_a),
        "prompt_variants": [str(prompt_a), str(prompt_b)],
        "budget_grid": {"math": [64]},
        "max_forecast_tokens": 100,
        "forecast_only": True,
        "metadata": {"repeats": {"forecast": 2, "solver": 1}},
    }
    config = tmp_path / "variants.yaml"
    config.write_text(yaml.safe_dump(base), encoding="utf-8")

    estimate = estimate_experiment_cost(config, pricing_path=pricing, cap_usd=20)

    assert estimate["repeats"] == 2
    assert estimate["prompt_variants"] == 2
    assert estimate["per_model"][0]["solver_call_cost_usd"] == 0.0
    assert estimate["estimated_total_cost_usd"] > 0
