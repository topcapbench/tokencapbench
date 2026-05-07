import json
import subprocess
import sys

import yaml

import scripts.make_paper_tables as make_paper_tables
from scripts.build_repeatability_subset import build_repeatability_subset
from scripts.run_budget_grid import _repeat_ids
from budget2success.analysis.paper import PaperRun
from budget2success.utils.jsonl import read_jsonl


def test_build_repeatability_subset_is_deterministic(tmp_path):
    math_source = tmp_path / "math.jsonl"
    coding_source = tmp_path / "coding.jsonl"
    math_rows = [
        {"task_id": f"m{i}", "track": "math", "prompt": "p", "verifier": "numeric_exact", "answer": "1", "source": "gsm8k"}
        for i in range(8)
    ]
    coding_rows = [
        {"task_id": f"c{i}", "track": "coding", "prompt": "p", "verifier": "evalplus", "source": "evalplus_humaneval"}
        for i in range(8)
    ]
    math_source.write_text("\n".join(json.dumps(row) for row in math_rows) + "\n", encoding="utf-8")
    coding_source.write_text("\n".join(json.dumps(row) for row in coding_rows) + "\n", encoding="utf-8")

    first = build_repeatability_subset(
        math_source=math_source,
        coding_source=coding_source,
        math_limit=4,
        coding_limit=4,
        seed=7,
        output=tmp_path / "repeat1.jsonl",
    )
    second = build_repeatability_subset(
        math_source=math_source,
        coding_source=coding_source,
        math_limit=4,
        coding_limit=4,
        seed=7,
        output=tmp_path / "repeat2.jsonl",
    )

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    rows = read_jsonl(first)
    assert len(rows) == 8
    assert {tuple(row["budget_grid"]) for row in rows if row["track"] == "math"} == {(128, 512, 2048)}
    assert {tuple(row["budget_grid"]) for row in rows if row["track"] == "coding"} == {(256, 1024, 2048)}


def test_run_experiment_suite_repeatability_dry_run_has_distinct_run_ids(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "prompt": "Compute.",
                "verifier": "numeric_exact",
                "answer": "1",
                "source": "gsm8k",
                "budget_grid": [128],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps({"pricing": {"mock-model": {"input_per_m": 0, "output_per_m": 0}}}), encoding="utf-8")
    config = tmp_path / "repeat.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "suite_name": "paper_repeatability_small",
                "task_file": str(task_file),
                "output_root": str(tmp_path / "runs"),
                "provider": "mock",
                "models": ["mock-model"],
                "budget_grid": {"math": [128]},
                "metadata": {"repeats": {"forecast": 2, "solver": 2}},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_suite.py",
            "--config",
            str(config),
            "--pricing",
            str(pricing),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "mock_model__forecast_repeat_1__solver_repeat_1" in result.stdout
    assert "mock_model__forecast_repeat_2__solver_repeat_2" in result.stdout


def test_run_experiment_suite_prompt_variants_dry_run_have_distinct_run_ids(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "prompt": "Compute.",
                "verifier": "numeric_exact",
                "answer": "1",
                "source": "gsm8k",
                "budget_grid": [128],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pricing = tmp_path / "pricing.json"
    pricing.write_text(json.dumps({"pricing": {"mock-model": {"input_per_m": 0, "output_per_m": 0}}}), encoding="utf-8")
    prompt_a = tmp_path / "forecast_prompt.md"
    prompt_b = tmp_path / "forecast_prompt_terse.md"
    prompt_a.write_text("Return JSON.", encoding="utf-8")
    prompt_b.write_text("Return terse JSON.", encoding="utf-8")
    config = tmp_path / "variants.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "suite_name": "paper_forecast_stability",
                "task_file": str(task_file),
                "output_root": str(tmp_path / "runs"),
                "provider": "mock",
                "models": ["mock-model"],
                "budget_grid": {"math": [128]},
                "prompt_variants": [str(prompt_a), str(prompt_b)],
                "metadata": {"repeats": {"forecast": 2, "solver": 1}},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_experiment_suite.py",
            "--config",
            str(config),
            "--pricing",
            str(pricing),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "mock_model__prompt_forecast_prompt__forecast_repeat_1__solver_repeat_1" in result.stdout
    assert "mock_model__prompt_forecast_prompt_terse__forecast_repeat_2__solver_repeat_1" in result.stdout


def test_run_budget_grid_repeat_ids_handles_nested_metadata_repeats():
    repeat_ids = _repeat_ids({"metadata": {"repeats": {"forecast": 2, "solver": 3}}})

    assert repeat_ids == [
        "forecast_1__solver_1",
        "forecast_1__solver_2",
        "forecast_1__solver_3",
        "forecast_2__solver_1",
        "forecast_2__solver_2",
        "forecast_2__solver_3",
    ]


def test_repeatability_table_aggregates_split_repeat_run_dirs(monkeypatch, tmp_path):
    def outcome(task_id, budget, repeat_id, success):
        return {
            "task_id": task_id,
            "budget": budget,
            "success": success,
            "metadata": {"repeat_id": repeat_id, "source": "gsm8k"},
        }

    runs = [
        PaperRun(
            run_dir=tmp_path / "run1",
            run_id="run1",
            model="mock-model",
            suite="paper_repeatability_small",
            forecasts=[],
            outcomes=[outcome("t1", 128, "forecast_1__solver_1", True)],
            metrics={},
            config={},
        ),
        PaperRun(
            run_dir=tmp_path / "run2",
            run_id="run2",
            model="mock-model",
            suite="paper_repeatability_small",
            forecasts=[],
            outcomes=[outcome("t1", 128, "forecast_1__solver_2", False)],
            metrics={},
            config={},
        ),
    ]
    monkeypatch.setattr(make_paper_tables, "load_paper_runs", lambda **_kwargs: runs)

    rows = make_paper_tables._repeatability_audit_rows()

    assert len(rows) == 1
    assert rows[0]["n_repeats"] == 2
    assert rows[0]["success_agreement_rate"] == 0.0
