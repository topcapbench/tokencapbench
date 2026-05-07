import csv
import json

from scripts.reverify_outcomes import reverify_outcomes
from budget2success.execution.math_verifier import TaskAwareMathVerifier
from budget2success.utils.jsonl import read_jsonl


def test_reverify_outcomes_writes_corrections_and_corrected_artifacts(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "prompt": "Compute.",
                "verifier": "numeric_exact",
                "answer": "4",
                "source": "toy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "config_snapshot.yaml").write_text(f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n", encoding="utf-8")
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.8}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t1",
                "model": "m",
                "budget": 64,
                "solution": "We saw 4 but Final answer: 5",
                "success": True,
                "metadata": {"track": "math", "source": "toy"},
                "verification": {"status": "success", "success": True, "details": {"mode": "lenient"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    reverify_outcomes(
        artifact_root=tmp_path / "artifacts",
        task_files=[task_file],
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "summary.csv",
        write_corrections=tmp_path / "corrections.jsonl",
        write_corrected_outcomes=tmp_path / "corrected",
        mode="strict",
    )

    corrections = read_jsonl(tmp_path / "corrections.jsonl")
    assert corrections[0]["old_success"] is True
    assert corrections[0]["new_success"] is False
    corrected = read_jsonl(tmp_path / "corrected" / "paper_math_core__m" / "outcomes.jsonl")
    assert corrected[0]["success"] is False
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["n_changed"] == "1"


def test_task_default_selects_math_verify_for_math_style_tasks(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "m1",
                "track": "math",
                "prompt": "Solve.",
                "verifier": "math_verify_optional",
                "answer": r"\left(3, \frac{\pi}{2}\right)",
                "source": "hendrycks_math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "config_snapshot.yaml").write_text(f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n", encoding="utf-8")
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "m1", "model": "m", "p_success_by_budget": {"64": 0.8}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "m1",
                "model": "m",
                "budget": 64,
                "solution": "Final answer: 0",
                "success": True,
                "metadata": {"track": "math", "source": "hendrycks_math"},
                "verification": {"status": "success", "success": True, "details": {"mode": "old"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    reverify_outcomes(
        artifact_root=tmp_path / "artifacts",
        task_files=[task_file],
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "summary.csv",
        write_corrections=tmp_path / "corrections.jsonl",
        mode="task_default",
    )

    corrections = read_jsonl(tmp_path / "corrections.jsonl")
    assert corrections[0]["answer_type"] == "tuple_or_coordinate"
    assert corrections[0]["verifier_selected"] == "math_verify_optional"
    assert "math_verify_available" in corrections[0]
    with (tmp_path / "audit.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["answer_type"] == "tuple_or_coordinate"
    assert rows[0]["verifier_selected"] == "math_verify_optional"


def test_task_default_keeps_gsm8k_strict_numeric_behavior(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "g1",
                "track": "math",
                "prompt": "Compute.",
                "verifier": "numeric_exact",
                "answer": "4",
                "source": "gsm8k",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "config_snapshot.yaml").write_text(f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n", encoding="utf-8")
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "g1", "model": "m", "p_success_by_budget": {"64": 0.8}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "g1",
                "model": "m",
                "budget": 64,
                "solution": "We saw 4, but final answer: 5",
                "success": True,
                "metadata": {"track": "math", "source": "gsm8k"},
                "verification": {"status": "success", "success": True, "details": {"mode": "lenient"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    reverify_outcomes(
        artifact_root=tmp_path / "artifacts",
        task_files=[task_file],
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "summary.csv",
        write_corrections=tmp_path / "corrections.jsonl",
        mode="task_default",
    )

    corrections = read_jsonl(tmp_path / "corrections.jsonl")
    assert corrections[0]["new_success"] is False
    assert corrections[0]["verifier_selected"] == "numeric_exact_strict"


def test_task_aware_strict_writes_unsupported_rows_when_math_verify_missing(tmp_path, monkeypatch):
    class MissingMathVerifyTaskAware(TaskAwareMathVerifier):
        def __init__(self, require_math_verify_for_symbolic: bool = False):
            super().__init__(require_math_verify_for_symbolic=require_math_verify_for_symbolic)
            self._math_verify._parse = None
            self._math_verify._verify = None

    monkeypatch.setattr("scripts.reverify_outcomes.TaskAwareMathVerifier", MissingMathVerifyTaskAware)

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "m1",
                "track": "math",
                "prompt": "Solve.",
                "verifier": "math_verify_optional",
                "answer": r"x^2",
                "source": "hendrycks_math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    (run_dir / "config_snapshot.yaml").write_text(f"suite: paper_math_core\nmodel: m\nrun_id: m\ntask_file: {task_file}\n", encoding="utf-8")
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "m1", "model": "m", "p_success_by_budget": {"64": 0.8}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "m1",
                "model": "m",
                "budget": 64,
                "solution": r"\boxed{x^2}",
                "success": True,
                "metadata": {"track": "math", "source": "hendrycks_math"},
                "verification": {"status": "success", "success": True, "details": {"mode": "old"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    reverify_outcomes(
        artifact_root=tmp_path / "artifacts",
        task_files=[task_file],
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "summary.csv",
        unsupported_output=tmp_path / "unsupported.csv",
        write_corrections=tmp_path / "corrections.jsonl",
        mode="task_aware_strict",
    )

    unsupported_rows = list(csv.DictReader((tmp_path / "unsupported.csv").open(encoding="utf-8", newline="")))
    assert unsupported_rows[0]["error"] == "math_verify_required"
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as f:
        summary = list(csv.DictReader(f))
    assert summary[0]["unsupported_rows"] == "1"
    corrections = read_jsonl(tmp_path / "corrections.jsonl")
    assert corrections[0]["unsupported"] is True
