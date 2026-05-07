import json

from budget2success.datasets.base import AdapterConfig
from budget2success.datasets.loaders.livecodebench import LiveCodeBenchAdapter
from budget2success.execution.evalplus_bridge import EvalPlusBridge
from budget2success.execution.external_harness import run_command
from budget2success.execution.livecodebench_bridge import LiveCodeBenchBridge
from budget2success.execution.swebench_bridge import SWEBenchBridge
from budget2success.schemas.records import BudgetRunRecord, TaskRecord, VerificationResult
from scripts.run_swebench_official import find_swebench_report, parse_swebench_report


def test_external_harness_missing_executable_fails_clearly():
    result = run_command(["definitely_missing_budget2success_command"])
    assert not result.success
    assert result.returncode == -127
    assert "not found" in result.stderr


def test_evalplus_bridge_exports_official_sample_shape(tmp_path):
    task = TaskRecord(
        task_id="evalplus_HumanEval/0",
        track="coding",
        prompt="Write code.",
        verifier="evalplus",
        external_id="HumanEval/0",
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="mock",
        budget=512,
        solution="def f():\n    pass\n",
        success=False,
        verification=VerificationResult.fail(),
    )
    path = EvalPlusBridge(tmp_path).write_samples_from_records([task], [outcome])
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row == {"solution": outcome.solution, "task_id": "HumanEval/0"}


def test_swebench_bridge_exports_prediction_shape(tmp_path):
    task = TaskRecord(
        task_id="swebench_x",
        track="swe",
        prompt="Fix bug.",
        verifier="swebench",
        external_id="repo__issue-1",
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="mock",
        budget=4096,
        solution="diff --git a/file.py b/file.py\n",
        success=False,
        verification=VerificationResult.fail(),
    )
    path = SWEBenchBridge(tmp_path).write_predictions_from_records([task], [outcome], model_name_or_path="mock")
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["instance_id"] == "repo__issue-1"
    assert row["model_name_or_path"] == "mock"
    assert row["model_patch"] == outcome.solution


def test_swebench_official_report_discovery_and_parse(tmp_path):
    report = tmp_path / "nested" / "run_report.json"
    report.parent.mkdir()
    report.write_text(
        json.dumps({"instance_id_to_report": {"repo__1": {"resolved": True}, "repo__2": {"resolved": False}}}),
        encoding="utf-8",
    )

    found = find_swebench_report(tmp_path, "run")

    assert found == report
    assert parse_swebench_report(found) == {"repo__1": True, "repo__2": False}


def test_livecodebench_bridge_exports_prediction_shape(tmp_path):
    task = TaskRecord(
        task_id="livecodebench_1",
        track="coding",
        prompt="Write code.",
        verifier="livecodebench",
        external_id="question-1",
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="mock",
        budget=512,
        solution="def solve():\n    pass\n",
        success=False,
        verification=VerificationResult.fail(),
    )
    path = LiveCodeBenchBridge(tmp_path).write_predictions_from_records([task], [outcome])
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows == [{"question_id": "question-1", "code_list": [outcome.solution.strip()]}]


def test_livecodebench_adapter_loads_local_official_export(tmp_path):
    export = tmp_path / "lcb.jsonl"
    export.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_content": "Write solve().",
                "input_output": {"inputs": [], "outputs": []},
                "contest_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = LiveCodeBenchAdapter(
        AdapterConfig(name="livecodebench", split="test", limit=1, budget_grid=[256], kwargs={"path": str(export)})
    )

    tasks = adapter.load_tasks()

    assert tasks[0].source == "livecodebench"
    assert tasks[0].track == "coding"
    assert tasks[0].external_eval["harness"] == "livecodebench"
    assert tasks[0].external_eval["input_output"] == {"inputs": [], "outputs": []}


def test_livecodebench_bridge_refuses_non_official_metadata(tmp_path):
    task = TaskRecord(
        task_id="local_1",
        track="coding",
        prompt="Write code.",
        verifier="livecodebench",
        external_id="question-1",
        external_eval={"harness": "local_smoke_only"},
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="mock",
        budget=512,
        solution="def solve():\n    pass\n",
        success=False,
        verification=VerificationResult.fail(),
    )

    try:
        LiveCodeBenchBridge(tmp_path).write_predictions_from_records([task], [outcome])
    except ValueError as exc:
        assert "official LiveCodeBench" in str(exc)
        return
    raise AssertionError("Expected official metadata failure")
