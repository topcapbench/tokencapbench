import json

from budget2success.execution.livecodebench_bridge import LiveCodeBenchBridge
from budget2success.schemas.records import BudgetRunRecord, TaskRecord, VerificationResult


def _task(task_id="t1", external_id="q1"):
    return TaskRecord(
        task_id=task_id,
        track="coding",
        prompt="Write solve().",
        verifier="livecodebench",
        external_id=external_id,
        external_eval={"harness": "livecodebench", "task_id": external_id},
    )


def _outcome(task_id="t1", budget=256, solution="def solve(): pass"):
    return BudgetRunRecord(
        task_id=task_id,
        model="mock",
        budget=budget,
        solution=solution,
        success=False,
        verification=VerificationResult.error(error="official_harness_required"),
    )


def test_livecodebench_bridge_writes_predictions_grouped_by_budget(tmp_path):
    paths = LiveCodeBenchBridge(tmp_path).write_predictions_grouped_by_budget(
        [_task()],
        [_outcome(budget=128), _outcome(budget=512)],
        tmp_path / "predictions",
    )

    assert sorted(paths) == [128, 512]
    rows = json.loads(paths[128].read_text(encoding="utf-8"))
    assert rows == [{"question_id": "q1", "code_list": ["def solve(): pass"]}]


def test_livecodebench_bridge_parses_official_jsonl_results(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps({"question_id": "q1", "passed": True}) + "\n"
        + json.dumps({"question_id": "q2", "status": "wrong_answer"}) + "\n",
        encoding="utf-8",
    )

    parsed = LiveCodeBenchBridge(tmp_path).parse_official_results(result_path)

    assert parsed == {"q1": True, "q2": False}


def test_livecodebench_bridge_parses_official_eval_all_results(tmp_path):
    result_path = tmp_path / "results_eval_all.json"
    result_path.write_text(
        json.dumps(
            [
                {"question_id": "q1", "graded_list": [True], "pass@1": 1.0},
                {"question_id": "q2", "graded_list": [False], "pass@1": 0.0},
            ]
        ),
        encoding="utf-8",
    )

    parsed = LiveCodeBenchBridge(tmp_path).parse_official_results(result_path)

    assert parsed == {"q1": True, "q2": False}


def test_livecodebench_bridge_merges_official_labels(tmp_path):
    merged = LiveCodeBenchBridge(tmp_path).merge_official_results_into_outcomes(
        [_outcome()],
        [_task()],
        {256: {"q1": True}},
    )

    assert merged[0]["success"] is True
    assert merged[0]["metadata"]["label_source"] == "official_livecodebench"
    assert merged[0]["verification"]["metadata"]["label_source"] == "official_livecodebench"
