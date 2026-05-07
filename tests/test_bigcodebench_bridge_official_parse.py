import json

from budget2success.execution.bigcodebench_bridge import BigCodeBenchBridge
from budget2success.schemas.records import BudgetRunRecord, TaskRecord, VerificationResult


def test_bigcodebench_bridge_parses_official_jsonl_results(tmp_path):
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(
        json.dumps({"task_id": "1", "passed": True}) + "\n"
        + json.dumps({"task_id": "2", "status": "wrong_answer"}) + "\n",
        encoding="utf-8",
    )

    parsed = BigCodeBenchBridge(tmp_path).parse_official_results(result_path)

    assert parsed == {"1": True, "2": False}


def test_bigcodebench_bridge_merges_official_labels(tmp_path):
    task = TaskRecord(
        task_id="bigcodebench_hard_1",
        track="coding",
        prompt="Write code.",
        verifier="bigcodebench_official",
        external_id="1",
        external_eval={"harness": "bigcodebench", "task_id": "1"},
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="mock",
        budget=256,
        solution="def f(): pass",
        success=False,
        verification=VerificationResult.error(error="official_harness_required"),
    )

    merged = BigCodeBenchBridge(tmp_path).merge_official_results_into_outcomes([outcome], [task], {"1": True})

    assert merged[0]["success"] is True
    assert merged[0]["metadata"]["label_source"] == "official_bigcodebench"
    assert merged[0]["verification"]["metadata"]["label_source"] == "official_bigcodebench"


def test_bigcodebench_bridge_writes_grouped_predictions_to_requested_root(tmp_path):
    task = TaskRecord(
        task_id="bigcodebench_hard_1",
        track="coding",
        prompt="Write code.",
        verifier="bigcodebench",
        external_id="1",
        external_eval={"harness": "bigcodebench", "task_id": "1"},
    )
    outcome = BudgetRunRecord(
        task_id=task.task_id,
        model="m",
        budget=256,
        solution="def f(): pass",
        success=False,
        verification=VerificationResult.error(error="official_harness_required"),
    )

    paths = BigCodeBenchBridge(tmp_path / "bridge").write_predictions_grouped_by_budget(
        [task],
        [outcome],
        tmp_path / "requested",
    )

    assert paths[256].parent == tmp_path / "requested"
    assert paths[256].exists()
