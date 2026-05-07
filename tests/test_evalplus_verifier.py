import pytest

from budget2success.execution.evalplus_verifier import EvalPlusOfficialVerifier
from budget2success.schemas.records import TaskRecord


def test_evalplus_verifier_accepts_canonical_solution():
    evalplus_data = pytest.importorskip("evalplus.data")
    problems = evalplus_data.get_human_eval_plus()
    problem = problems["HumanEval/0"]
    task = TaskRecord(
        task_id="evalplus_HumanEval/0",
        track="coding",
        prompt=problem["prompt"],
        verifier="evalplus",
        source="evalplus_humaneval",
        external_id="HumanEval/0",
        external_eval={"harness": "evalplus", "dataset": "humaneval", "task_id": "HumanEval/0"},
    )
    result = EvalPlusOfficialVerifier(dataset="humaneval").verify(task, problem["prompt"] + problem["canonical_solution"])
    assert result.success
    assert result.details["base_status"] == "pass"
    assert result.details["plus_status"] == "pass"
