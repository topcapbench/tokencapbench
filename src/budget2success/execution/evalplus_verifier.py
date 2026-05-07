from __future__ import annotations

from functools import lru_cache
from typing import Any

from budget2success.execution.coding_verifier import extract_python_code
from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult


class EvalPlusOfficialVerifier(Verifier):
    """Per-task wrapper around the official EvalPlus evaluator functions."""

    def __init__(
        self,
        dataset: str = "humaneval",
        base_only: bool = False,
        min_time_limit: float = 1.0,
        gt_time_limit_factor: float = 4.0,
        mini: bool = False,
        noextreme: bool = False,
        version: str = "default",
    ) -> None:
        self.dataset = dataset
        self.base_only = base_only
        self.min_time_limit = min_time_limit
        self.gt_time_limit_factor = gt_time_limit_factor
        self.mini = mini
        self.noextreme = noextreme
        self.version = version

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        dataset = _normalize_dataset(str(task.external_eval.get("dataset") or self.dataset))
        task_id = task.external_id or task.metadata.get("raw_task_id") or task.task_id
        try:
            official = _load_evalplus(dataset, self.mini, self.noextreme, self.version)
        except Exception as exc:  # noqa: BLE001 - report missing optional harness clearly.
            return VerificationResult.error(error="evalplus_unavailable", message=str(exc))

        problems = official["problems"]
        expected_output = official["expected_output"]
        if task_id not in problems:
            return VerificationResult.error(error="evalplus_task_not_found", dataset=dataset, task_id=task_id)

        try:
            result = official["check_correctness"](
                dataset,
                0,
                problems[task_id],
                extract_python_code(solution),
                expected_output[task_id],
                self.base_only,
                True,
                f"{task_id}:{task.task_id}",
                self.min_time_limit,
                self.gt_time_limit_factor,
            )
        except Exception as exc:  # noqa: BLE001 - official evaluator failures should be logged, not raised.
            return VerificationResult.error(error="evalplus_exception", message=str(exc), dataset=dataset, task_id=task_id)

        base_status = result["base"][0]
        plus_status = result.get("plus", (None,))[0] if not self.base_only else None
        pass_label = official["PASS"]
        success = base_status == pass_label and (self.base_only or plus_status == pass_label)
        details: dict[str, Any] = {
            "harness": "evalplus",
            "dataset": dataset,
            "task_id": task_id,
            "base_status": base_status,
            "plus_status": plus_status,
        }
        if success:
            return VerificationResult.ok(**details)
        return VerificationResult.fail(**details)


def _normalize_dataset(dataset: str) -> str:
    key = dataset.lower().replace("_", "").replace("-", "")
    if key in {"humaneval", "humanevalplus"}:
        return "humaneval"
    if key in {"mbpp", "mbppplus"}:
        return "mbpp"
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


@lru_cache(maxsize=8)
def _load_evalplus(dataset: str, mini: bool, noextreme: bool, version: str) -> dict[str, Any]:
    try:
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
        from evalplus.eval import PASS
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
        from evalplus.evaluate import check_correctness, get_groundtruth
    except ImportError as exc:
        raise RuntimeError("Install `evalplus` to use EvalPlusOfficialVerifier.") from exc

    if dataset == "humaneval":
        problems = get_human_eval_plus(mini=mini, noextreme=noextreme, version=version)
        dataset_hash = get_human_eval_plus_hash(mini=mini, noextreme=noextreme, version=version)
        expected_output = get_groundtruth(problems, dataset_hash, [])
    elif dataset == "mbpp":
        problems = get_mbpp_plus(mini=mini, noextreme=noextreme, version=version)
        dataset_hash = get_mbpp_plus_hash(mini=mini, noextreme=noextreme, version=version)
        expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)
    else:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")

    return {
        "PASS": PASS,
        "check_correctness": check_correctness,
        "expected_output": expected_output,
        "problems": problems,
    }
