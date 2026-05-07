#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.schemas.records import VerificationStatus
from budget2success.utils.jsonl import read_jsonl, write_jsonl


PASS = "pass"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-verify EvalPlus outcomes and merge official labels.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--work-dir", default="reports/evalplus_batch")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    outcomes_path = run_dir / "outcomes.jsonl"
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)

    tasks = load_tasks_jsonl(args.task_file)
    outcomes = read_jsonl(outcomes_path)
    task_by_id = {task.task_id: task for task in tasks}
    official_ids = _official_ids(args.dataset)
    work_root = Path(args.work_dir) / run_dir.parent.name / run_dir.name
    work_root.mkdir(parents=True, exist_ok=True)

    labels_by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    for budget in sorted({int(row["budget"]) for row in outcomes}):
        samples_path = work_root / f"samples_budget_{budget}.jsonl"
        result_path = samples_path.with_name(f"{samples_path.stem}_eval_results.json")
        if result_path.exists():
            result_path.unlink()
        _write_samples(samples_path, official_ids, tasks, outcomes, budget)
        command = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            args.dataset,
            "--samples",
            str(samples_path),
            "--parallel",
            str(max(1, int(args.parallel))),
        ]
        subprocess.run(command, check=True)
        labels_by_budget[budget] = _read_labels(result_path)

    backup = outcomes_path.with_suffix(".jsonl.before_evalplus_batch.bak")
    if not backup.exists():
        shutil.copy2(outcomes_path, backup)
    merged = [_merge_row(row, task_by_id, labels_by_budget) for row in outcomes]
    write_jsonl(outcomes_path, merged)
    print(json.dumps({"run_dir": str(run_dir), "outcomes": len(merged), "budgets": sorted(labels_by_budget)}, indent=2))


def _official_ids(dataset: str) -> list[str]:
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus

        return sorted(get_human_eval_plus().keys())
    from evalplus.data import get_mbpp_plus

    return sorted(get_mbpp_plus().keys())


def _write_samples(samples_path: Path, official_ids: list[str], tasks: list[Any], outcomes: list[dict[str, Any]], budget: int) -> None:
    solution_by_external: dict[str, str] = {}
    task_by_id = {task.task_id: task for task in tasks}
    for row in outcomes:
        if int(row.get("budget", -1)) != budget:
            continue
        task = task_by_id.get(str(row.get("task_id")))
        if task is None:
            continue
        external_id = str(task.external_id or task.task_id)
        solution_by_external[external_id] = str(row.get("solution") or "")

    with samples_path.open("w", encoding="utf-8") as handle:
        for official_id in official_ids:
            payload = {
                "task_id": official_id,
                "solution": solution_by_external.get(official_id, ""),
                "_identifier": f"{samples_path.stem}:{official_id}",
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_labels(result_path: Path) -> dict[str, dict[str, Any]]:
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    labels: dict[str, dict[str, Any]] = {}
    for task_id, rows in data.get("eval", {}).items():
        if rows:
            labels[str(task_id)] = rows[0]
    return labels


def _merge_row(
    row: dict[str, Any],
    task_by_id: dict[str, Any],
    labels_by_budget: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    task = task_by_id[str(row["task_id"])]
    external_id = str(task.external_id or task.task_id)
    budget = int(row["budget"])
    label = labels_by_budget[budget][external_id]
    success = label.get("base_status") == PASS and label.get("plus_status") == PASS
    merged = dict(row)
    merged["success"] = bool(success)
    verification = dict(merged.get("verification") or {})
    verification["status"] = VerificationStatus.SUCCESS.value if success else VerificationStatus.FAILURE.value
    verification["success"] = bool(success)
    verification["details"] = {
        "harness": "evalplus",
        "dataset": task.external_eval.get("dataset") or task.source,
        "task_id": external_id,
        "base_status": label.get("base_status"),
        "plus_status": label.get("plus_status"),
    }
    verification["metadata"] = {"label_source": "official_evalplus_batch"}
    merged["verification"] = verification
    metadata = dict(merged.get("metadata") or {})
    metadata["label_source"] = "official_evalplus_batch"
    merged["metadata"] = metadata
    merged["verifier"] = "evalplus"
    merged["verifier_version"] = "official_evalplus_batch"
    return merged


if __name__ == "__main__":
    main()
