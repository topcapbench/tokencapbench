#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_paper_runs
from budget2success.execution.math_verifier import (
    MathVerifyOptionalVerifier,
    NumericExactVerifier,
    TaskAwareMathVerifier,
    classify_math_answer,
)
from budget2success.execution.verifier_registry import get_verifier
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl, write_jsonl


def reverify_outcomes(
    *,
    artifact_root: str | Path = "reports/artifacts",
    task_files: list[str | Path] | None = None,
    output: str | Path = "reports/tables/math_reverification_audit.csv",
    summary_output: str | Path = "reports/tables/math_verifier_delta_summary.csv",
    unsupported_output: str | Path = "reports/tables/math_reverification_unsupported.csv",
    write_corrections: str | Path | None = None,
    write_corrected_outcomes: str | Path | None = None,
    mode: str = "strict",
) -> Path:
    runs = load_paper_runs(artifact_root=artifact_root, include_artifacts=True)
    task_files = task_files or _task_files_from_runs(runs)
    tasks = _load_tasks(task_files)
    verifier_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    unsupported_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "n_rows": 0,
            "supported_rows": 0,
            "unsupported_rows": 0,
            "n_changed": 0,
            "old_success": 0,
            "new_success": 0,
            "verifier_policy": "",
            "math_verify_available": False,
        }
    )
    for run in runs:
        corrected_rows: list[dict[str, Any]] = []
        run_has_math = False
        for outcome in run.outcomes:
            task_id = str(outcome.get("task_id"))
            task = tasks.get(task_id)
            if task is None or task.track != "math" or task.answer is None:
                corrected_rows.append(outcome)
                continue
            run_has_math = True
            verifier, verifier_selected = _verifier_for_task(task, mode, verifier_cache)
            result = verifier.verify(task, str(outcome.get("solution") or ""))
            old_success = bool(outcome.get("success"))
            source = str((outcome.get("metadata") or {}).get("source") or task.source or "unknown")
            answer_type = classify_math_answer(task.answer)
            math_verify_available = bool(result.metadata.get("math_verify_available", False))
            unsupported = result.details.get("error") == "math_verify_required"
            new_success = old_success if unsupported else bool(result.success)
            changed = (old_success != new_success) if not unsupported else False
            old_verifier = _old_verifier(outcome)
            new_verifier = verifier_selected
            correction = {
                "suite": run.suite or "",
                "run_id": run.run_id,
                "task_id": task_id,
                "budget": outcome.get("budget"),
                "old_success": old_success,
                "new_success": new_success,
                "old_verifier": old_verifier,
                "new_verifier": new_verifier,
                "reverification_mode": mode,
                "answer_type": answer_type,
                "verifier_selected": verifier_selected,
                "math_verify_available": math_verify_available,
                "unsupported": unsupported,
                "prediction_extract": result.metadata.get("extracted_prediction") or result.details.get("extracted"),
                "gold_extract": result.metadata.get("extracted_gold") or result.details.get("gold"),
            }
            if unsupported:
                unsupported_rows.append(
                    {
                        **correction,
                        "error": result.details.get("error"),
                        "task_aware_policy": result.metadata.get("task_aware_policy", ""),
                    }
                )
            rows.append(
                {
                    "suite": run.suite or "",
                    "run_id": run.run_id,
                    "model": run.model,
                    "source": source,
                    "task_id": task_id,
                    "budget": outcome.get("budget"),
                    "recorded_success": old_success,
                    "reverified_success": new_success,
                    "changed": changed,
                    "unsupported": unsupported,
                    "mode": mode,
                    "answer_type": answer_type,
                    "verifier_selected": verifier_selected,
                    "verifier_policy": result.metadata.get("task_aware_policy", verifier_selected),
                    "math_verify_available": math_verify_available,
                    "extracted": correction["prediction_extract"],
                    "gold": correction["gold_extract"],
                    "artifact_source": run.artifact_source,
                }
            )
            if changed or unsupported:
                correction_rows.append(correction)
            group = summary_groups[(run.suite or "", run.run_id, run.model, source)]
            group["n_rows"] += 1
            group["verifier_policy"] = result.metadata.get("task_aware_policy", verifier_selected)
            group["math_verify_available"] = bool(group["math_verify_available"] or math_verify_available)
            if unsupported:
                group["unsupported_rows"] += 1
            else:
                group["supported_rows"] += 1
                group["n_changed"] += 1 if changed else 0
                group["old_success"] += 1 if old_success else 0
                group["new_success"] += 1 if new_success else 0
            corrected = dict(outcome)
            corrected["success"] = bool(new_success)
            verification = dict(corrected.get("verification") or {})
            verification["status"] = result.status.value if hasattr(result.status, "value") else str(result.status)
            verification["success"] = bool(new_success)
            verification["details"] = result.details
            verification["metadata"] = result.metadata
            corrected["verification"] = verification
            metadata = dict(corrected.get("metadata") or {})
            metadata["label_mode"] = mode
            metadata["verifier_mode"] = result.metadata.get("verifier_mode", new_verifier)
            metadata["verifier_selected"] = verifier_selected
            metadata["verifier_policy"] = result.metadata.get("task_aware_policy", verifier_selected)
            metadata["answer_type"] = answer_type
            metadata["math_verify_available"] = math_verify_available
            metadata["unsupported_reverification"] = unsupported
            corrected["metadata"] = metadata
            corrected_rows.append(corrected)
        if write_corrected_outcomes and run_has_math:
            corrected_dir = Path(write_corrected_outcomes) / run.run_dir.name
            _copy_run_shell(run.run_dir, corrected_dir)
            write_jsonl(corrected_dir / "outcomes.jsonl", corrected_rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    _write_csv(Path(summary_output), _summary_rows(summary_groups, mode))
    _write_csv(Path(unsupported_output), unsupported_rows)
    if write_corrections is not None:
        write_jsonl(write_corrections, correction_rows)
    return output


def _verifier_for_mode(mode: str):
    if mode in {"strict", "lenient"}:
        return NumericExactVerifier(mode=mode)
    if mode == "math_verify_optional":
        return MathVerifyOptionalVerifier()
    if mode == "task_aware_strict":
        return TaskAwareMathVerifier(require_math_verify_for_symbolic=True)
    raise ValueError("mode must be strict, lenient, math_verify_optional, task_default, or task_aware_strict")


def _verifier_for_task(task: TaskRecord, mode: str, cache: dict[str, Any]):
    if mode != "task_default":
        key = mode
        if key not in cache:
            cache[key] = _verifier_for_mode(mode)
        return cache[key], _mode_label(mode)
    source = (task.source or "").lower()
    verifier_name = (task.verifier or "").lower()
    if source == "gsm8k":
        selected = "numeric_exact_strict"
        cache_key = "task_default:gsm8k"
        if cache_key not in cache:
            cache[cache_key] = NumericExactVerifier(mode="strict")
        return cache[cache_key], selected
    if source in {"hendrycks_math", "math", "math500"} or verifier_name == "math_verify_optional":
        selected = "math_verify_optional"
        cache_key = "task_default:math_verify_optional"
        if cache_key not in cache:
            cache[cache_key] = MathVerifyOptionalVerifier()
        return cache[cache_key], selected
    selected = task.verifier
    cache_key = f"task_default:{selected}"
    if cache_key not in cache:
        cache[cache_key] = get_verifier(selected)
    return cache[cache_key], selected


def _mode_label(mode: str) -> str:
    if mode == "strict":
        return "numeric_exact_strict"
    if mode == "task_aware_strict":
        return "task_aware_strict_math"
    return mode


def _old_verifier(outcome: dict[str, Any]) -> str:
    verification = outcome.get("verification") or {}
    details = verification.get("details") or {}
    metadata = verification.get("metadata") or {}
    row_metadata = outcome.get("metadata") or {}
    return str(
        metadata.get("verifier_mode")
        or row_metadata.get("verifier_mode")
        or details.get("mode")
        or ("math_verify_optional" if details.get("math_verify_available") is not None else "unknown")
    )


def _summary_rows(groups: dict[tuple[str, str, str, str], dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (suite, run_id, model, source), values in sorted(groups.items()):
        n_rows = int(values["n_rows"])
        supported_rows = int(values.get("supported_rows", n_rows))
        unsupported_rows = int(values.get("unsupported_rows", 0))
        old_rate = values["old_success"] / supported_rows if supported_rows else 0.0
        new_rate = values["new_success"] / supported_rows if supported_rows else 0.0
        rows.append(
            {
                "suite": suite,
                "run_id": run_id,
                "model": model,
                "source": source,
                "n_rows": n_rows,
                "supported_rows": supported_rows,
                "unsupported_rows": unsupported_rows,
                "n_changed": int(values["n_changed"]),
                "change_rate": int(values["n_changed"]) / supported_rows if supported_rows else 0.0,
                "old_success_rate": old_rate,
                "new_success_rate": new_rate,
                "success_delta": new_rate - old_rate,
                "verifier_mode": mode,
                "verifier_policy": values.get("verifier_policy", ""),
                "math_verify_available": bool(values.get("math_verify_available", False)),
            }
        )
    return rows


def _task_files_from_runs(runs: list[Any]) -> list[Path]:
    paths: list[Path] = []
    for run in runs:
        task_file = run.config.get("task_file")
        if task_file and Path(task_file).exists():
            paths.append(Path(task_file))
    fallback = Path("data/processed/paper_math_core.jsonl")
    if fallback.exists():
        paths.append(fallback)
    return sorted(set(paths))


def _copy_run_shell(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name == "outcomes.jsonl":
            continue
        destination = target / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        elif path.is_file():
            shutil.copy2(path, destination)


def _load_tasks(task_files: list[str | Path]) -> dict[str, TaskRecord]:
    tasks: dict[str, TaskRecord] = {}
    for task_file in task_files:
        path = Path(task_file)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task = TaskRecord.model_validate(row)
            tasks[task.task_id] = task
    return tasks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverify math outcomes with task-aware final-answer verifiers.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--task-files", nargs="+", default=None)
    parser.add_argument("--output", default="reports/tables/math_reverification_audit.csv")
    parser.add_argument("--summary-output", default="reports/tables/math_verifier_delta_summary.csv")
    parser.add_argument("--unsupported-output", default="reports/tables/math_reverification_unsupported.csv")
    parser.add_argument("--write-corrections", default=None)
    parser.add_argument("--write-corrected-outcomes", default=None)
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient", "math_verify_optional", "task_default", "task_aware_strict"],
        default="strict",
    )
    args = parser.parse_args()
    path = reverify_outcomes(
        artifact_root=args.artifact_root,
        task_files=args.task_files,
        output=args.output,
        summary_output=args.summary_output,
        unsupported_output=args.unsupported_output,
        write_corrections=args.write_corrections,
        write_corrected_outcomes=args.write_corrected_outcomes,
        mode=args.mode,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
