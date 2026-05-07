#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.execution.bigcodebench_bridge import BigCodeBenchBridge
from budget2success.execution.external_harness import run_command
from budget2success.schemas.records import TaskRecord
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl, write_jsonl
from budget2success.utils.manifest import sha256_file, write_redacted_config_snapshot, write_run_manifest


STATUS_FIELDS = ["model", "budget", "status", "error_message", "predictions_path", "result_path"]


def run_bigcodebench_official(
    *,
    task_file: str | Path,
    run_root: str | Path,
    output_root: str | Path,
    command_template: str,
    timeout_seconds: float,
) -> tuple[Path, bool]:
    task_file = Path(task_file)
    run_root = Path(run_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [TaskRecord.model_validate(row) for row in read_jsonl(task_file)]
    status_rows: list[dict[str, Any]] = []
    any_completed = False

    for run_dir in _discover_run_dirs(run_root):
        outcomes_path = run_dir / "outcomes.jsonl"
        forecasts_path = run_dir / "forecasts.jsonl"
        config_path = run_dir / "config_snapshot.yaml"
        if not outcomes_path.exists() or not forecasts_path.exists() or not config_path.exists():
            continue
        config = load_yaml(config_path)
        model = str(config.get("model") or run_dir.name)
        model_dir = output_root / _safe_model_slug(run_dir.name)
        model_dir.mkdir(parents=True, exist_ok=True)
        outcomes = read_jsonl(outcomes_path)
        grouped = _group_outcomes_by_budget(outcomes)
        merged_outcomes: list[dict[str, Any]] = []
        completed_budgets = 0
        for budget, budget_outcomes in sorted(grouped.items()):
            budget_dir = model_dir / f"budget_{budget}"
            budget_dir.mkdir(parents=True, exist_ok=True)
            bridge = BigCodeBenchBridge(budget_dir)
            predictions_path = bridge.write_predictions_from_records(
                tasks,
                budget_outcomes,
                filename=f"bigcodebench_predictions_budget_{budget}.jsonl",
            )
            try:
                if command_template == "official-package-direct":
                    result_path = _evaluate_predictions_direct(predictions_path, budget_dir)
                    labels = bridge.parse_official_results(result_path)
                    harness_error = ""
                else:
                    command = _render_command(command_template, predictions_path=predictions_path, output_dir=budget_dir)
                    harness = run_command(command, cwd=budget_dir, timeout_seconds=timeout_seconds)
                    (budget_dir / "stdout.txt").write_text(harness.stdout, encoding="utf-8")
                    (budget_dir / "stderr.txt").write_text(harness.stderr, encoding="utf-8")
                    try:
                        labels, result_path = _parse_labels(bridge, budget_dir, harness.stdout)
                    except Exception:
                        result_path = _evaluate_predictions_direct(predictions_path, budget_dir)
                        labels = bridge.parse_official_results(result_path)
                    harness_error = "" if harness.success else f"evaluator_returncode={harness.returncode}"
                merged = bridge.merge_official_results_into_outcomes(budget_outcomes, tasks, labels)
            except Exception as exc:
                status_rows.append(
                    {
                        "model": model,
                        "budget": budget,
                        "status": "official_labels_absent",
                        "error_message": str(exc),
                        "predictions_path": str(predictions_path),
                        "result_path": "",
                    }
                )
                continue
            write_jsonl(budget_dir / "official_labeled_outcomes.jsonl", merged)
            merged_outcomes.extend(merged)
            completed_budgets += 1
            status_rows.append(
                {
                    "model": model,
                    "budget": budget,
                    "status": "official_labels_completed",
                    "error_message": harness_error,
                    "predictions_path": str(predictions_path),
                    "result_path": str(result_path),
                }
            )
        if completed_budgets != len(grouped):
            continue
        _write_model_artifact(
            model_dir=model_dir,
            source_run_dir=run_dir,
            config=config,
            forecasts_path=forecasts_path,
            merged_outcomes=merged_outcomes,
            task_file=task_file,
        )
        any_completed = True

    status_path = Path("reports/tables/bigcodebench_official_status.csv")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(status_rows)
    return status_path, any_completed


def _discover_run_dirs(run_root: Path) -> list[Path]:
    if (run_root / "outcomes.jsonl").exists():
        return [run_root]
    if not run_root.exists():
        return []
    return [path for path in sorted(run_root.iterdir()) if path.is_dir()]


def _group_outcomes_by_budget(outcomes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if row.get("budget") is not None:
            grouped[int(row["budget"])].append(row)
    return dict(grouped)


def _render_command(command_template: str, *, predictions_path: Path, output_dir: Path) -> list[str]:
    rendered = command_template.format(predictions=predictions_path, output_dir=output_dir)
    return shlex.split(rendered)


def _parse_labels(bridge: BigCodeBenchBridge, budget_dir: Path, stdout: str) -> tuple[dict[str, bool], Path]:
    candidates = [
        path
        for path in sorted(budget_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and "prediction" not in path.name.lower()
        and path.name != "official_labeled_outcomes.jsonl"
    ]
    for candidate in candidates:
        try:
            return bridge.parse_official_results(candidate), candidate
        except Exception:
            continue
    for index, line in enumerate(reversed(stdout.splitlines()), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        path = budget_dir / f"stdout_result_{index}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        try:
            return bridge.parse_official_results(path), path
        except Exception:
            continue
    return bridge.parse_official_results(budget_dir), budget_dir


def _evaluate_predictions_direct(predictions_path: Path, budget_dir: Path) -> Path:
    from bigcodebench.data import get_bigcodebench  # type: ignore[import-not-found]
    from bigcodebench.eval import PASS, untrusted_check  # type: ignore[import-not-found]

    problems = dict(get_bigcodebench(subset="hard"))
    rows = []
    for sample in read_jsonl(predictions_path):
        task_id = str(sample.get("task_id") or "")
        problem = problems.get(task_id)
        if problem is None:
            rows.append({"task_id": task_id, "passed": False, "status": "missing_problem"})
            continue
        try:
            status, details = untrusted_check(
                str(sample.get("solution") or ""),
                problem["test"],
                problem["entry_point"],
                30 * 1024,
                30 * 1024,
                10,
                1,
                20,
            )
            rows.append({"task_id": task_id, "passed": status == PASS, "status": str(status), "details": _json_safe(details)})
        except Exception as exc:  # noqa: BLE001 - keep per-sample official evaluator errors parseable.
            rows.append({"task_id": task_id, "passed": False, "status": "error", "error": str(exc)})
    result_path = budget_dir / "official_package_direct_results.jsonl"
    write_jsonl(result_path, rows)
    return result_path


def _write_model_artifact(
    *,
    model_dir: Path,
    source_run_dir: Path,
    config: dict[str, Any],
    forecasts_path: Path,
    merged_outcomes: list[dict[str, Any]],
    task_file: Path,
) -> None:
    official_config = {
        **config,
        "run_id": model_dir.name,
        "output_dir": str(model_dir.parent),
        "task_file": str(task_file),
        "metadata": {
            **(config.get("metadata") or {}),
            "official_harness_status": "official_labels_completed",
            "official_label_source": "bigcodebench",
            "source_run_dir": str(source_run_dir),
        },
    }
    write_redacted_config_snapshot(official_config, model_dir / "config_snapshot.yaml")
    (model_dir / "forecasts.jsonl").write_text(forecasts_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_jsonl(model_dir / "outcomes.jsonl", merged_outcomes)
    subprocess.run(
        [sys.executable, "scripts/score_results.py", "--config", str(model_dir / "config_snapshot.yaml")],
        check=True,
    )
    write_run_manifest(
        model_dir,
        config=official_config,
        command_line_arguments=sys.argv[1:],
        phase="bigcodebench_official_labels",
        extra={"source_run_dir": str(source_run_dir), "task_file": str(task_file)},
    )
    _write_sha_manifest(model_dir)


def _write_sha_manifest(artifact_dir: Path) -> Path:
    rows = {}
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[str(path.relative_to(artifact_dir))] = sha256_file(path)
    out_path = artifact_dir / "sha256_manifest.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)


def _safe_model_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "model"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official BigCodeBench evaluation and merge labels into outcomes.")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--command-template",
        default=f"{sys.executable} -m bigcodebench.evaluate --samples {{predictions}} --out_dir {{output_dir}}",
    )
    args = parser.parse_args()
    status_path, ok = run_bigcodebench_official(
        task_file=args.task_file,
        run_root=args.run_root,
        output_root=args.output_root,
        command_template=args.command_template,
        timeout_seconds=args.timeout_seconds,
    )
    print(status_path)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
