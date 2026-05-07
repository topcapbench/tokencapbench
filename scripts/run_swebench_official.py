#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import forecast_curves, outcomes_by_task, score_curve_set
from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.execution.swebench_bridge import SWEBenchBridge
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl, write_jsonl
from budget2success.utils.manifest import sha256_file


COPY_NAMES = (
    "forecasts.jsonl",
    "config_snapshot.yaml",
    "source_config_snapshot.yaml",
    "task_file_hash.json",
    "run_manifest.json",
    "sha256_manifest.json",
)
REPORT_KEYS = {"resolved", "resolved_ids", "instance_id_to_report", "report", "successful_tasks"}


def run_swebench_official(
    *,
    config: str | Path,
    run_root: str | Path,
    output_dir: str | Path,
    corrected_artifact_root: str | Path,
    dataset_name: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    timeout_seconds: float | None = 7200.0,
) -> list[Path]:
    cfg = load_yaml(config)
    tasks = load_tasks_jsonl(str(cfg["task_file"]))
    run_dirs = _discover_run_dirs(run_root)
    output_root = Path(output_dir)
    corrected_root = Path(corrected_artifact_root)
    corrected_dirs: list[Path] = []
    for run_dir in run_dirs:
        corrected_dirs.append(
            _run_one(
                tasks=tasks,
                run_dir=run_dir,
                output_root=output_root,
                corrected_root=corrected_root,
                dataset_name=dataset_name,
                split=split,
                timeout_seconds=timeout_seconds,
            )
        )
    return corrected_dirs


def _run_one(
    *,
    tasks,
    run_dir: Path,
    output_root: Path,
    corrected_root: Path,
    dataset_name: str,
    split: str,
    timeout_seconds: float | None,
) -> Path:
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")
    forecasts = read_jsonl(run_dir / "forecasts.jsonl") if (run_dir / "forecasts.jsonl").exists() else []
    task_by_id = {task.task_id: task for task in tasks}
    run_output_dir = output_root / run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    bridge = SWEBenchBridge(run_output_dir)
    labels_by_budget: dict[int, dict[str, bool]] = {}
    harness_results: dict[str, Any] = {}
    for budget in sorted({int(row["budget"]) for row in outcomes if row.get("budget") is not None}):
        budget_outcomes = [row for row in outcomes if int(row.get("budget") or 0) == budget]
        predictions = []
        model = _first_value(budget_outcomes, "model") or run_dir.name
        for row in budget_outcomes:
            task = task_by_id.get(str(row.get("task_id") or ""))
            if task is None:
                continue
            predictions.append(
                {
                    "instance_id": task.external_id or task.task_id,
                    "model_name_or_path": f"{model}-budget-{budget}",
                    "model_patch": str(row.get("solution") or ""),
                }
            )
        prediction_path = bridge.write_predictions(predictions, filename=f"{run_dir.name}__budget_{budget}_predictions.jsonl")
        official_run_id = f"{run_dir.name}_budget_{budget}"
        result = bridge.run_evaluation(
            prediction_path,
            dataset_name=dataset_name,
            split=split,
            run_id=official_run_id,
            timeout_seconds=timeout_seconds,
        )
        result_record: dict[str, Any] = {
            "prediction_path": str(prediction_path),
            "returncode": result.returncode,
            "success": result.success,
            "timed_out": result.timed_out,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        (run_output_dir / f"{official_run_id}_harness_result.json").write_text(
            json.dumps(result_record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_path = find_swebench_report(run_output_dir, official_run_id)
        if report_path is None:
            report_path = find_swebench_report(Path("logs"), official_run_id)
        if report_path is not None:
            result_record["report_path"] = str(report_path)
            parsed = parse_swebench_report(report_path)
            labels_by_budget[budget] = parsed
            result_record["parsed_labels"] = len(parsed)
        else:
            labels_by_budget[budget] = {}
            result_record["parsed_labels"] = 0
            result_record["parse_error"] = "official report not found"
        harness_results[str(budget)] = result_record

    corrected_outcomes = _merge_labels(outcomes, tasks, labels_by_budget, harness_results)
    corrected_dir = corrected_root / run_dir.name
    corrected_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(corrected_dir / "outcomes.jsonl", corrected_outcomes)
    _copy_run_context(run_dir, corrected_dir)
    if not (corrected_dir / "forecasts.jsonl").exists() and forecasts:
        write_jsonl(corrected_dir / "forecasts.jsonl", forecasts)
    metrics = _metrics(forecasts, corrected_outcomes, harness_results)
    (corrected_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (corrected_dir / "official_swebench_results.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run_dir": str(run_dir),
                "official_output_dir": str(run_output_dir),
                "harness_results": harness_results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (corrected_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run_dir": str(run_dir),
                "corrected_dir": str(corrected_dir),
                "label_source": "official_swebench",
                "official_labels_available": any(bool(labels) for labels in labels_by_budget.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_sha_manifests(corrected_dir)
    return corrected_dir


def find_swebench_report(output_dir: Path, run_id: str) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.rglob("*.json")
        if path.is_file() and (run_id in str(path) or _json_contains_report_key(path))
    ]
    candidates = [path for path in candidates if _json_contains_report_key(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_swebench_report(report_path: Path) -> dict[str, bool]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    labels: dict[str, bool] = {}
    _collect_labels(payload, labels)
    return labels


def _collect_labels(value: Any, labels: dict[str, bool], instance_id: str | None = None) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("resolved_ids"), list):
            for item in value["resolved_ids"]:
                labels[str(item)] = True
        if isinstance(value.get("successful_tasks"), list):
            for item in value["successful_tasks"]:
                labels[str(item)] = True
        if isinstance(value.get("resolved"), list):
            for item in value["resolved"]:
                labels[str(item)] = True
        if instance_id and isinstance(value.get("resolved"), bool):
            labels[instance_id] = bool(value["resolved"])
        for container_key in ("instance_id_to_report", "report"):
            container = value.get(container_key)
            if isinstance(container, dict):
                for key, child in container.items():
                    child_instance = str(key)
                    if isinstance(child, bool):
                        labels[child_instance] = child
                    else:
                        _collect_labels(child, labels, child_instance)
            elif isinstance(container, list):
                for child in container:
                    _collect_labels(child, labels, None)
        row_id = value.get("instance_id")
        if row_id is not None:
            for key in ("resolved", "success", "passed"):
                if isinstance(value.get(key), bool):
                    labels[str(row_id)] = bool(value[key])
        for key, child in value.items():
            if key in {"instance_id_to_report", "report"}:
                continue
            if isinstance(child, (dict, list)):
                child_instance = str(key) if isinstance(child, dict) and "__" in str(key) else None
                _collect_labels(child, labels, child_instance)
    elif isinstance(value, list):
        for child in value:
            _collect_labels(child, labels, instance_id)


def _merge_labels(
    outcomes: list[dict[str, Any]],
    tasks,
    labels_by_budget: dict[int, dict[str, bool]],
    harness_results: dict[str, Any],
) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in tasks}
    merged: list[dict[str, Any]] = []
    for row in outcomes:
        budget = int(row.get("budget") or 0)
        task = task_by_id.get(str(row.get("task_id") or ""))
        instance_id = task.external_id if task else str(row.get("task_id") or "")
        labels = labels_by_budget.get(budget, {})
        if instance_id in labels:
            success = bool(labels[instance_id])
            verification = {
                "status": "success" if success else "failure",
                "success": success,
                "details": {"instance_id": instance_id, "budget": budget},
                "metadata": {"label_source": "official_swebench"},
            }
            label_source = "official_swebench"
            official_status = "completed"
        else:
            success = False
            budget_result = harness_results.get(str(budget), {})
            verification = {
                "status": "error",
                "success": False,
                "details": {
                    "error": "official_swebench_label_unavailable",
                    "instance_id": instance_id,
                    "budget": budget,
                    "harness_returncode": budget_result.get("returncode"),
                    "parse_error": budget_result.get("parse_error", ""),
                },
                "metadata": {"label_source": "official_swebench_error"},
            }
            label_source = "official_swebench_error"
            official_status = "label_unavailable"
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "label_source": label_source,
                "official_harness": "swebench",
                "official_harness_status": official_status,
                "external_id": instance_id,
            }
        )
        merged.append({**row, "success": success, "verification": verification, "metadata": metadata})
    return merged


def _metrics(forecasts: list[dict[str, Any]], outcomes: list[dict[str, Any]], harness_results: dict[str, Any]) -> dict[str, Any]:
    scored = score_curve_set(forecast_curves(forecasts), outcomes_by_task(outcomes), outcome_rows=outcomes) if forecasts and outcomes else {}
    return {
        "official_harness": "swebench",
        "official_harness_status": "completed" if all(item.get("success") for item in harness_results.values()) else "failed_or_incomplete",
        "official_labels": sum(
            1 for row in outcomes if (row.get("metadata") or {}).get("label_source") == "official_swebench"
        ),
        "n_outcomes": len(outcomes),
        "n_success": sum(1 for row in outcomes if row.get("success")),
        "score": scored,
    }


def _json_contains_report_key(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return _contains_report_key(payload)


def _contains_report_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in REPORT_KEYS):
            return True
        return any(_contains_report_key(child) for child in value.values() if isinstance(child, (dict, list)))
    if isinstance(value, list):
        return any(_contains_report_key(child) for child in value)
    return False


def _copy_run_context(run_dir: Path, corrected_dir: Path) -> None:
    for name in COPY_NAMES:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, corrected_dir / name)
    prompts = run_dir / "prompts"
    if prompts.exists():
        target = corrected_dir / "prompts"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(prompts, target)


def _write_sha_manifests(directory: Path) -> None:
    files = [path for path in sorted(directory.rglob("*")) if path.is_file() and path.name not in {"sha256_manifest.json", "sha256_manifest.txt"}]
    payload = {str(path.relative_to(directory)): sha256_file(path) for path in files}
    (directory / "sha256_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (directory / "sha256_manifest.txt").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in sorted(payload.items())) + "\n",
        encoding="utf-8",
    )


def _discover_run_dirs(run_root: str | Path) -> list[Path]:
    root = Path(run_root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "outcomes.jsonl").exists())


def _first_value(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        if row.get(key) not in {"", None}:
            return row.get(key)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SWE-bench patches, run the official harness, and merge labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrected-artifact-root", required=True)
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    corrected = run_swebench_official(
        config=args.config,
        run_root=args.run_root,
        output_dir=args.output_dir,
        corrected_artifact_root=args.corrected_artifact_root,
        dataset_name=args.dataset_name,
        split=args.split,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"corrected_artifacts": [str(path) for path in corrected]}, indent=2))


if __name__ == "__main__":
    main()
