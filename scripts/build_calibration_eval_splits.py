#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_paper_runs


def stable_split(suite: str, task_id: str, *, seed: int, calibration_frac: float) -> str:
    value = int(hashlib.sha256(f"{suite}:{task_id}:{seed}".encode()).hexdigest(), 16) / 2**256
    return "calibration" if value < calibration_frac else "evaluation"


def build_calibration_eval_splits(
    *,
    artifact_root: str | Path = "reports/artifacts",
    output_dir: str | Path = "reports/splits",
    summary_path: str | Path = "reports/tables/calibration_eval_split_summary.csv",
    calibration_frac: float = 0.30,
    seed: int = 20260428,
    suites: Iterable[str] | None = None,
) -> list[Path]:
    if not 0.0 < calibration_frac < 1.0:
        raise ValueError("calibration_frac must be between 0 and 1")
    requested_suites = [suite for suite in suites or [] if suite]
    if requested_suites:
        runs = []
        for suite in requested_suites:
            runs.extend(
                load_paper_runs(
                    suite=suite,
                    run_root=Path(artifact_root) / "__no_reports_runs__",
                    artifact_root=artifact_root,
                    include_artifacts=True,
                )
            )
    else:
        runs = load_paper_runs(
            run_root=Path(artifact_root) / "__no_reports_runs__",
            artifact_root=artifact_root,
            include_artifacts=True,
        )
    task_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for run in runs:
        suite = run.suite or ""
        if not suite:
            continue
        for row in list(run.forecasts) + list(run.outcomes):
            task_id = row.get("task_id")
            if task_id is None:
                continue
            source = str((row.get("metadata") or {}).get("source") or row.get("source") or "unknown")
            task_sources[suite][str(task_id)].add(source)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for suite, by_task in sorted(task_sources.items()):
        task_splits = {
            task_id: stable_split(suite, task_id, seed=seed, calibration_frac=calibration_frac)
            for task_id in sorted(by_task)
        }
        counts = {
            "calibration": sum(1 for split in task_splits.values() if split == "calibration"),
            "evaluation": sum(1 for split in task_splits.values() if split == "evaluation"),
        }
        path = output_dir / f"{suite}_calibration_eval_split.json"
        path.write_text(
            json.dumps(
                {
                    "suite": suite,
                    "seed": seed,
                    "calibration_frac": calibration_frac,
                    "counts": counts,
                    "task_splits": task_splits,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.append(path)
        by_source: dict[str, list[str]] = defaultdict(list)
        for task_id, sources in by_task.items():
            for source in sources or {"unknown"}:
                by_source[source].append(task_id)
        for source, task_ids in sorted(by_source.items()):
            summary_rows.append(
                {
                    "suite": suite,
                    "source": source,
                    "n_tasks": len(set(task_ids)),
                    "n_calibration": sum(1 for task_id in set(task_ids) if task_splits[task_id] == "calibration"),
                    "n_evaluation": sum(1 for task_id in set(task_ids) if task_splits[task_id] == "evaluation"),
                    "seed": seed,
                    "calibration_frac": calibration_frac,
                }
            )
    _write_csv(Path(summary_path), summary_rows)
    return outputs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic calibration/evaluation task splits.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--output-dir", default="reports/splits")
    parser.add_argument("--summary-path", default="reports/tables/calibration_eval_split_summary.csv")
    parser.add_argument("--calibration-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--suite", action="append", default=[], help="Suite to include; repeatable.")
    args = parser.parse_args()
    outputs = build_calibration_eval_splits(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        calibration_frac=args.calibration_frac,
        seed=args.seed,
        suites=args.suite,
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
