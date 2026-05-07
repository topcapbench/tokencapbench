#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_paper_runs
from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.utils.config import load_yaml


def solver_prompt_contains_forecast(forecast: dict[str, Any], solver_prompt: str) -> bool:
    text = solver_prompt.lower()
    probes = []
    curve = forecast.get("p_success_by_budget") or {}
    if curve:
        probes.append(json.dumps(curve, sort_keys=True))
        probes.append(str(curve))
    if forecast.get("short_rationale"):
        probes.append(str(forecast["short_rationale"]))
    if forecast.get("raw_text"):
        probes.append(str(forecast["raw_text"])[:120])
    return any(probe and probe.lower() in text for probe in probes)


def audit_forecast_leakage(
    *,
    suite: str | None = None,
    run_dir: str | Path | None = None,
    sample_size: int = 30,
    seed: int = 0,
    output: str | Path = "reports/tables/forecast_leakage_audit.csv",
) -> Path:
    runs = load_paper_runs(suite=suite, run_dirs=[run_dir] if run_dir else None)
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for run in runs:
        run_suite = suite if suite is not None else run.suite or ""
        task_prompts = _task_prompts(run.config)
        forecasts = [row for row in run.forecasts if "p_success_by_budget" in row]
        forecasts = rng.sample(forecasts, min(sample_size, len(forecasts))) if forecasts else []
        for forecast in forecasts:
            task_id = str(forecast["task_id"])
            solver_prompt = _build_solver_prompt(run.config, task_prompts.get(task_id, ""), forecast)
            leaked = solver_prompt_contains_forecast(forecast, solver_prompt)
            rows.append(
                {
                    "suite": run_suite,
                    "run_id": run.run_id,
                    "model": run.model,
                    "task_id": task_id,
                    "forecast_prompt_hash_present": bool((run.config.get("forecast_prompt"))),
                    "solver_prompt_length_chars": len(solver_prompt),
                    "forecast_probability_strings_in_solver": leaked,
                    "rationale_in_solver": bool(
                        forecast.get("short_rationale") and str(forecast["short_rationale"]).lower() in solver_prompt.lower()
                    ),
                    "status": "FAIL" if leaked else "PASS",
                }
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    return output


def _task_prompts(config: dict[str, Any]) -> dict[str, str]:
    task_file = config.get("task_file")
    if not task_file or not Path(task_file).exists():
        return {}
    return {task.task_id: task.prompt for task in load_tasks_jsonl(task_file)}


def _build_solver_prompt(config: dict[str, Any], task_prompt: str, forecast: dict[str, Any]) -> str:
    track = (forecast.get("metadata") or {}).get("track") or "default"
    prompt_path = (config.get("solver_prompts") or {}).get(track)
    if prompt_path and Path(prompt_path).exists():
        template = Path(prompt_path).read_text(encoding="utf-8")
        return f"{template}\n\nTask:\n{task_prompt}\n"
    return task_prompt


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit forecast-to-solver prompt leakage.")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/tables/forecast_leakage_audit.csv")
    args = parser.parse_args()
    path = audit_forecast_leakage(
        suite=args.suite,
        run_dir=args.run_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        output=args.output,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
