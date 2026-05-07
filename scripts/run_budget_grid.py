#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from budget2success.clients.factory import build_client
from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.execution.runner import run_task_under_budget
from budget2success.execution.verifier_registry import get_verifier
from budget2success.schemas.records import BudgetRunRecord, ExperimentConfig, VerificationResult
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import append_jsonl, read_jsonl
from budget2success.utils.manifest import write_redacted_config_snapshot, write_run_manifest
try:
    from estimate_experiment_cost import estimate_experiment_cost
    from run_experiment_suite import suite_model_configs
except ImportError:  # pragma: no cover - package import path used by tests.
    from scripts.estimate_experiment_cost import estimate_experiment_cost
    from scripts.run_experiment_suite import suite_model_configs

def build_solver_prompt(template_path: str | None, task) -> str:
    if template_path and Path(template_path).exists():
        template = Path(template_path).read_text(encoding="utf-8")
        rendered = template.replace("{{ task.prompt }}", task.prompt)
        for key, value in (task.metadata or {}).items():
            rendered = rendered.replace(f"{{{{ task.metadata.{key} }}}}", "" if value is None else str(value))
        if rendered != template:
            return rendered
        return f"{template}\n\nTask:\n{task.prompt}\n"
    return task.prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run solver contexts under token-budget grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Archive an existing outcomes.jsonl before writing a fresh one.")
    parser.add_argument("--resume", action="store_true", help="Append only missing task/budget outcomes when outcomes.jsonl exists.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent solver calls.")
    parser.add_argument("--pricing", default="reports/live_runs/provider_live_cost_estimate.json")
    parser.add_argument("--cap-usd", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, help="Override configured models for suite configs.")
    args = parser.parse_args()
    raw_config = load_yaml(args.config)
    if _is_suite_config(raw_config):
        _run_suite_budget_grid(args.config, raw_config, args)
        return
    cfg = ExperimentConfig.model_validate(raw_config)
    run_dir = Path(cfg.output_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_redacted_config_snapshot(raw_config, run_dir / "config_snapshot.yaml")
    out_path = run_dir / "outcomes.jsonl"
    repeat_ids = _repeat_ids(raw_config)
    completed_pairs: set[tuple[str, int, str | None]] = set()
    if out_path.exists():
        if args.resume:
            completed_pairs = {
                (str(row.get("task_id")), int(row.get("budget")), _row_repeat_id(row))
                for row in read_jsonl(out_path)
                if row.get("task_id") is not None and row.get("budget") is not None
            }
        elif not args.overwrite:
            raise FileExistsError(f"{out_path} already exists. Use --overwrite to archive and replace it.")
        else:
            archive_path = out_path.with_name(
                f"{out_path.name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
            )
            out_path.rename(archive_path)

    client = build_client(raw_config)
    tasks = load_tasks_jsonl(cfg.task_file)
    if cfg.limit:
        tasks = tasks[: cfg.limit]

    jobs = []
    for task_index, task in enumerate(tasks, start=1):
        budget_grid = task.budget_grid or cfg.budget_grid.get(task.track) or cfg.budget_grid.get("default")
        if not budget_grid:
            raise ValueError(f"No budget grid for task {task.task_id} track {task.track}")
        solver_template = cfg.solver_prompts.get(task.track)
        prompt = build_solver_prompt(solver_template, task)
        for budget in budget_grid:
            for repeat_id in repeat_ids:
                if (task.task_id, int(budget), repeat_id) in completed_pairs:
                    continue
                jobs.append((task_index, task, prompt, int(budget), repeat_id))

    total_calls = len(jobs) + len(completed_pairs)
    completed_calls = len(completed_pairs)
    if completed_calls:
        print(
            f"[run_budget_grid] {cfg.run_id}: resuming after {completed_calls}/{total_calls} calls",
            file=sys.stderr,
            flush=True,
        )

    def run_one(job):
        task_index, task, prompt, budget, repeat_id = job
        verifier_name = "record_only" if _defer_verification(raw_config) else task.verifier
        verifier = get_verifier(verifier_name)
        try:
            result = run_task_under_budget(
                client=client,
                verifier=verifier,
                task=task,
                model=cfg.model,
                prompt=prompt,
                budget=budget,
                temperature=cfg.temperature,
                scaffold=cfg.scaffold,
            )
        except Exception as exc:  # noqa: BLE001 - preserve API/transport failures as raw outcomes.
            result = BudgetRunRecord(
                task_id=task.task_id,
                model=cfg.model,
                scaffold=cfg.scaffold,
                budget=budget,
                solution="",
                success=False,
                verification=VerificationResult.error(error="generation_exception", message=str(exc)),
                wall_time_seconds=None,
                metadata={
                    "track": task.track,
                    "source": task.source,
                    "source_version": task.source_version,
                    "external_id": task.external_id,
                },
            )
        if repeat_id is not None:
            result.metadata = {**result.metadata, "repeat_id": repeat_id}
        return task_index, result

    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            task_index, result = run_one(job)
            append_jsonl(out_path, result)
            completed_calls += 1
            if completed_calls == 1 or completed_calls % 50 == 0 or completed_calls == total_calls:
                print(
                    f"[run_budget_grid] {cfg.run_id}: task {task_index}/{len(tasks)}, call {completed_calls}/{total_calls}",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, job) for job in jobs]
            for future in as_completed(futures):
                task_index, result = future.result()
                append_jsonl(out_path, result)
                completed_calls += 1
                if completed_calls == 1 or completed_calls % 50 == 0 or completed_calls == total_calls:
                    print(
                        f"[run_budget_grid] {cfg.run_id}: task {task_index}/{len(tasks)}, call {completed_calls}/{total_calls}",
                        file=sys.stderr,
                        flush=True,
                    )
    write_run_manifest(
        run_dir,
        config=raw_config,
        command_line_arguments=sys.argv[1:],
        phase="outcomes",
        extra={"outcome_file": str(out_path), "repeat_ids": repeat_ids, "repeats": len(repeat_ids)},
    )
    print(json.dumps({"outcomes": str(out_path), "n_tasks": len(tasks)}, indent=2))


def _repeat_ids(config: dict) -> list[str | None]:
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    raw_ids = metadata.get("repeat_ids")
    if raw_ids:
        return [str(value) for value in raw_ids]
    repeats_config = metadata.get("repeats") or config.get("repeats") or 1
    if isinstance(repeats_config, dict):
        forecast_repeats = max(1, int(repeats_config.get("forecast") or 1))
        solver_repeats = max(1, int(repeats_config.get("solver") or 1))
        return [
            f"forecast_{forecast_index}__solver_{solver_index}"
            for forecast_index in range(1, forecast_repeats + 1)
            for solver_index in range(1, solver_repeats + 1)
        ]
    repeats = int(repeats_config)
    if repeats <= 1:
        return [None]
    return [str(index + 1) for index in range(repeats)]


def _defer_verification(config: dict) -> bool:
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    return bool(config.get("defer_verification") or metadata.get("defer_verification"))


def _row_repeat_id(row: dict) -> str | None:
    metadata = row.get("metadata") or {}
    value = metadata.get("repeat_id") if isinstance(metadata, dict) else None
    return str(value) if value is not None else None


def _is_suite_config(config: dict) -> bool:
    return "run_id" not in config and bool(config.get("suite_name") or config.get("suite") or config.get("models"))


def _suite_cap(config: dict, explicit: float | None, default: float) -> float:
    if explicit is not None:
        return explicit
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    if metadata.get("cost_cap_usd") is not None:
        return float(metadata["cost_cap_usd"])
    return default


def _run_suite_budget_grid(config_path: str, suite_cfg: dict, args: argparse.Namespace) -> None:
    cap = _suite_cap(suite_cfg, args.cap_usd, 20.0)
    estimate_experiment_cost(
        config_path,
        pricing_path=args.pricing,
        cap_usd=cap,
        force=args.force,
        models_override=args.models,
    )
    suite_name = str(suite_cfg.get("suite_name") or suite_cfg.get("suite") or Path(config_path).stem)
    output_root = Path(str(suite_cfg.get("output_root") or "reports/runs"))
    overwrite_args = ["--overwrite"] if args.overwrite else []
    resume_args = ["--resume"] if args.resume else []
    for model_cfg in suite_model_configs(suite_cfg, suite_name=suite_name, models_override=args.models):
        run_dir = output_root / suite_name / model_cfg["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = run_dir / "config_snapshot.yaml"
        write_redacted_config_snapshot(model_cfg, snapshot)
        command = [
            sys.executable,
            "scripts/run_budget_grid.py",
            "--config",
            str(snapshot),
            *overwrite_args,
            *resume_args,
            "--workers",
            str(max(1, int(args.workers))),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
