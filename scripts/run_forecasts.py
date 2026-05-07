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

from budget2success.clients.base import GenerationRequest
from budget2success.clients.factory import build_client
from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.forecasting.parse_forecast import parse_forecast_json, validate_forecast_budget_grid
from budget2success.forecasting.prompt_builder import build_forecast_prompt
from budget2success.schemas.records import ExperimentConfig, ForecastErrorRecord
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import append_jsonl, read_jsonl
from budget2success.utils.manifest import write_redacted_config_snapshot, write_run_manifest
try:
    from estimate_experiment_cost import estimate_experiment_cost
    from run_experiment_suite import suite_model_configs
except ImportError:  # pragma: no cover - package import path used by tests.
    from scripts.estimate_experiment_cost import estimate_experiment_cost
    from scripts.run_experiment_suite import suite_model_configs

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TokenCapBench forecasts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Archive an existing forecasts.jsonl before writing a fresh one.")
    parser.add_argument("--resume", action="store_true", help="Append only missing task forecasts when forecasts.jsonl exists.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent forecast calls.")
    parser.add_argument("--pricing", default="reports/live_runs/provider_live_cost_estimate.json")
    parser.add_argument("--cap-usd", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, help="Override configured models for suite configs.")
    args = parser.parse_args()
    raw_config = load_yaml(args.config)
    if _is_suite_config(raw_config):
        _run_suite_forecasts(args.config, raw_config, args)
        return
    cfg = ExperimentConfig.model_validate(raw_config)
    run_dir = Path(cfg.output_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_redacted_config_snapshot(raw_config, run_dir / "config_snapshot.yaml")
    out_path = run_dir / "forecasts.jsonl"
    completed_task_ids: set[str] = set()
    if out_path.exists():
        if args.resume:
            completed_task_ids = {str(row.get("task_id")) for row in read_jsonl(out_path) if row.get("task_id")}
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
    tasks_to_run = [task for task in tasks if task.task_id not in completed_task_ids]

    def run_one(task):
        budget_grid = task.budget_grid or cfg.budget_grid.get(task.track) or cfg.budget_grid.get("default")
        if not budget_grid:
            raise ValueError(f"No budget grid for task {task.task_id} track {task.track}")
        prompt = build_forecast_prompt(cfg.forecast_prompt, task, budget_grid, cfg.scaffold)
        last_response_text: str | None = None
        last_error: Exception | None = None
        for attempt in range(2):
            request_prompt = prompt
            if attempt == 1:
                request_prompt = (
                    f"{prompt}\n\nYour previous response was not valid TokenCapBench forecast JSON. "
                    "Return only the requested JSON object with exactly the requested budget keys."
                )
            started = time.perf_counter()
            try:
                response = client.generate(
                    GenerationRequest(
                        model=cfg.model,
                        prompt=request_prompt,
                        max_tokens=cfg.max_forecast_tokens,
                        temperature=0.0,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve per-task transport failures.
                last_error = exc
                continue
            elapsed = time.perf_counter() - started
            last_response_text = response.text
            try:
                forecast = parse_forecast_json(response.text)
                validate_forecast_budget_grid(forecast, budget_grid)
                forecast.task_id = task.task_id
                forecast.model = cfg.model
                forecast.scaffold = cfg.scaffold
                forecast.budget_grid = budget_grid
                forecast.metadata.update(
                    {
                        "source": task.source,
                        "source_version": task.source_version,
                        "external_id": task.external_id,
                        "track": task.track,
                        "attempt": attempt + 1,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                        "reasoning_tokens": response.reasoning_tokens,
                        "wall_time_seconds": elapsed,
                    }
                )
                return forecast
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            return ForecastErrorRecord(
                task_id=task.task_id,
                model=cfg.model,
                error=str(last_error),
                raw_text=last_response_text,
                metadata={
                    "source": task.source,
                    "source_version": task.source_version,
                    "external_id": task.external_id,
                    "track": task.track,
                    "attempts": 2,
                    "error_type": type(last_error).__name__,
                },
            )
        raise RuntimeError("unreachable forecast state")

    completed = len(completed_task_ids)
    if completed:
        print(f"[run_forecasts] {cfg.run_id}: resuming after {completed}/{len(tasks)} tasks", file=sys.stderr, flush=True)
    workers = max(1, int(args.workers))
    if workers == 1:
        for task in tasks_to_run:
            append_jsonl(out_path, run_one(task))
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                print(f"[run_forecasts] {cfg.run_id}: {completed}/{len(tasks)} tasks", file=sys.stderr, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks_to_run}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - keep unrelated task failures local.
                    record = ForecastErrorRecord(
                        task_id=task.task_id,
                        model=cfg.model,
                        error=str(exc),
                        metadata={
                            "source": task.source,
                            "source_version": task.source_version,
                            "external_id": task.external_id,
                            "track": task.track,
                            "attempts": 0,
                            "error_type": type(exc).__name__,
                        },
                    )
                append_jsonl(out_path, record)
                completed += 1
                if completed == 1 or completed % 25 == 0 or completed == len(tasks):
                    print(f"[run_forecasts] {cfg.run_id}: {completed}/{len(tasks)} tasks", file=sys.stderr, flush=True)
    write_run_manifest(
        run_dir,
        config=raw_config,
        command_line_arguments=sys.argv[1:],
        phase="forecasts",
        extra={"forecast_file": str(out_path)},
    )
    print(json.dumps({"forecasts": str(out_path), "n_tasks": len(tasks)}, indent=2))


def _is_suite_config(config: dict) -> bool:
    return "run_id" not in config and bool(config.get("suite_name") or config.get("suite") or config.get("models"))


def _suite_cap(config: dict, explicit: float | None, default: float) -> float:
    if explicit is not None:
        return explicit
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    if metadata.get("cost_cap_usd") is not None:
        return float(metadata["cost_cap_usd"])
    return default


def _run_suite_forecasts(config_path: str, suite_cfg: dict, args: argparse.Namespace) -> None:
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
            "scripts/run_forecasts.py",
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
