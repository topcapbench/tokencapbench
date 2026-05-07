#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from estimate_experiment_cost import estimate_experiment_cost
    from package_run_artifacts import package_run_artifacts
except ImportError:  # pragma: no cover - package import path used by tests.
    from scripts.estimate_experiment_cost import estimate_experiment_cost
    from scripts.package_run_artifacts import package_run_artifacts
from budget2success.analysis.paper import slugify
from budget2success.utils.config import load_yaml
from budget2success.utils.manifest import write_redacted_config_snapshot


def run_suite(
    config_path: str | Path,
    *,
    cap_usd: float = 20.0,
    pricing_path: str | Path = "reports/live_runs/provider_live_cost_estimate.json",
    dry_run: bool = False,
    resume: bool = False,
    force: bool = False,
    overwrite: bool = False,
    workers: int = 1,
    models_override: list[str] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    suite_cfg = load_yaml(config_path)
    suite_name = str(suite_cfg.get("suite_name") or config_path.stem)
    forecast_only = bool(suite_cfg.get("forecast_only") or (suite_cfg.get("metadata") or {}).get("forecast_only"))
    estimate = estimate_experiment_cost(
        config_path,
        pricing_path=pricing_path,
        cap_usd=cap_usd,
        force=force or dry_run,
        models_override=models_override,
    )
    output_root = Path(str(suite_cfg.get("output_root") or "reports/runs"))
    suite_root = output_root / suite_name
    suite_root.mkdir(parents=True, exist_ok=True)

    planned: list[dict[str, Any]] = []
    for model_cfg in suite_model_configs(suite_cfg, suite_name=suite_name, models_override=models_override):
        run_dir = Path(model_cfg["output_dir"]) / model_cfg["run_id"]
        config_snapshot = run_dir / "config_snapshot.yaml"
        commands = _commands(config_snapshot, overwrite=overwrite, resume=resume, workers=workers, forecast_only=forecast_only)
        planned.append({"model": model_cfg["model"], "run_dir": str(run_dir), "config": str(config_snapshot), "commands": commands})
        if dry_run:
            continue
        if resume and _complete_run(run_dir, forecast_only=forecast_only):
            if not forecast_only:
                package_run_artifacts(run_dir, strict=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        write_redacted_config_snapshot(model_cfg, config_snapshot)
        commands_to_execute = commands if forecast_only else commands[:-1]
        for command in _commands_to_run(run_dir, commands_to_execute, resume=resume):
            _run(command)
        if not forecast_only:
            package_run_artifacts(run_dir, strict=True)

    result = {"suite": suite_name, "estimate": estimate, "planned_runs": planned, "dry_run": dry_run}
    if dry_run:
        print(_dry_run_text(result))
    return result


def suite_model_configs(
    suite_cfg: dict[str, Any],
    *,
    suite_name: str | None = None,
    models_override: list[str] | None = None,
) -> list[dict[str, Any]]:
    suite_name = str(suite_name or suite_cfg.get("suite_name") or suite_cfg.get("suite") or "suite")
    configs: list[dict[str, Any]] = []
    prompt_variants = _prompt_variants(suite_cfg)
    repeat_pairs = _repeat_pairs(suite_cfg)
    include_prompt_in_run_id = len(prompt_variants) > 1
    for model in _models(suite_cfg, models_override=models_override):
        model_slug = slugify(model)
        for prompt_index, prompt_path in enumerate(prompt_variants, start=1):
            for repeat_pair in repeat_pairs:
                configs.append(
                    _model_config(
                        suite_cfg,
                        suite_name=suite_name,
                        model=model,
                        model_slug=model_slug,
                        repeat_pair=repeat_pair,
                        prompt_path=prompt_path,
                        prompt_index=prompt_index,
                        include_prompt_in_run_id=include_prompt_in_run_id,
                    )
                )
    return configs


def _models(cfg: dict[str, Any], *, models_override: list[str] | None = None) -> list[str]:
    if models_override:
        return [str(model) for model in models_override]
    if cfg.get("model"):
        return [str(cfg["model"])]
    models: list[str] = []
    for entry in cfg.get("models") or []:
        if isinstance(entry, dict):
            value = entry.get("name") or entry.get("model")
        else:
            value = entry
        if value:
            models.append(str(value))
    if not models:
        raise ValueError("Suite config must define model or models.")
    return models


def _model_config(
    suite_cfg: dict[str, Any],
    *,
    suite_name: str,
    model: str,
    model_slug: str,
    repeat_pair: tuple[int | None, int | None] = (None, None),
    prompt_path: str | None = None,
    prompt_index: int | None = None,
    include_prompt_in_run_id: bool = False,
) -> dict[str, Any]:
    forecast_repeat, solver_repeat = repeat_pair
    run_id = model_slug
    metadata = {**(suite_cfg.get("metadata") or {}), "suite_name": suite_name, "model_slug": model_slug}
    selected_prompt = prompt_path or str(suite_cfg.get("forecast_prompt") or "prompts/forecast_prompt.md")
    if prompt_index is not None:
        metadata = {
            **metadata,
            "prompt_variant_index": prompt_index,
            "prompt_variant_path": selected_prompt,
        }
    if include_prompt_in_run_id:
        run_id = f"{run_id}__prompt_{slugify(Path(selected_prompt).stem)}"
    if forecast_repeat is not None and solver_repeat is not None:
        run_id = f"{run_id}__forecast_repeat_{forecast_repeat}__solver_repeat_{solver_repeat}"
        repeat_id = f"forecast_{forecast_repeat}__solver_{solver_repeat}"
        metadata = {
            **metadata,
            "forecast_repeat_index": forecast_repeat,
            "solver_repeat_index": solver_repeat,
            "repeat_index": forecast_repeat * 1000 + solver_repeat,
            "repeat_ids": [repeat_id],
        }
        metadata.pop("repeats", None)
    cfg = {
        "run_id": run_id,
        "task_file": suite_cfg["task_file"],
        "output_dir": str(Path(str(suite_cfg.get("output_root") or "reports/runs")) / suite_name),
        "provider": suite_cfg.get("provider", "mock"),
        "model": model,
        "scaffold": suite_cfg.get("scaffold", "direct"),
        "forecast_prompt": selected_prompt,
        "solver_prompts": suite_cfg.get("solver_prompts") or {},
        "budget_grid": suite_cfg.get("budget_grid") or {},
        "temperature": suite_cfg.get("temperature", 0.0),
        "max_forecast_tokens": suite_cfg.get("max_forecast_tokens", 1200),
        "limit": suite_cfg.get("limit"),
        "forecast_only": bool(suite_cfg.get("forecast_only", False)),
        "metadata": metadata,
    }
    for passthrough_key in ("base_url", "endpoint", "api_key", "timeout", "timeout_seconds"):
        if passthrough_key in suite_cfg:
            cfg[passthrough_key] = suite_cfg[passthrough_key]
    return cfg


def _prompt_variants(cfg: dict[str, Any]) -> list[str]:
    variants = cfg.get("prompt_variants")
    if not variants:
        return [str(cfg.get("forecast_prompt") or "prompts/forecast_prompt.md")]
    result = [str(value) for value in variants if value]
    if not result:
        raise ValueError("prompt_variants was configured but no usable prompt paths were found.")
    return result


def _repeat_pairs(cfg: dict[str, Any]) -> list[tuple[int | None, int | None]]:
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    repeats = metadata.get("repeats")
    if isinstance(repeats, dict):
        forecast_repeats = max(1, int(repeats.get("forecast") or 1))
        solver_repeats = max(1, int(repeats.get("solver") or 1))
        return [
            (forecast_index, solver_index)
            for forecast_index in range(1, forecast_repeats + 1)
            for solver_index in range(1, solver_repeats + 1)
        ]
    return [(None, None)]


def _commands(config_snapshot: Path, *, overwrite: bool, resume: bool, workers: int, forecast_only: bool = False) -> list[list[str]]:
    overwrite_args = ["--overwrite"] if overwrite else []
    resume_args = ["--resume"] if resume else []
    worker_args = ["--workers", str(max(1, int(workers)))]
    forecast_command = [sys.executable, "scripts/run_forecasts.py", "--config", str(config_snapshot), *overwrite_args, *resume_args, *worker_args]
    if forecast_only:
        return [forecast_command]
    return [
        forecast_command,
        [sys.executable, "scripts/run_budget_grid.py", "--config", str(config_snapshot), *overwrite_args, *resume_args, *worker_args],
        [sys.executable, "scripts/score_results.py", "--config", str(config_snapshot)],
        [sys.executable, "scripts/package_run_artifacts.py", "--run-dir", str(config_snapshot.parent)],
    ]


def _complete_run(run_dir: Path, *, forecast_only: bool = False) -> bool:
    required = ["forecasts.jsonl"] if forecast_only else ["forecasts.jsonl", "outcomes.jsonl", "metrics.json"]
    return all((run_dir / name).exists() for name in required)


def _commands_to_run(run_dir: Path, commands: list[list[str]], *, resume: bool) -> list[list[str]]:
    if not resume:
        return commands
    required_by_script = {"score_results.py": "metrics.json"}
    selected: list[list[str]] = []
    for command in commands:
        script_name = Path(command[1]).name if len(command) > 1 else ""
        artifact_name = required_by_script.get(script_name)
        if artifact_name and (run_dir / artifact_name).exists():
            print(f"[run_experiment_suite] resume: skipping {script_name} for {run_dir}", flush=True)
            continue
        selected.append(command)
    return selected


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _dry_run_text(result: dict[str, Any]) -> str:
    lines = [
        json.dumps(result["estimate"], indent=2, sort_keys=True),
        "",
        "Planned commands:",
    ]
    for planned in result["planned_runs"]:
        lines.append(f"# {planned['model']} -> {planned['run_dir']}")
        for command in planned["commands"]:
            lines.append(" ".join(command))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a repeatable TokenCapBench experiment suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pricing", default="reports/live_runs/provider_live_cost_estimate.json")
    parser.add_argument("--cap-usd", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Archive and replace existing raw JSONL files.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent forecast/solver calls within each model run.")
    parser.add_argument("--models", nargs="+", default=None, help="Override the suite config model list for this invocation.")
    args = parser.parse_args()
    result = run_suite(
        args.config,
        cap_usd=args.cap_usd,
        pricing_path=args.pricing,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
        overwrite=args.overwrite,
        workers=args.workers,
        models_override=args.models,
    )
    if not args.dry_run:
        print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
