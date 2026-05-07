#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.forecasting.prompt_builder import build_forecast_prompt
from budget2success.schemas.records import TaskRecord
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl
from budget2success.utils.token_counting import approximate_token_count


DEFAULT_PRICING_PATH = Path("reports/live_runs/provider_live_cost_estimate.json")


def estimate_experiment_cost(
    config_path: str | Path,
    *,
    pricing_path: str | Path = DEFAULT_PRICING_PATH,
    mode: str = "conservative",
    cost_mode: str = "active_artifact",
    cap_usd: float = 20.0,
    force: bool = False,
    models_override: list[str] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    pricing = _load_pricing(pricing_path)
    tasks = _load_tasks(cfg)
    models = [str(model) for model in models_override] if models_override else _models(cfg)
    repeats = _repeat_count(cfg)
    prompt_variant_paths = _prompt_variants(cfg)
    prompt_variants = len(prompt_variant_paths)
    run_multiplier = repeats * prompt_variants
    forecast_only = _forecast_only(cfg)
    if not models:
        raise ValueError(f"No models configured in {config_path}")

    rows: list[dict[str, Any]] = []
    total = 0.0
    for model in models:
        rates = _rates_for_model(model, pricing)
        model_forecast = 0.0
        model_solver = 0.0
        model_worst = 0.0
        by_source: dict[str, dict[str, Any]] = {}
        for task in tasks:
            grid = _budget_grid_for_task(cfg, task)
            forecast_prompt_token_counts = [
                _forecast_prompt_tokens(cfg, task, grid, prompt_path=prompt_path)
                for prompt_path in prompt_variant_paths
            ]
            forecast_output_tokens = int(cfg.get("max_forecast_tokens") or 1200)
            solver_prompt_tokens = _solver_prompt_tokens(cfg, task)
            observed_factor = _observed_scaling_factor(mode, model)
            forecast_cost = repeats * sum(
                _token_cost(rates, tokens, forecast_output_tokens * observed_factor)
                for tokens in forecast_prompt_token_counts
            )
            solver_cost = 0.0 if forecast_only else run_multiplier * sum(
                _token_cost(rates, solver_prompt_tokens, budget * observed_factor) for budget in grid
            )
            worst_cost = repeats * sum(
                _token_cost(rates, tokens, forecast_output_tokens) for tokens in forecast_prompt_token_counts
            ) + sum(
                0.0 if forecast_only else run_multiplier * _token_cost(rates, solver_prompt_tokens, budget)
                for budget in grid
            )
            model_forecast += forecast_cost
            model_solver += solver_cost
            model_worst += worst_cost
            source = task.source
            entry = by_source.setdefault(source, {"source": source, "tasks": 0, "solver_calls": 0, "estimated_cost_usd": 0.0})
            entry["tasks"] += 1
            entry["solver_calls"] += 0 if forecast_only else len(grid) * run_multiplier
            entry["estimated_cost_usd"] += forecast_cost + solver_cost
        model_total = model_forecast + model_solver
        total += model_total
        rows.append(
            {
                "model": model,
                "forecast_call_cost_usd": round(model_forecast, 6),
                "solver_call_cost_usd": round(model_solver, 6),
                "total_cost_usd": round(model_total, 6),
                "worst_case_cost_usd": round(model_worst, 6),
                "by_source": [dict(value, estimated_cost_usd=round(value["estimated_cost_usd"], 6)) for value in by_source.values()],
                "pricing_basis": rates.get("basis", ""),
            }
        )

    result = {
        "config": str(config_path),
        "suite_name": cfg.get("suite_name") or cfg.get("run_id") or config_path.stem,
        "mode": mode,
        "pricing_config_version": "2026-04-28",
        "cost_mode": "active_artifact_estimate" if cost_mode == "active_artifact" else "historical_ledger",
        "reasoning_tokens_available": False,
        "warning": "Provider invoices may differ from token-based estimate.",
        "cap_usd": cap_usd,
        "tasks": len(tasks),
        "repeats": repeats,
        "prompt_variants": prompt_variants,
        "forecast_only": forecast_only,
        "models": models,
        "estimated_total_cost_usd": round(total, 6),
        "worst_case_total_cost_usd": round(sum(row["worst_case_cost_usd"] for row in rows), 6),
        "per_model": rows,
        "exceeds_cap": total > cap_usd,
    }
    if result["exceeds_cap"] and not force:
        raise SystemExit(
            f"Estimated cost ${total:.4f} exceeds cap ${cap_usd:.2f}. "
            "Use --force only after intentionally accepting the spend."
        )
    return result


def _load_pricing(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pricing file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("pricing", data)


def _load_tasks(cfg: dict[str, Any]) -> list[TaskRecord]:
    task_file = Path(str(cfg.get("task_file") or ""))
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}. Run scripts/build_paper_splits.py first.")
    rows = read_jsonl(task_file)
    limit = cfg.get("limit")
    if limit:
        rows = rows[: int(limit)]
    return [TaskRecord.model_validate(row) for row in rows]


def _models(cfg: dict[str, Any]) -> list[str]:
    if cfg.get("model"):
        return [str(cfg["model"])]
    result: list[str] = []
    for model in cfg.get("models") or []:
        if isinstance(model, dict):
            value = model.get("name") or model.get("model")
        else:
            value = model
        if value:
            result.append(str(value))
    return result


def _repeat_count(cfg: dict[str, Any]) -> int:
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    repeats = metadata.get("repeats") or cfg.get("repeats")
    if isinstance(repeats, dict):
        forecast_repeats = max(1, int(repeats.get("forecast") or 1))
        solver_repeats = max(1, int(repeats.get("solver") or 1))
        return forecast_repeats * solver_repeats
    if metadata.get("repeat_ids"):
        return max(1, len(metadata["repeat_ids"]))
    return max(1, int(repeats or 1))


def _prompt_variants(cfg: dict[str, Any]) -> list[str]:
    variants = cfg.get("prompt_variants")
    if isinstance(variants, list) and variants:
        result = [str(value) for value in variants if value]
        if result:
            return result
    return [str(cfg.get("forecast_prompt") or "prompts/forecast_prompt.md")]


def _forecast_only(cfg: dict[str, Any]) -> bool:
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    return bool(cfg.get("forecast_only") or metadata.get("forecast_only"))


def _rates_for_model(model: str, pricing: dict[str, Any]) -> dict[str, Any]:
    if model == "mock-model":
        return {"input_per_m": 0.0, "output_per_m": 0.0, "basis": "mock client has no API cost"}
    if model not in pricing:
        raise KeyError(f"Missing pricing for model {model!r}. Add it to the pricing JSON before live calls.")
    rates = pricing[model]
    if "input_per_m" not in rates or "output_per_m" not in rates:
        raise KeyError(f"Pricing for {model!r} must include input_per_m and output_per_m.")
    return rates


def _budget_grid_for_task(cfg: dict[str, Any], task: TaskRecord) -> list[int]:
    if task.budget_grid:
        return task.budget_grid
    grid_cfg = cfg.get("budget_grid") or {}
    grid = grid_cfg.get(task.track) or grid_cfg.get("default")
    if not grid:
        raise ValueError(f"No budget grid for task {task.task_id}")
    return [int(value) for value in grid]


def _forecast_prompt_tokens(cfg: dict[str, Any], task: TaskRecord, grid: list[int], *, prompt_path: str | None = None) -> int:
    prompt_path = str(prompt_path or cfg.get("forecast_prompt") or "prompts/forecast_prompt.md")
    prompt = build_forecast_prompt(prompt_path, task, grid, str(cfg.get("scaffold") or "direct"))
    return approximate_token_count(prompt)


def _solver_prompt_tokens(cfg: dict[str, Any], task: TaskRecord) -> int:
    prompt_path = (cfg.get("solver_prompts") or {}).get(task.track)
    if prompt_path and Path(prompt_path).exists():
        template = Path(prompt_path).read_text(encoding="utf-8")
        return approximate_token_count(f"{template}\n\nTask:\n{task.prompt}\n")
    return approximate_token_count(task.prompt)


def _observed_scaling_factor(mode: str, model: str) -> float:
    if mode == "conservative":
        return 1.0
    if mode != "observed-scaling":
        raise ValueError("mode must be conservative or observed-scaling")
    summary_path = Path("reports/live_runs/provider_live_summary.csv")
    if not summary_path.exists():
        return 0.65
    # Keep this deliberately conservative: historical completions usually stop
    # below caps, but the estimator should not become a spend-approval shortcut.
    return 0.75


def _token_cost(rates: dict[str, Any], input_tokens: float, output_tokens: float) -> float:
    return (input_tokens / 1_000_000.0) * float(rates["input_per_m"]) + (
        output_tokens / 1_000_000.0
    ) * float(rates["output_per_m"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate TokenCapBench experiment cost before live API calls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pricing", default=str(DEFAULT_PRICING_PATH))
    parser.add_argument("--mode", choices=["conservative", "observed-scaling"], default="conservative")
    parser.add_argument("--cost-mode", choices=["active_artifact", "historical_ledger"], default="active_artifact")
    parser.add_argument("--cap-usd", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--models", nargs="+", default=None, help="Override configured models for this estimate.")
    args = parser.parse_args()
    result = estimate_experiment_cost(
        args.config,
        pricing_path=args.pricing,
        mode=args.mode,
        cost_mode=args.cost_mode,
        cap_usd=args.cap_usd,
        force=args.force,
        models_override=args.models,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
