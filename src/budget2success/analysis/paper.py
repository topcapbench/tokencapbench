from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from budget2success.metrics.calibration import brier_score, expected_calibration_error
from budget2success.metrics.regret import (
    normalized_budget_regret,
    oracle_utility,
    selected_budget_from_forecast,
    utility,
)
from budget2success.metrics.first_success_budget import (
    absolute_log_budget_error,
    censored_lower_bound_error,
    max_budget_failure_rate,
    observed_censored_at_budget,
    observed_first_success_budget,
    overbudget_ratio,
    overbudget_waste_factor,
    signed_log_budget_error,
    solved_only_log_token_error,
    underbudgeted,
    underbudget_shortfall_factor,
)
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl


DEFAULT_RUN_ROOT = Path("reports/runs")
DEFAULT_ARTIFACT_ROOT = Path("reports/artifacts")

LEGACY_SUITE_GLOBS = {
    "paper_math_core": ["provider_heldout60_*"],
    "paper_evalplus_humaneval_full": ["provider_evalplus20_*"],
}

DEFAULT_PAPER_SUITES = (
    "paper_math_core",
    "paper_evalplus_humaneval_full",
    "paper_evalplus_mbpp_full",
)


@dataclass(frozen=True)
class PaperRun:
    run_dir: Path
    run_id: str
    model: str
    suite: str | None
    forecasts: list[dict[str, Any]]
    outcomes: list[dict[str, Any]]
    metrics: dict[str, Any]
    config: dict[str, Any]
    artifact_source: str = "runs"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "unnamed"


def discover_run_dirs(suite: str | None = None, run_root: str | Path = DEFAULT_RUN_ROOT) -> list[Path]:
    root = Path(run_root)
    if not root.exists():
        return []
    candidates: list[Path] = []
    if suite:
        suite_root = root / suite
        if suite_root.exists():
            candidates.extend(path for path in suite_root.iterdir() if path.is_dir())
        if not candidates:
            for pattern in LEGACY_SUITE_GLOBS.get(suite, []):
                candidates.extend(root.glob(pattern))
    else:
        for suite_name in DEFAULT_PAPER_SUITES:
            suite_root = root / suite_name
            if suite_root.exists():
                candidates.extend(path for path in suite_root.iterdir() if path.is_dir())
    unique = sorted({path.resolve() for path in candidates})
    return [path for path in unique if (path / "forecasts.jsonl").exists() and (path / "outcomes.jsonl").exists()]


def discover_artifact_dirs(
    suite: str | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> list[Path]:
    root = Path(artifact_root)
    if not root.exists():
        return []
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and (path / "forecasts.jsonl").exists() and (path / "outcomes.jsonl").exists()
    ]
    result: list[Path] = []
    for path in candidates:
        config = load_run_config(path)
        inferred = infer_suite_from_artifact_dir(path, config)
        if suite and suite not in str(path) and inferred != suite:
            continue
        if suite is None and inferred not in DEFAULT_PAPER_SUITES:
            continue
        result.append(path.resolve())
    return sorted({path for path in result})


def infer_suite_from_artifact_dir(path: str | Path, config: dict[str, Any] | None = None) -> str | None:
    config = config or {}
    for key in ("suite", "suite_name"):
        if config.get(key):
            return str(config[key])
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    for key in ("suite", "suite_name"):
        if metadata.get(key):
            return str(metadata[key])
    name = Path(path).name
    for suite_name in DEFAULT_PAPER_SUITES:
        if name == suite_name or name.startswith(f"{suite_name}__"):
            return suite_name
    for part in Path(path).parts:
        if part in DEFAULT_PAPER_SUITES or part.startswith("paper_"):
            return part
    return None


def load_paper_runs(
    suite: str | None = None,
    run_dirs: Iterable[str | Path] | None = None,
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    artifact_root: str | Path | None = DEFAULT_ARTIFACT_ROOT,
    include_artifacts: bool = True,
    corrected_artifact_root: str | Path | None = None,
    official_artifact_roots: Iterable[str | Path] | None = None,
    math_label_mode: str = "original",
) -> list[PaperRun]:
    if math_label_mode not in {"original", "strict", "corrected"}:
        raise ValueError("math_label_mode must be 'original', 'strict', or 'corrected'")
    if run_dirs:
        paths_with_source = [(Path(path), "explicit") for path in run_dirs]
    else:
        paths_with_source = [(path, "runs") for path in discover_run_dirs(suite, run_root)]
        if include_artifacts and artifact_root is not None:
            artifact_paths = discover_artifact_dirs(suite, artifact_root)
            paths_with_source.extend((path, "artifacts") for path in artifact_paths)
    explicit_run_dirs = bool(run_dirs)
    paths_with_source = sorted(
        _dedupe_paths(paths_with_source),
        key=lambda item: _candidate_priority(item[0], item[1]),
    )
    runs: list[PaperRun] = []
    seen_semantic: set[tuple[str | None, str, str]] = set()
    for run_dir, source in paths_with_source:
        forecasts = read_jsonl(run_dir / "forecasts.jsonl")
        outcomes = read_jsonl(run_dir / "outcomes.jsonl")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        config = load_run_config(run_dir)
        model = infer_model(run_dir, config, forecasts, outcomes)
        run_suite = suite if suite is not None else (
            infer_suite_from_artifact_dir(run_dir, config) if source == "artifacts" else infer_suite(run_dir)
        )
        if math_label_mode in {"strict", "corrected"} and _is_math_run(run_suite, outcomes):
            if corrected_artifact_root is None:
                raise FileNotFoundError("Strict math label mode requires corrected_artifact_root.")
            corrected_path = _find_corrected_outcomes(run_dir, corrected_artifact_root)
            if corrected_path is None:
                raise FileNotFoundError(
                    f"Strict math labels requested, but no corrected outcomes were found for {run_dir.name} "
                    f"under {corrected_artifact_root}."
                )
            outcomes = read_jsonl(corrected_path)
        official_path = _find_official_outcomes(run_dir, official_artifact_roots or [], suite=run_suite)
        if official_path is not None:
            outcomes = read_jsonl(official_path)
            official_metrics = official_path.parent / "metrics.json"
            if official_metrics.exists():
                metrics = json.loads(official_metrics.read_text(encoding="utf-8"))
        else:
            outcomes = _exclude_unverified_outcomes(outcomes)
        semantic_key = _semantic_dedupe_key(
            run_suite,
            config.get("run_id") or run_dir.name,
            model,
            explicit_run_dirs=explicit_run_dirs,
        )
        if semantic_key in seen_semantic:
            continue
        seen_semantic.add(semantic_key)
        runs.append(
            PaperRun(
                run_dir=run_dir,
                run_id=config.get("run_id") or run_dir.name,
                model=model,
                suite=run_suite,
                forecasts=forecasts,
                outcomes=outcomes,
                metrics=metrics,
                config=config,
                artifact_source=source,
            )
        )
    return runs


def _is_math_run(suite: str | None, outcomes: list[dict[str, Any]]) -> bool:
    if suite == "paper_math_core":
        return True
    return any(str((row.get("metadata") or {}).get("track") or row.get("track") or "") == "math" for row in outcomes)


def _find_corrected_outcomes(run_dir: Path, corrected_artifact_root: str | Path) -> Path | None:
    root = Path(corrected_artifact_root)
    direct = root / run_dir.name / "outcomes.jsonl"
    if direct.exists():
        return direct
    for path in sorted(root.rglob("outcomes.jsonl")):
        if path.parent.name == run_dir.name:
            return path
    return None


def _find_official_outcomes(
    run_dir: Path,
    official_artifact_roots: Iterable[str | Path],
    *,
    suite: str | None,
) -> Path | None:
    if suite not in {
        "paper_livecodebench_fresh_small",
        "paper_livecodebench_fresh_200",
        "paper_livecodebench_fresh_300",
        "paper_swe_verified_mini_official",
    }:
        return None
    run_name = run_dir.name
    candidate_names = {run_name}
    if "__" in run_name:
        candidate_names.add(run_name.split("__", maxsplit=1)[-1])
    for raw_root in official_artifact_roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for name in candidate_names:
            if suite:
                suite_direct = root / suite / name / "outcomes.jsonl"
                if suite_direct.exists():
                    return suite_direct
            direct = root / name / "outcomes.jsonl"
            if direct.exists():
                return direct
        for path in sorted(root.rglob("outcomes.jsonl")):
            parent = path.parent.name
            if parent in candidate_names or any(parent.endswith(f"__{name}") for name in candidate_names):
                return path
    return None


def _exclude_unverified_outcomes(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in outcomes if not _exclude_from_main_metrics(row)]


def _exclude_from_main_metrics(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    verification_metadata = verification.get("metadata") if isinstance(verification.get("metadata"), dict) else {}
    return bool(metadata.get("exclude_from_main_metrics") is True or verification_metadata.get("exclude_from_main_metrics") is True)


def _dedupe_paths(paths_with_source: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[Path] = set()
    result: list[tuple[Path, str]] = []
    for path, source in paths_with_source:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append((resolved, source))
    return result


def _candidate_priority(path: Path, source: str) -> tuple[int, str]:
    config = load_run_config(path)
    suite = infer_suite_from_artifact_dir(path, config) if source == "artifacts" else infer_suite(path)
    if source == "artifacts" and suite in DEFAULT_PAPER_SUITES and path.name.startswith(f"{suite}__"):
        return (0, str(path))
    if source == "artifacts" and suite in DEFAULT_PAPER_SUITES:
        return (2, str(path))
    if source == "runs" and suite in DEFAULT_PAPER_SUITES:
        return (4, str(path))
    return (6, str(path))


def _semantic_dedupe_key(
    suite: str | None,
    run_id: str,
    model: str,
    *,
    explicit_run_dirs: bool,
) -> tuple[str | None, str, str]:
    if not explicit_run_dirs and suite in DEFAULT_PAPER_SUITES:
        return (suite, "", model)
    return (suite, run_id, model)


def infer_suite(run_dir: str | Path) -> str | None:
    parent = Path(run_dir).parent.name
    if parent in DEFAULT_PAPER_SUITES or parent.startswith("paper_"):
        return parent
    return None


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    for name in ("config_snapshot.yaml", "config.yaml"):
        path = run_path / name
        if path.exists():
            return load_yaml(path)
    legacy = project_root() / "reports" / "live_configs" / f"{run_path.name}.yaml"
    if legacy.exists():
        return load_yaml(legacy)
    return {"run_id": run_path.name}


def infer_model(
    run_dir: str | Path,
    config: dict[str, Any],
    forecasts: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> str:
    if config.get("model"):
        return str(config["model"])
    for rows in (forecasts, outcomes):
        for row in rows:
            if row.get("model"):
                return str(row["model"])
    return Path(run_dir).name


def forecast_curves(forecasts: Iterable[dict[str, Any]]) -> dict[str, dict[int, float]]:
    curves: dict[str, dict[int, float]] = {}
    for row in forecasts:
        if "task_id" not in row or "p_success_by_budget" not in row:
            continue
        curves[str(row["task_id"])] = {int(k): float(v) for k, v in row["p_success_by_budget"].items()}
    return curves


def forecast_medians(forecasts: Iterable[dict[str, Any]]) -> dict[str, float | None]:
    medians: dict[str, float | None] = {}
    for row in forecasts:
        if "task_id" in row and "p_success_by_budget" in row:
            value = row.get("median_budget2success")
            medians[str(row["task_id"])] = float(value) if value is not None else _median_from_curve(
                {int(k): float(v) for k, v in row["p_success_by_budget"].items()}
            )
    return medians


def outcomes_by_task(outcomes: Iterable[dict[str, Any]]) -> dict[str, dict[int, bool]]:
    grouped: dict[str, dict[int, bool]] = defaultdict(dict)
    for row in outcomes:
        grouped[str(row["task_id"])][int(row["budget"])] = bool(row["success"])
    return dict(grouped)


def forecast_monotonicity_violation_rate(curves_by_task: dict[str, dict[int, float]]) -> float | None:
    if not curves_by_task:
        return None
    violations = 0
    for curve in curves_by_task.values():
        ordered = [float(probability) for _, probability in sorted(curve.items())]
        if any(later < earlier - 1e-9 for earlier, later in zip(ordered, ordered[1:])):
            violations += 1
    return violations / len(curves_by_task)


def outcome_nonmonotonicity_rate(outcomes: dict[str, dict[int, bool]]) -> float | None:
    if not outcomes:
        return None
    violations = 0
    for task_outcomes in outcomes.values():
        seen_success = False
        nonmonotone = False
        for _, success in sorted(task_outcomes.items()):
            if seen_success and not success:
                nonmonotone = True
                break
            seen_success = seen_success or bool(success)
        violations += 1 if nonmonotone else 0
    return violations / len(outcomes)


def task_budget_ranking_accuracy(
    predicted_ttg_by_task: dict[str, float | None],
    outcomes: dict[str, dict[int, bool]],
) -> float | None:
    comparable: list[tuple[float, int]] = []
    for task_id, predicted in predicted_ttg_by_task.items():
        observed = observed_first_success_budget(outcomes.get(task_id, {}))
        if predicted is not None and observed is not None:
            comparable.append((float(predicted), int(observed)))
    n_pairs = 0
    score = 0.0
    for i in range(len(comparable)):
        for j in range(i + 1, len(comparable)):
            pred_i, obs_i = comparable[i]
            pred_j, obs_j = comparable[j]
            if obs_i == obs_j:
                continue
            n_pairs += 1
            if pred_i == pred_j:
                score += 0.5
            elif (pred_i < pred_j) == (obs_i < obs_j):
                score += 1.0
    return score / n_pairs if n_pairs else None


def sampled_task_budget_ranking_accuracy(
    predicted_ttg: dict[str, float | None],
    observed_ttg: dict[str, float | None],
    censored_tasks: set[str] | None = None,
    max_pairs: int = 10000,
    seed: int = 0,
) -> float | None:
    """Estimate pairwise task-budget ranking accuracy by sampling task pairs.

    Censored tasks are treated as harder than solved tasks when exactly one task
    in the pair is censored. Pairs where both tasks are censored, both observed
    budget2success values are tied, or either prediction is missing are skipped.
    """
    censored_tasks = censored_tasks or set()
    task_ids = sorted(set(predicted_ttg) & (set(observed_ttg) | censored_tasks))
    if len(task_ids) < 2:
        return None
    predicted_values = np.asarray([float(predicted_ttg.get(task_id) or np.nan) for task_id in task_ids], dtype=float)
    observed_values = np.asarray(
        [float(observed_ttg.get(task_id)) if observed_ttg.get(task_id) is not None else np.nan for task_id in task_ids],
        dtype=float,
    )
    censored = np.asarray([task_id in censored_tasks for task_id in task_ids], dtype=bool)
    all_pair_count = len(task_ids) * (len(task_ids) - 1) // 2
    if all_pair_count <= max_pairs:
        left_idx, right_idx = np.triu_indices(len(task_ids), k=1)
    else:
        rng = np.random.default_rng(seed)
        left_idx = rng.integers(0, len(task_ids), size=max_pairs * 2)
        right_idx = rng.integers(0, len(task_ids), size=max_pairs * 2)
        mask = left_idx != right_idx
        left_idx = left_idx[mask]
        right_idx = right_idx[mask]
        swap = left_idx > right_idx
        left_idx, right_idx = np.where(swap, right_idx, left_idx), np.where(swap, left_idx, right_idx)
        if len(left_idx) > max_pairs:
            left_idx = left_idx[:max_pairs]
            right_idx = right_idx[:max_pairs]

    pred_left = predicted_values[left_idx]
    pred_right = predicted_values[right_idx]
    obs_left = observed_values[left_idx]
    obs_right = observed_values[right_idx]
    left_censored = censored[left_idx]
    right_censored = censored[right_idx]

    valid = np.isfinite(pred_left) & np.isfinite(pred_right) & (pred_left > 0) & (pred_right > 0)
    valid &= ~(left_censored & right_censored)
    exactly_one_censored = left_censored != right_censored
    both_solved = ~left_censored & ~right_censored & np.isfinite(obs_left) & np.isfinite(obs_right) & (obs_left != obs_right)
    valid &= exactly_one_censored | both_solved
    if not np.any(valid):
        return None

    observed_order = np.zeros_like(pred_left, dtype=int)
    observed_order[exactly_one_censored & right_censored] = -1
    observed_order[exactly_one_censored & left_censored] = 1
    observed_order[both_solved & (obs_left < obs_right)] = -1
    observed_order[both_solved & (obs_left > obs_right)] = 1
    predicted_order = np.where(pred_left < pred_right, -1, np.where(pred_left > pred_right, 1, 0))
    scores = np.where(predicted_order == 0, 0.5, (predicted_order == observed_order).astype(float))
    return float(np.mean(scores[valid]))


def truncation_rate(outcome_rows: Iterable[dict[str, Any]]) -> float | None:
    values: list[bool] = []
    for row in outcome_rows:
        if row.get("truncated") is not None:
            values.append(bool(row["truncated"]))
            continue
        completion = row.get("completion_tokens")
        budget = row.get("budget")
        if completion is not None and budget is not None:
            values.append(int(completion) >= int(budget))
    return float(np.mean(values)) if values else None


def task_metadata(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id == "None":
            continue
        row_meta = row.get("metadata") or {}
        metadata.setdefault(task_id, {}).update(
            {
                "track": row_meta.get("track") or row.get("track"),
                "source": row_meta.get("source") or row.get("source"),
                "source_version": row_meta.get("source_version") or row.get("source_version"),
                "external_id": row_meta.get("external_id") or row.get("external_id"),
            }
        )
    return metadata


def score_curve_set(
    curves_by_task: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
    *,
    predicted_ttg_by_task: dict[str, float | None] | None = None,
    outcome_rows: Iterable[dict[str, Any]] | None = None,
    include_pairwise: bool = True,
    token_cost: float = 0.0,
    ranking_max_pairs: int = 10000,
    ranking_seed: int = 0,
) -> dict[str, Any]:
    probabilities: list[float] = []
    labels: list[bool] = []
    log_errors: list[float] = []
    lower_bound_errors: list[float] = []
    signed_errors: list[float] = []
    absolute_errors: list[float] = []
    under_flags: list[bool] = []
    over_flags: list[bool] = []
    over_ratios: list[float] = []
    shortfall_factors: list[float] = []
    waste_factors: list[float] = []
    regrets: list[float] = []
    normalized_regrets: list[float] = []
    all_task_ids = sorted(set(outcomes) & set(curves_by_task))
    first_successes: list[int | None] = []
    censored_ats: list[int | None] = []
    max_fail_flags: list[bool] = []
    predicted_for_ranking: dict[str, float | None] = {}
    observed_for_ranking: dict[str, float | None] = {}
    censored_task_ids: set[str] = set()

    for task_id in all_task_ids:
        curve = curves_by_task[task_id]
        task_outcomes = outcomes.get(task_id, {})
        for budget, probability in curve.items():
            if budget in task_outcomes:
                probabilities.append(float(probability))
                labels.append(bool(task_outcomes[budget]))
        observed = observed_first_success_budget(task_outcomes)
        censored_at = observed_censored_at_budget(task_outcomes)
        first_successes.append(observed)
        censored_ats.append(censored_at)
        observed_for_ranking[task_id] = float(observed) if observed is not None else None
        if censored_at is not None:
            censored_task_ids.add(task_id)
        if task_outcomes:
            max_budget = max(task_outcomes)
            max_fail_flags.append(not bool(task_outcomes[max_budget]))
        predicted = (
            predicted_ttg_by_task.get(task_id)
            if predicted_ttg_by_task and task_id in predicted_ttg_by_task
            else _median_from_curve(curve)
        )
        predicted_for_ranking[task_id] = predicted
        err = solved_only_log_token_error(predicted, observed)
        if err is not None:
            log_errors.append(err)
        signed = signed_log_budget_error(predicted, observed)
        if signed is not None:
            signed_errors.append(signed)
        absolute = absolute_log_budget_error(predicted, observed)
        if absolute is not None:
            absolute_errors.append(absolute)
        lb_err = censored_lower_bound_error(predicted, censored_at)
        if lb_err is not None:
            lower_bound_errors.append(lb_err)
        under = underbudgeted(predicted, observed)
        if under is not None:
            under_flags.append(under)
        over = None if predicted is None or observed is None else predicted > observed
        if over is not None:
            over_flags.append(over)
        shortfall = underbudget_shortfall_factor(predicted, observed)
        if shortfall is not None:
            shortfall_factors.append(shortfall)
        waste = overbudget_waste_factor(predicted, observed)
        if waste is not None:
            waste_factors.append(waste)
        over = overbudget_ratio(predicted, observed)
        if over is not None:
            over_ratios.append(over)
        if task_outcomes and curve:
            selected = selected_budget_from_forecast(curve, reward=1.0, token_cost=token_cost)
            regrets.append(
                oracle_utility(task_outcomes, reward=1.0, token_cost=token_cost)
                - utility(task_outcomes.get(selected, False), selected, reward=1.0, token_cost=token_cost)
            )
            normalized_regrets.append(
                float(normalized_budget_regret(task_outcomes, selected, reward=1.0, token_cost=token_cost))
            )

    censored = [value is None for value in first_successes]
    return {
        "n_tasks": len(all_task_ids),
        "n_budget_observations": len(labels),
        "brier": finite_or_none(brier_score(probabilities, labels)),
        "ece": finite_or_none(expected_calibration_error(probabilities, labels)),
        "success_at_max_budget": finite_or_none(1.0 - float(np.mean(max_fail_flags))) if max_fail_flags else None,
        "censoring_rate": finite_or_none(float(np.mean(censored))) if censored else None,
        "solved_only_log_ttg_error": finite_or_none(float(np.mean(log_errors))) if log_errors else None,
        "signed_log_budget_error_mean": finite_or_none(float(np.mean(signed_errors))) if signed_errors else None,
        "absolute_log_budget_error_mean": finite_or_none(float(np.mean(absolute_errors))) if absolute_errors else None,
        "censored_lower_bound_error": finite_or_none(float(np.mean(lower_bound_errors))) if lower_bound_errors else None,
        "max_budget_failure_rate": finite_or_none(max_budget_failure_rate(list(outcomes.values()))),
        "underbudget_rate": finite_or_none(float(np.mean(under_flags))) if under_flags else None,
        "overbudget_rate": finite_or_none(float(np.mean(over_flags))) if over_flags else None,
        "underbudget_shortfall_factor_mean": finite_or_none(float(np.mean(shortfall_factors))) if shortfall_factors else None,
        "overbudget_waste_factor_mean": finite_or_none(float(np.mean(waste_factors))) if waste_factors else None,
        "overbudget_ratio": finite_or_none(float(np.mean(over_ratios))) if over_ratios else None,
        "regret": finite_or_none(float(np.mean(regrets))) if regrets else None,
        "normalized_regret": finite_or_none(float(np.mean(normalized_regrets))) if normalized_regrets else None,
        "forecast_monotonicity_violation_rate": finite_or_none(forecast_monotonicity_violation_rate(curves_by_task)),
        "outcome_nonmonotonicity_rate": finite_or_none(outcome_nonmonotonicity_rate(outcomes)),
        "task_budget_ranking_accuracy": finite_or_none(
            sampled_task_budget_ranking_accuracy(
                predicted_for_ranking,
                observed_for_ranking,
                censored_tasks=censored_task_ids,
                max_pairs=ranking_max_pairs,
                seed=ranking_seed,
            )
        )
        if include_pairwise
        else None,
        "truncation_rate": finite_or_none(truncation_rate(outcome_rows or [])),
    }


def success_by_budget_rows(
    outcomes: list[dict[str, Any]],
    *,
    suite: str | None,
    run_id: str,
    model: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    for row in outcomes:
        meta = row.get("metadata") or {}
        track = str(meta.get("track") or row.get("track") or "unknown")
        source = str(meta.get("source") or row.get("source") or "unknown")
        grouped[(track, source, int(row["budget"]))].append(bool(row["success"]))
    result = []
    for (track, source, budget), values in sorted(grouped.items()):
        result.append(
            {
                "suite": suite or "",
                "run_id": run_id,
                "model": model,
                "track": track,
                "source": source,
                "budget": budget,
                "n": len(values),
                "success_rate": sum(values) / len(values),
            }
        )
    return result


def calibration_points(
    probabilities: list[float],
    outcomes: list[bool],
    *,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    points: list[tuple[float, float, int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if np.any(mask):
            points.append((float(np.mean(p[mask])), float(np.mean(y[mask])), int(np.sum(mask))))
    return points


def probability_label_pairs(
    curves_by_task: dict[str, dict[int, float]],
    outcomes: dict[str, dict[int, bool]],
) -> tuple[list[float], list[bool]]:
    probabilities: list[float] = []
    labels: list[bool] = []
    for task_id, curve in curves_by_task.items():
        task_outcomes = outcomes.get(task_id, {})
        for budget, probability in curve.items():
            if budget in task_outcomes:
                probabilities.append(float(probability))
                labels.append(bool(task_outcomes[budget]))
    return probabilities, labels


def metric_ci(values: list[float], confidence: float) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    alpha = 1.0 - confidence
    return (
        float(np.mean(arr)),
        float(np.quantile(arr, alpha / 2.0)),
        float(np.quantile(arr, 1.0 - alpha / 2.0)),
    )


def ci_string(mean: Any, low: Any, high: Any, digits: int = 3) -> str:
    if mean is None or low is None or high is None:
        return ""
    try:
        if not all(math.isfinite(float(v)) for v in [mean, low, high]):
            return ""
    except Exception:
        return ""
    return f"{float(mean):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _median_from_curve(curve: dict[int, float]) -> float | None:
    if not curve:
        return None
    for budget, probability in sorted(curve.items()):
        if probability >= 0.5:
            return float(budget)
    return float(max(curve))
