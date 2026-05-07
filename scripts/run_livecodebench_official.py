#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.data.load_tasks import load_tasks_jsonl
from budget2success.execution.livecodebench_bridge import LiveCodeBenchBridge
from budget2success.utils.config import load_yaml
from budget2success.utils.jsonl import read_jsonl, write_jsonl
from budget2success.utils.manifest import sha256_file


COPY_NAMES = (
    "forecasts.jsonl",
    "metrics.json",
    "config_snapshot.yaml",
    "source_config_snapshot.yaml",
    "task_file_hash.json",
    "run_manifest.json",
    "sha256_manifest.json",
)


def run_livecodebench_official(
    *,
    task_file: str | Path,
    run_dirs: list[Path],
    output_dir: str | Path,
    corrected_artifact_root: str | Path,
    timeout_seconds: float | None = None,
    release_version: str = "release_v1",
    num_process_evaluate: int = 4,
    lcb_timeout: int = 6,
) -> list[Path]:
    tasks = load_tasks_jsonl(str(task_file))
    output_root = Path(output_dir)
    corrected_root = Path(corrected_artifact_root)
    corrected_paths: list[Path] = []
    for run_dir in run_dirs:
        corrected_paths.append(
            _run_one(
                tasks=tasks,
                run_dir=run_dir,
                output_root=output_root,
                corrected_root=corrected_root,
                timeout_seconds=timeout_seconds,
                output_root_is_collection=len(run_dirs) > 1,
                release_version=release_version,
                num_process_evaluate=num_process_evaluate,
                lcb_timeout=lcb_timeout,
            )
        )
    return corrected_paths


def _run_one(
    *,
    tasks,
    run_dir: Path,
    output_root: Path,
    corrected_root: Path,
    timeout_seconds: float | None,
    output_root_is_collection: bool,
    release_version: str,
    num_process_evaluate: int,
    lcb_timeout: int,
) -> Path:
    if not (run_dir / "outcomes.jsonl").exists():
        raise FileNotFoundError(run_dir / "outcomes.jsonl")
    run_output_dir = output_root / run_dir.name if output_root_is_collection else output_root
    run_output_dir.mkdir(parents=True, exist_ok=True)
    bridge = LiveCodeBenchBridge(run_output_dir)
    outcomes = read_jsonl(run_dir / "outcomes.jsonl")
    prediction_paths = bridge.write_predictions_grouped_by_budget(tasks, outcomes, run_output_dir)
    start_date, end_date = _task_date_window(tasks)
    lcb_cwd = _livecodebench_repo_root()
    success_by_budget: dict[int, dict[str, bool]] = {}
    harness_results: dict[str, Any] = {}
    for budget, prediction_path in sorted(prediction_paths.items()):
        budget_output_dir = run_output_dir / f"budget_{budget}"
        budget_output_dir.mkdir(parents=True, exist_ok=True)
        _expand_predictions_to_official_benchmark(
            prediction_path,
            release_version=release_version,
            start_date=start_date,
            end_date=end_date,
        )
        command = [
            "env",
            f"PYTHONPATH={_official_lcb_pythonpath(lcb_cwd)}",
            "python",
            "-m",
            "lcb_runner.runner.custom_evaluator",
            "--scenario",
            "codegeneration",
            "--release_version",
            release_version,
            "--num_process_evaluate",
            str(num_process_evaluate),
            "--timeout",
            str(lcb_timeout),
            "--custom_output_file",
            str(prediction_path.resolve()),
        ]
        if start_date:
            command.extend(["--start_date", start_date])
        if end_date:
            command.extend(["--end_date", end_date])
        result = bridge.run_evaluation(prediction_path, command=command, cwd=lcb_cwd, timeout_seconds=timeout_seconds)
        harness_results[str(budget)] = {
            "returncode": result.returncode,
            "success": result.success,
            "timed_out": result.timed_out,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
            "release_version": release_version,
            "start_date": start_date,
            "end_date": end_date,
            "num_process_evaluate": num_process_evaluate,
            "lcb_timeout": lcb_timeout,
            "lcb_cwd": str(lcb_cwd),
        }
        if not result.success:
            raise RuntimeError(
                f"LiveCodeBench official evaluation failed for {run_dir.name} budget {budget}: "
                f"returncode={result.returncode}; stderr={result.stderr[-1000:]}"
            )
        result_path = _find_result_path(result.stdout, run_output_dir, prediction_path)
        if result_path is None:
            raise FileNotFoundError(
                f"Could not find official LiveCodeBench result JSON/JSONL for budget {budget} under {run_output_dir}"
            )
        shutil.copy2(result_path, budget_output_dir / result_path.name)
        harness_results[str(budget)]["result_path"] = str(result_path)
        success_by_budget[budget] = bridge.parse_official_results(result_path)

    corrected_dir = corrected_root / run_dir.name
    corrected_dir.mkdir(parents=True, exist_ok=True)
    merged = bridge.merge_official_results_into_outcomes(outcomes, tasks, success_by_budget)
    write_jsonl(corrected_dir / "outcomes.jsonl", merged)
    _copy_run_context(run_dir, corrected_dir)
    _write_manifest(
        corrected_dir=corrected_dir,
        source_run_dir=run_dir,
        output_dir=run_output_dir,
        prediction_paths=prediction_paths,
        harness_results=harness_results,
    )
    _write_sha_manifest(corrected_dir)
    return corrected_dir


def _find_result_path(stdout: str, output_dir: Path, prediction_path: Path) -> Path | None:
    for token in re.findall(r"[\w./:-]+\.jsonl?|[\w./:-]+\.json", stdout):
        path = Path(token)
        if path.exists() and path != prediction_path:
            return path
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and path != prediction_path
        and not path.name.endswith("_official_subset_manifest.json")
        and path.name != "official_livecodebench_manifest.json"
    ]
    if not candidates:
        return None
    preferred = [path for path in candidates if path.name.endswith("_eval_all.json")]
    if preferred:
        return max(preferred, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _task_date_window(tasks) -> tuple[str | None, str | None]:
    dates = []
    for task in tasks:
        contest_date = (task.metadata or {}).get("contest_date")
        if isinstance(contest_date, str) and len(contest_date) >= 10:
            dates.append(contest_date[:10])
    if not dates:
        return None, None
    return min(dates), max(dates)


def _expand_predictions_to_official_benchmark(
    prediction_path: Path,
    *,
    release_version: str,
    start_date: str | None,
    end_date: str | None,
) -> None:
    """Pad custom outputs so lcb_runner can evaluate a task subset.

    The official custom evaluator currently requires one record for every loaded
    benchmark problem. TokenCapBench only generates a curated fresh subset, so
    missing official problems are padded with an empty program and ignored when
    labels are merged back into our outcomes.
    """

    raw = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a LiveCodeBench custom-output list in {prediction_path}")
    provided: dict[str, list[str]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError(f"Expected dict rows in {prediction_path}")
        question_id = str(row.get("question_id") or "")
        code_list = row.get("code_list")
        if not question_id or not isinstance(code_list, list):
            raise ValueError(f"Missing question_id/code_list in {prediction_path}")
        provided[question_id] = [str(item) for item in code_list]

    benchmark = _load_official_benchmark(release_version, start_date, end_date)
    official_ids = [str(problem.question_id) for problem in sorted(benchmark, key=lambda item: item.question_id)]
    provided_ids = set(provided)
    if not provided_ids.issubset(set(official_ids)):
        benchmark = _load_official_benchmark(release_version, None, None)
        official_ids = [str(problem.question_id) for problem in sorted(benchmark, key=lambda item: item.question_id)]
    missing_from_official = sorted(provided_ids.difference(official_ids))
    if missing_from_official:
        raise ValueError(
            "LiveCodeBench official benchmark does not contain generated question ids: "
            + ", ".join(missing_from_official[:10])
        )
    if len(raw) == len(official_ids) and provided_ids == set(official_ids):
        return

    expanded = [{"question_id": question_id, "code_list": provided.get(question_id, [""])} for question_id in official_ids]
    prediction_path.write_text(json.dumps(expanded, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "release_version": release_version,
        "start_date": start_date,
        "end_date": end_date,
        "provided_questions": len(provided_ids),
        "official_questions": len(official_ids),
        "padded_questions": len(official_ids) - len(provided_ids),
    }
    prediction_path.with_name(f"{prediction_path.stem}_official_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_official_benchmark(release_version: str, start_date: str | None, end_date: str | None):
    _ensure_official_lcb_on_path()
    try:
        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    except ImportError as exc:
        raise RuntimeError(
            "lcb_runner is required for official LiveCodeBench labeling. "
            "Install the official LiveCodeBench package or run this script with its environment."
        ) from exc
    return load_code_generation_dataset(release_version, start_date=start_date, end_date=end_date)


def _livecodebench_repo_root() -> Path:
    external_root = Path("external/LiveCodeBench").resolve()
    if (external_root / "lcb_runner" / "runner" / "custom_evaluator.py").exists():
        return external_root
    try:
        import importlib.util

        spec = importlib.util.find_spec("lcb_runner")
    except ImportError as exc:
        raise RuntimeError(
            "lcb_runner is required for official LiveCodeBench labeling. "
            "Install the official LiveCodeBench package or run this script with its environment."
        ) from exc
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed lcb_runner package.")
    return Path(next(iter(spec.submodule_search_locations))).resolve().parents[0]


def _official_lcb_pythonpath(repo_root: Path) -> str:
    entries = [
        str(Path("tools/lcb_torch_stub").resolve()),
        str(repo_root.resolve()),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _ensure_official_lcb_on_path() -> None:
    for path in (Path("tools/lcb_torch_stub").resolve(), Path("external/LiveCodeBench").resolve()):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


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


def _write_manifest(
    *,
    corrected_dir: Path,
    source_run_dir: Path,
    output_dir: Path,
    prediction_paths: dict[int, Path],
    harness_results: dict[str, Any],
) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(source_run_dir),
        "official_output_dir": str(output_dir),
        "corrected_dir": str(corrected_dir),
        "label_source": "official_livecodebench",
        "prediction_paths": {str(budget): str(path) for budget, path in sorted(prediction_paths.items())},
        "harness_results": harness_results,
    }
    (corrected_dir / "official_livecodebench_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_sha_manifest(directory: Path) -> None:
    payload = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    (directory / "sha256_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _discover_run_dirs(run_dir: str | Path | None, run_root: str | Path | None) -> list[Path]:
    if run_dir:
        return [Path(run_dir)]
    if not run_root:
        raise ValueError("Provide --run-dir or --run-root")
    root = Path(run_root)
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "outcomes.jsonl").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official LiveCodeBench evaluation and merge labels into outcomes.")
    parser.add_argument("--config", help="Optional suite config; preserves explicit --task-file/--run-* behavior.")
    parser.add_argument("--task-file")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--run-dir")
    group.add_argument("--run-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--corrected-artifact-root")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--release-version", default="release_v1")
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    parser.add_argument("--lcb-timeout", type=int, default=6)
    args = parser.parse_args()
    task_file, run_root, output_dir, corrected_root = _resolve_cli_paths(args)
    run_dirs = _discover_run_dirs(args.run_dir, run_root)
    if not run_dirs:
        raise SystemExit("No LiveCodeBench run directories with outcomes.jsonl were found.")
    corrected = run_livecodebench_official(
        task_file=task_file,
        run_dirs=run_dirs,
        output_dir=output_dir,
        corrected_artifact_root=corrected_root,
        timeout_seconds=args.timeout_seconds,
        release_version=args.release_version,
        num_process_evaluate=args.num_process_evaluate,
        lcb_timeout=args.lcb_timeout,
    )
    print(json.dumps({"corrected_artifacts": [str(path) for path in corrected]}, indent=2))


def _resolve_cli_paths(args: argparse.Namespace) -> tuple[str, Path | None, Path, Path]:
    if args.config:
        cfg = load_yaml(args.config)
        suite_name = str(cfg.get("suite_name") or cfg.get("suite") or Path(args.config).stem)
        output_root = Path(str(cfg.get("output_root") or "reports/runs"))
        task_file = str(args.task_file or cfg.get("task_file") or "")
        run_root = Path(args.run_root) if args.run_root else output_root / suite_name
        output_dir = Path(args.output_dir) if args.output_dir else Path("reports/livecodebench_official") / suite_name
        corrected_root = (
            Path(args.corrected_artifact_root)
            if args.corrected_artifact_root
            else Path("reports/artifacts_livecodebench_official") / suite_name
        )
    else:
        task_file = str(args.task_file or "")
        run_root = Path(args.run_root) if args.run_root else None
        output_dir = Path(args.output_dir) if args.output_dir else None
        corrected_root = Path(args.corrected_artifact_root) if args.corrected_artifact_root else None
    if not task_file:
        raise SystemExit("Provide --task-file or --config with task_file.")
    if args.run_dir is None and run_root is None:
        raise SystemExit("Provide --run-dir, --run-root, or --config.")
    if output_dir is None:
        raise SystemExit("Provide --output-dir or --config.")
    if corrected_root is None:
        raise SystemExit("Provide --corrected-artifact-root or --config.")
    return task_file, run_root, output_dir, corrected_root


if __name__ == "__main__":
    main()
