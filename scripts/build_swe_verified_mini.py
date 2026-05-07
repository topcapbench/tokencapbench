#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl, write_jsonl


def build_swe_verified_mini(
    *,
    output: str | Path,
    n_tasks: int = 20,
    seed: int = 20260430,
    dataset: str = "SWE-bench/SWE-bench_Verified",
    exclude_smoke: bool = False,
) -> Path:
    rows, source_version = _load_swebench_verified(dataset)
    smoke_ids = _smoke_instance_ids() if exclude_smoke else set()
    candidates = []
    for row in rows:
        instance_id = str(row.get("instance_id") or "")
        problem = str(row.get("problem_statement") or "")
        if not instance_id or not problem or instance_id in smoke_ids:
            continue
        candidates.append(row)
    if not candidates:
        raise RuntimeError("No SWE-bench Verified candidates were loaded.")
    median_length = statistics.median(len(str(row.get("problem_statement") or "")) for row in candidates)
    preferred = [row for row in candidates if len(str(row.get("problem_statement") or "")) <= median_length]
    selected = _stratified_by_repo(preferred, n_tasks=n_tasks, seed=seed)
    if len(selected) < n_tasks:
        seen = {str(row.get("instance_id")) for row in selected}
        selected.extend(
            row
            for row in _stratified_by_repo(candidates, n_tasks=n_tasks, seed=seed)
            if str(row.get("instance_id")) not in seen
        )
        selected = selected[:n_tasks]
    if len(selected) < n_tasks:
        raise RuntimeError(f"Could only select {len(selected)} SWE-bench Verified tasks, requested {n_tasks}.")

    task_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        task = _task_from_swebench_row(row, source_version=source_version, paper_order=index)
        dumped = task.model_dump(mode="json")
        for optional_key in ("fresh_split", "verifier_policy"):
            if dumped.get(optional_key) is None:
                dumped.pop(optional_key, None)
        task_rows.append(dumped)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, task_rows)
    return output_path


def _load_swebench_verified(dataset_name: str) -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to build SWE-bench Verified mini tasks.") from exc
    errors: list[str] = []
    for name in [dataset_name, "princeton-nlp/SWE-bench_Verified"]:
        try:
            dataset = load_dataset(name, split="test")
            return [dict(row) for row in dataset], name
        except Exception as exc:  # noqa: BLE001 - try the documented fallback dataset id.
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Could not load SWE-bench Verified dataset. " + " | ".join(errors))


def _stratified_by_repo(rows: list[dict[str, Any]], *, n_tasks: int, seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("repo") or "unknown")].append(row)
    for repo_rows in grouped.values():
        repo_rows.sort(key=lambda row: str(row.get("instance_id") or ""))
        rng.shuffle(repo_rows)
    selected: list[dict[str, Any]] = []
    repos = sorted(grouped)
    while len(selected) < n_tasks and any(grouped[repo] for repo in repos):
        for repo in repos:
            if grouped[repo]:
                selected.append(grouped[repo].pop(0))
                if len(selected) >= n_tasks:
                    break
    return selected


def _task_from_swebench_row(row: dict[str, Any], *, source_version: str, paper_order: int) -> TaskRecord:
    instance_id = str(row.get("instance_id") or "")
    repo = str(row.get("repo") or "")
    base_commit = str(row.get("base_commit") or "")
    problem = str(row.get("problem_statement") or "")
    prompt = (
        f"Repository: {repo}\n"
        f"Base commit: {base_commit}\n\n"
        f"Issue:\n{problem}\n\n"
        "Produce a patch that resolves the issue."
    )
    return TaskRecord(
        task_id=f"swebench_{instance_id}",
        track="swe",
        prompt=prompt,
        verifier="swebench",
        answer=None,
        source="swebench",
        source_version=source_version,
        external_id=instance_id,
        budget_grid=[4096, 16384],
        external_eval={
            "dataset": source_version,
            "split": "test",
            "harness": "swebench",
            "instance_id": instance_id,
        },
        metadata={
            "repo": repo,
            "base_commit": base_commit,
            "problem_statement": problem,
            "paper_role": "swe_official_mini",
            "paper_order": paper_order,
        },
    )


def _smoke_instance_ids(path: str | Path = "data/processed/swe_verified_smoke.jsonl") -> set[str]:
    smoke_path = Path(path)
    if not smoke_path.exists():
        return set()
    ids = set()
    for row in read_jsonl(smoke_path):
        external_id = row.get("external_id")
        if external_id:
            ids.add(str(external_id))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official-harness SWE-bench Verified mini task file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Verified")
    parser.add_argument("--exclude-smoke", action="store_true")
    args = parser.parse_args()
    path = build_swe_verified_mini(
        output=args.output,
        n_tasks=args.n_tasks,
        seed=args.seed,
        dataset=args.dataset,
        exclude_smoke=args.exclude_smoke,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
