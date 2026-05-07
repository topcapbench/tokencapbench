#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl, write_jsonl


MATH_BUDGETS = [128, 512, 2048]
CODING_BUDGETS = [256, 1024, 2048]


def build_repeatability_subset(
    *,
    math_source: str | Path = "data/processed/paper_math_core.jsonl",
    coding_source: str | Path = "data/processed/paper_evalplus_humaneval_full.jsonl",
    math_limit: int = 50,
    coding_limit: int = 50,
    seed: int = 20260428,
    output: str | Path = "data/processed/paper_repeatability_small.jsonl",
) -> Path:
    rng = random.Random(seed)
    math_tasks = _sample_tasks(math_source, math_limit, rng, track="math", budget_grid=MATH_BUDGETS)
    coding_tasks = _sample_tasks(coding_source, coding_limit, rng, track="coding", budget_grid=CODING_BUDGETS)
    tasks = math_tasks + coding_tasks
    for index, task in enumerate(tasks):
        task.metadata = {
            **task.metadata,
            "paper_split": "paper_repeatability_small",
            "paper_role": "repeatability",
            "paper_order": index,
            "repeatability_seed": seed,
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, tasks)
    return output


def _sample_tasks(
    source_path: str | Path,
    limit: int,
    rng: random.Random,
    *,
    track: str,
    budget_grid: list[int],
) -> list[TaskRecord]:
    rows = read_jsonl(source_path)
    candidates = [TaskRecord.model_validate(row) for row in rows if str(row.get("track") or "") == track]
    candidates.sort(key=lambda task: task.task_id)
    if len(candidates) > limit:
        candidates = rng.sample(candidates, limit)
        candidates.sort(key=lambda task: task.task_id)
    for task in candidates:
        task.budget_grid = list(budget_grid)
        task.fresh_split = "repeatability"
    return candidates[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stable math/coding subset for API repeatability experiments.")
    parser.add_argument("--math-source", default="data/processed/paper_math_core.jsonl")
    parser.add_argument("--coding-source", default="data/processed/paper_evalplus_humaneval_full.jsonl")
    parser.add_argument("--math-limit", type=int, default=50)
    parser.add_argument("--coding-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--output", default="data/processed/paper_repeatability_small.jsonl")
    args = parser.parse_args()
    path = build_repeatability_subset(
        math_source=args.math_source,
        coding_source=args.coding_source,
        math_limit=args.math_limit,
        coding_limit=args.coding_limit,
        seed=args.seed,
        output=args.output,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
