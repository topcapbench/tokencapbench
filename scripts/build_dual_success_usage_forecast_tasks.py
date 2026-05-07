#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_COUNTS = {
    "gsm8k": 75,
    "hendrycks_math": 75,
    "evalplus_humaneval": 75,
    "evalplus_mbpp": 75,
}
SOURCE_FILES = {
    "gsm8k": Path("data/processed/paper_math_core.jsonl"),
    "hendrycks_math": Path("data/processed/paper_math_core.jsonl"),
    "evalplus_humaneval": Path("data/processed/paper_evalplus_humaneval_full.jsonl"),
    "evalplus_mbpp": Path("data/processed/paper_evalplus_mbpp_full.jsonl"),
}
DUAL_BUDGET_GRID = [64, 128, 256, 512, 1024, 2048]


def build_dual_success_usage_forecast_tasks(
    *,
    output: str | Path = "data/processed/paper_dual_success_usage_forecast_300.jsonl",
    artifact_root: str | Path = "reports/artifacts",
    counts: dict[str, int] | None = None,
    seed: int = 20260430,
) -> Path:
    counts = counts or dict(DEFAULT_COUNTS)
    eligible_ids = _task_ids_with_outcomes(Path(artifact_root))
    rng = random.Random(seed)
    selected: list[TaskRecord] = []
    for source, count in counts.items():
        candidates = [
            TaskRecord.model_validate(row)
            for row in read_jsonl(SOURCE_FILES[source])
            if str(row.get("source") or "") == source and str(row.get("task_id") or "") in eligible_ids
        ]
        if len(candidates) < count:
            raise ValueError(f"Requested {count} {source} tasks, but only {len(candidates)} have existing outcomes.")
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:count], key=lambda task: task.task_id))
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(selected):
        task.budget_grid = list(DUAL_BUDGET_GRID)
        task.metadata = {
            **task.metadata,
            "paper_role": "token_usage_proxy_expanded_300",
            "paper_split": Path(output).stem,
            "paper_order": index,
            "reuse_existing_outcomes": True,
        }
        row = task.model_dump(mode="json")
        for optional_key in ("fresh_split", "verifier_policy"):
            if row.get(optional_key) is None:
                row.pop(optional_key, None)
        rows.append(row)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    return output_path


def _task_ids_with_outcomes(artifact_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in artifact_root.glob("**/outcomes.jsonl"):
        for row in read_jsonl(path):
            if row.get("task_id") is not None:
                ids.add(str(row["task_id"]))
    return ids


def _parse_counts(raw: str | None) -> dict[str, int]:
    if not raw:
        return dict(DEFAULT_COUNTS)
    counts = dict(DEFAULT_COUNTS)
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        source, value = chunk.split("=", maxsplit=1)
        source = source.strip()
        if source not in counts:
            raise ValueError(f"Unknown source {source!r}; expected one of {sorted(counts)}")
        counts[source] = int(value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the expanded dual success-vs-usage forecast task file.")
    parser.add_argument("--output", default="data/processed/paper_dual_success_usage_forecast_300.jsonl")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--counts", default=None, help="Comma-separated source=count overrides.")
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()
    path = build_dual_success_usage_forecast_tasks(
        output=args.output,
        artifact_root=args.artifact_root,
        counts=_parse_counts(args.counts),
        seed=args.seed,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
