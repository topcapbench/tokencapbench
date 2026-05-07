#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.utils.jsonl import read_jsonl


OUTPUT_COLUMNS = [
    "task_id",
    "source",
    "model",
    "budget",
    "old_success",
    "new_success",
    "solution_excerpt",
    "gold_answer",
    "human_audit_label",
    "human_false_accept",
    "human_false_reject",
    "audit_note",
    "annotator",
]


def sample_math_verifier_audit(
    *,
    corrections: str | Path,
    out: str | Path,
    n_changed_per_source: int = 25,
    n_unchanged_per_source: int = 25,
    seed: int = 20260428,
    artifact_root: str | Path = "reports/artifacts",
    per_source: int | None = None,
    audit: str | Path | None = None,
) -> Path:
    if per_source is not None:
        n_changed_per_source = max(0, int(per_source) // 2)
        n_unchanged_per_source = max(0, int(per_source) - n_changed_per_source)
    corrections_path = Path(corrections)
    changed_rows = _load_changed_rows(corrections_path)
    unchanged_rows = _load_unchanged_rows(Path(audit) if audit else corrections_path)
    solution_index = _solution_index(Path(artifact_root))
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    sampled.extend(_sample_by_source(changed_rows, n_changed_per_source, rng))
    sampled.extend(_sample_by_source(unchanged_rows, n_unchanged_per_source, rng))
    output_rows = [_render_row(row, solution_index) for row in sampled]
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return output


def _load_changed_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        rows = _read_csv(path)
    normalized = []
    for row in rows:
        source = row.get("source") or _source_from_task_id(str(row.get("task_id") or ""))
        normalized.append(
            {
                **row,
                "source": source,
                "model": row.get("model") or row.get("run_id") or "",
                "old_success": row.get("old_success", row.get("recorded_success", "")),
                "new_success": row.get("new_success", row.get("reverified_success", "")),
                "gold_answer": row.get("gold_answer") or row.get("gold_extract") or row.get("gold") or "",
            }
        )
    return normalized


def _load_unchanged_rows(corrections_path: Path) -> list[dict[str, Any]]:
    audit_path = corrections_path if corrections_path.name.endswith("_audit.csv") else corrections_path.with_name("math_reverification_audit.csv")
    if not audit_path.exists():
        return []
    rows = []
    for row in _read_csv(audit_path):
        if _truthy(row.get("changed")):
            continue
        rows.append(
            {
                **row,
                "old_success": row.get("recorded_success", ""),
                "new_success": row.get("reverified_success", ""),
                "gold_answer": row.get("gold", ""),
            }
        )
    return rows


def _sample_by_source(rows: list[dict[str, Any]], n: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source") or "unknown")].append(row)
    sampled: list[dict[str, Any]] = []
    for source_rows in grouped.values():
        source_rows = list(source_rows)
        rng.shuffle(source_rows)
        sampled.extend(source_rows[:n])
    return sampled


def _solution_index(artifact_root: Path) -> dict[tuple[str, str, str, int], str]:
    index: dict[tuple[str, str, str, int], str] = {}
    for outcomes_path in artifact_root.glob("*/outcomes.jsonl"):
        run_id = outcomes_path.parent.name.split("__")[-1]
        try:
            rows = read_jsonl(outcomes_path)
        except Exception:
            continue
        for row in rows:
            metadata = row.get("metadata") or {}
            key = (
                run_id,
                str(row.get("task_id") or ""),
                str(metadata.get("source") or row.get("source") or ""),
                int(row.get("budget") or 0),
            )
            index[key] = str(row.get("solution") or "")
    return index


def _render_row(row: dict[str, Any], solution_index: dict[tuple[str, str, str, int], str]) -> dict[str, Any]:
    budget = int(row.get("budget") or 0)
    source = str(row.get("source") or _source_from_task_id(str(row.get("task_id") or "")))
    run_id = str(row.get("run_id") or row.get("model") or "")
    solution = solution_index.get((run_id, str(row.get("task_id") or ""), source, budget), "")
    return {
        "task_id": row.get("task_id", ""),
        "source": source,
        "model": row.get("model") or row.get("run_id") or "",
        "budget": budget,
        "old_success": row.get("old_success", ""),
        "new_success": row.get("new_success", ""),
        "solution_excerpt": _excerpt(solution or str(row.get("prediction_extract") or row.get("extracted") or "")),
        "gold_answer": row.get("gold_answer") or row.get("gold_extract") or row.get("gold") or "",
        "human_audit_label": "",
        "human_false_accept": "",
        "human_false_reject": "",
        "audit_note": "",
        "annotator": "",
    }


def _excerpt(text: str, limit: int = 280) -> str:
    compact = " ".join(str(text).split())
    return compact[:limit]


def _source_from_task_id(task_id: str) -> str:
    if task_id.startswith("gsm8k"):
        return "gsm8k"
    if task_id.startswith("math") or "hendrycks" in task_id:
        return "hendrycks_math"
    return "unknown"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample rows for manual audit of math verifier corrections.")
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--audit", default=None, help="Existing audit CSV used for unchanged rows; retained for CLI compatibility.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-source", type=int, default=None)
    parser.add_argument("--n-changed-per-source", type=int, default=25)
    parser.add_argument("--n-unchanged-per-source", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--artifact-root", default="reports/artifacts")
    args = parser.parse_args()
    output_path = args.output or args.out
    if not output_path:
        raise SystemExit("Provide --output or --out.")
    output = sample_math_verifier_audit(
        corrections=args.corrections,
        out=output_path,
        n_changed_per_source=args.n_changed_per_source,
        n_unchanged_per_source=args.n_unchanged_per_source,
        seed=args.seed,
        artifact_root=args.artifact_root,
        per_source=args.per_source,
        audit=args.audit,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
