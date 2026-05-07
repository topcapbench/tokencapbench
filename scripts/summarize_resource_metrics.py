#!/usr/bin/env python
"""Summarize token and timing resource metrics from TokenCapBench outcomes.

This script is intentionally lightweight: it reads one or more outcomes.jsonl
files and writes a CSV grouped by model, source/suite, and budget. It is for
SLA-style appendix diagnostics, not for main claims unless measurements are
repeated under controlled serving conditions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

NUMERIC_FIELDS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_visible_tokens",
    "reasoning_tokens",
    "retry_count",
    "wall_time_seconds",
    "generation_wall_time_seconds",
    "verification_wall_time_seconds",
    "end_to_end_wall_time_seconds",
    "generation_wall_time_s",
    "verification_wall_time_s",
    "end_to_end_wall_time_s",
]

TIME_FIELDS = [
    "generation_wall_time_seconds",
    "verification_wall_time_seconds",
    "end_to_end_wall_time_seconds",
    "generation_wall_time_s",
    "verification_wall_time_s",
    "end_to_end_wall_time_s",
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _first_num(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _num(row.get(name))
        if value is not None:
            return value
    return None


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    model = str(row.get("model") or "unknown_model")
    source = str(row.get("suite") or meta.get("source") or row.get("source") or "unknown_source")
    budget = str(row.get("budget") or "unknown_budget")
    return model, source, budget


def summarize(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            groups[group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for (model, source, budget), rows in sorted(groups.items()):
        n = len(rows)
        successes = sum(1 for row in rows if bool(row.get("success")))
        truncated = sum(1 for row in rows if bool(row.get("truncated")))
        cap_hits = sum(1 for row in rows if bool(row.get("cap_hit")))
        retry_values = [_num(row.get("retry_count")) or 0.0 for row in rows]
        finish_length = sum(
            1
            for row in rows
            if str(row.get("finish_reason") or "").strip().lower()
            in {"length", "max_tokens", "max_output_tokens", "token_limit"}
        )
        result: dict[str, Any] = {
            "model": model,
            "source": source,
            "budget": budget,
            "n": n,
            "success_rate": successes / n if n else "",
            "truncated_rate": truncated / n if n else "",
            "cap_hit_rate": cap_hits / n if n else "",
            "length_finish_rate": finish_length / n if n else "",
            "retry_rate": sum(1 for value in retry_values if value > 0) / n if n else "",
            "mean_retry_count": sum(retry_values) / n if n else "",
        }
        for field in NUMERIC_FIELDS:
            values = [v for row in rows if (v := _num(row.get(field))) is not None]
            result[f"{field}_coverage"] = len(values) / n if n else ""
            result[f"{field}_median"] = pct(values, 0.5) if values else ""
            result[f"{field}_p95"] = pct(values, 0.95) if values else ""

        token_sec_values: list[float] = []
        for row in rows:
            completion_tokens = _num(row.get("completion_tokens"))
            generation_time = _first_num(row, "generation_wall_time_seconds", "generation_wall_time_s")
            if completion_tokens is not None and generation_time and generation_time > 0:
                token_sec_values.append(completion_tokens / generation_time)
        result["tokens_per_second_median"] = pct(token_sec_values, 0.5) if token_sec_values else ""
        result["tokens_per_second_p95"] = pct(token_sec_values, 0.95) if token_sec_values else ""
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TokenCapBench token/timing resource metrics.")
    parser.add_argument("--outcomes", nargs="+", required=True, help="One or more outcomes.jsonl files.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()
    paths = [Path(p) for p in args.outcomes]
    rows = summarize(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model", "source", "budget", "n"]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "groups": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
