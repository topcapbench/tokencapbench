from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from budget2success.schemas.records import BudgetRunRecord, ForecastRecord


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield idx, json.loads(line)


def validate_forecasts(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        for line_no, row in _read_jsonl(path):
            try:
                ForecastRecord.model_validate(row)
            except Exception as exc:  # noqa: BLE001 - validation CLI should surface all schema failures.
                raise ValueError(f"{path}:{line_no}: invalid forecast row: {exc}") from exc
            count += 1
    return count


def validate_outcomes(paths: list[Path]) -> int:
    count = 0
    for path in paths:
        for line_no, row in _read_jsonl(path):
            try:
                BudgetRunRecord.model_validate(row)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{path}:{line_no}: invalid outcome row: {exc}") from exc
            count += 1
    return count


def _discover_forecasts(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.jsonl"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "forecasts.jsonl" or (name.startswith(("tokencapbench_", "budget2success_")) and "forecast" in name):
            out.append(path)
    return sorted(out)


def _discover_outcomes(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*.jsonl"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "outcomes.jsonl" or (name.startswith(("tokencapbench_", "budget2success_")) and "outcome" in name):
            out.append(path)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TokenCapBench forecast/outcome JSONL schemas.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--all", action="store_true", help="Discover common forecast/outcome JSONL files under root.")
    parser.add_argument("--forecasts", nargs="*", default=[])
    parser.add_argument("--outcomes", nargs="*", default=[])
    args = parser.parse_args()
    root = Path(args.root)
    forecast_paths = [Path(p) for p in args.forecasts]
    outcome_paths = [Path(p) for p in args.outcomes]
    if args.all:
        forecast_paths.extend(_discover_forecasts(root))
        outcome_paths.extend(_discover_outcomes(root))
    forecast_paths = sorted(set(forecast_paths))
    outcome_paths = sorted(set(outcome_paths))
    forecast_count = validate_forecasts(forecast_paths) if forecast_paths else 0
    outcome_count = validate_outcomes(outcome_paths) if outcome_paths else 0
    print(json.dumps({"forecast_files": len(forecast_paths), "forecast_rows": forecast_count, "outcome_files": len(outcome_paths), "outcome_rows": outcome_count}, indent=2))


if __name__ == "__main__":
    main()
