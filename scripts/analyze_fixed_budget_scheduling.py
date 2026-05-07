#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from scripts.run_allocation_frontier import run_allocation_frontier
except ModuleNotFoundError:  # pragma: no cover
    from run_allocation_frontier import run_allocation_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-budget scheduling and allocation-frontier analysis.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument(
        "--frontier-table",
        default="reports/tables/paper_table_allocation_frontier_raw.csv",
    )
    parser.add_argument(
        "--output-table",
        default="reports/tables/paper_table_fixed_budget_scheduling.csv",
    )
    parser.add_argument("--suite-filter", nargs="*", default=None)
    args = parser.parse_args()
    outputs = run_allocation_frontier(
        artifact_root=args.artifact_root,
        split_dir=args.split_dir,
        output_table=args.frontier_table,
        figures_dir=args.figures_dir,
        write_figures=True,
        fixed_budget_table=args.output_table,
        suite_filter=set(args.suite_filter) if args.suite_filter else None,
    )
    figures_dir = Path(args.figures_dir)
    for suffix in (".png", ".svg"):
        source = figures_dir / f"paper_figure9_allocation_frontier{suffix}"
        target = figures_dir / f"paper_figure_allocation_frontier{suffix}"
        if source.exists():
            shutil.copyfile(source, target)
            outputs.append(target)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
