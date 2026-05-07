#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from scripts.analyze_token_usage_proxy import analyze_token_usage_proxy
except ModuleNotFoundError:  # pragma: no cover
    from analyze_token_usage_proxy import analyze_token_usage_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare token-usage proxy forecasts to verified success forecasts.")
    parser.add_argument("--artifact-root", action="append", default=None)
    parser.add_argument("--split-dir", default="reports/splits")
    parser.add_argument("--dual-forecast-root", default="reports/runs/paper_dual_success_usage_forecast")
    parser.add_argument(
        "--output-table",
        default="reports/tables/paper_table_token_usage_proxy_vs_success.csv",
    )
    parser.add_argument(
        "--output-figure-prefix",
        default="reports/figures/paper_figure_token_usage_proxy_vs_success",
    )
    parser.add_argument("--suite-filter", nargs="*", default=None)
    args = parser.parse_args()
    outputs = analyze_token_usage_proxy(
        artifact_root=args.artifact_root or ["reports/artifacts"],
        split_dir=args.split_dir,
        dual_forecast_root=args.dual_forecast_root,
        output_table=args.output_table,
        output_figure_prefix=args.output_figure_prefix,
        write_figure=True,
        suite_filter=set(args.suite_filter) if args.suite_filter else None,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
