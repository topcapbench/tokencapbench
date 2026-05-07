#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_experiment_suite import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a repeatability suite with forecast/solver repeat IDs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pricing", default="reports/live_runs/provider_live_cost_estimate.json")
    parser.add_argument("--cap-usd", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    run_suite(
        args.config,
        pricing_path=args.pricing,
        cap_usd=args.cap_usd,
        dry_run=args.dry_run,
        resume=args.resume,
        force=args.force,
        overwrite=args.overwrite,
        workers=args.workers,
        models_override=args.models,
    )


if __name__ == "__main__":
    main()
