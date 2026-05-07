#!/usr/bin/env python
from __future__ import annotations


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
from pathlib import Path

from budget2success.datasets.base import AdapterConfig
from budget2success.datasets.registry import get_adapter
from budget2success.utils.jsonl import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TokenCapBench tasks from a benchmark source.")
    parser.add_argument("--source", default=None, help="Source adapter name, e.g. local, gsm8k, math, evalplus")
    parser.add_argument("--adapter", default=None, help="Alias for --source used by experiment specs.")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget-grid", nargs="*", type=int, default=None)
    parser.add_argument("--input", default=None, help="Local JSONL input path for --source local")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Source adapter name when --source is omitted; otherwise optional dataset name/subset for source adapters.",
    )
    parser.add_argument("--hf-dataset", default=None, help="Hugging Face dataset override for adapters that support it.")
    parser.add_argument("--source-root", default=None, help="External benchmark checkout root, e.g. Aider Polyglot.")
    parser.add_argument("--languages", default=None, help="Comma-separated language list for polyglot adapters.")
    parser.add_argument("--instruction-style", default=None, help="Instruction style for edit adapters such as CanItEdit.")
    args = parser.parse_args()

    source = args.adapter or args.source or args.dataset
    if not source:
        raise SystemExit("Specify --source or --dataset with a source adapter name.")
    kwargs = {}
    if args.input:
        kwargs["path"] = args.input
    if args.dataset and args.source:
        kwargs["dataset"] = args.dataset
    if args.hf_dataset:
        kwargs["dataset"] = args.hf_dataset
        kwargs["hf_dataset"] = args.hf_dataset
    if args.source_root:
        kwargs["source_root"] = args.source_root
    if args.languages:
        kwargs["languages"] = args.languages
    if args.instruction_style:
        kwargs["instruction_style"] = args.instruction_style
    cfg = AdapterConfig(name=source, split=args.split, limit=args.limit, budget_grid=args.budget_grid, kwargs=kwargs)
    adapter = get_adapter(source, cfg)
    tasks = adapter.load_tasks()
    write_jsonl(args.output, tasks)
    print(f"Wrote {len(tasks)} tasks to {Path(args.output)}")


if __name__ == "__main__":
    main()
