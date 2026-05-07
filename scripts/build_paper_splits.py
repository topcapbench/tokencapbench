#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.datasets.base import AdapterConfig
from budget2success.datasets.registry import get_adapter
from budget2success.execution.math_verifier import classify_math_answer
from budget2success.schemas.records import TaskRecord
from budget2success.utils.jsonl import read_jsonl, write_jsonl


OUTPUTS = {
    "paper_math_core": Path("data/processed/paper_math_core.jsonl"),
    "paper_evalplus_humaneval_full": Path("data/processed/paper_evalplus_humaneval_full.jsonl"),
    "paper_evalplus_mbpp_full": Path("data/processed/paper_evalplus_mbpp_full.jsonl"),
    "paper_livecodebench_fresh_small": Path("data/processed/paper_livecodebench_fresh_small.jsonl"),
    "paper_livecodebench_fresh_300": Path("data/processed/paper_livecodebench_fresh_300.jsonl"),
    "paper_repeatability_small": Path("data/processed/paper_repeatability_small.jsonl"),
    "paper_bfcl_lite": Path("data/processed/paper_bfcl_lite.jsonl"),
    "swe_verified_smoke": Path("data/processed/swe_verified_smoke.jsonl"),
    "paper_bigcodebench_hard": Path("data/processed/paper_bigcodebench_hard.jsonl"),
    "paper_canitedit_descriptive": Path("data/processed/paper_canitedit_descriptive.jsonl"),
    "paper_aider_polyglot_python": Path("data/processed/paper_aider_polyglot_python.jsonl"),
}

DEFAULT_OUTPUT_NAMES = [
    "paper_math_core",
    "paper_evalplus_humaneval_full",
    "paper_evalplus_mbpp_full",
    "paper_livecodebench_fresh_small",
    "paper_repeatability_small",
    "paper_bigcodebench_hard",
    "paper_canitedit_descriptive",
]

BUDGETS = {
    "gsm8k": [64, 128, 256, 512, 1024],
    "math": [128, 256, 512, 1024, 2048],
    "humaneval": [64, 128, 256, 512, 1024, 2048],
    "mbpp": [64, 128, 256, 512, 1024, 2048],
    "livecodebench": [64, 128, 256, 512, 1024, 2048],
    "repeatability": [128, 512, 2048],
    "bfcl": [128, 256, 512, 1024, 2048],
    "swe": [8192, 32768],
    "bigcodebench_hard": [128, 256, 512, 1024, 2048, 4096],
    "canitedit": [128, 256, 512, 1024, 2048],
    "aider_polyglot": [512, 1024, 2048, 4096],
}


def build_splits(
    *,
    track: str = "all",
    small: bool = False,
    output_dir: str | Path = "data/processed",
    bfcl_path: str | None = None,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    wanted = _wanted_outputs(track)
    for name in wanted:
        output_path = output_dir / OUTPUTS[name].name
        if small:
            tasks = _small_tasks(name)
        else:
            tasks = _full_tasks(name, bfcl_path=bfcl_path)
        _write_task_records(output_path, tasks)
        counts[str(output_path)] = len(tasks)
    return counts


def _write_task_records(path: str | Path, tasks: Iterable[TaskRecord]) -> None:
    """Write task manifests without churn from absent optional task annotations."""
    rows = []
    for task in tasks:
        row = task.model_dump(mode="json")
        for optional_key in ("fresh_split", "verifier_policy"):
            if row.get(optional_key) is None:
                row.pop(optional_key, None)
        rows.append(row)
    write_jsonl(path, rows)


def _wanted_outputs(track: str) -> list[str]:
    if track == "all":
        return list(DEFAULT_OUTPUT_NAMES)
    if track == "math":
        return ["paper_math_core"]
    if track == "coding":
        return ["paper_evalplus_humaneval_full", "paper_evalplus_mbpp_full"]
    if track in {"fresh_coding", "livecodebench"}:
        return ["paper_livecodebench_fresh_small"]
    if track in {"livecodebench_300", "fresh_coding_300"}:
        return ["paper_livecodebench_fresh_300"]
    if track == "repeatability":
        return ["paper_repeatability_small"]
    if track == "agentic":
        return ["paper_bfcl_lite"]
    if track == "swe":
        return ["swe_verified_smoke"]
    if track in {"replacement", "replacement_benchmarks", "provider_gateway"}:
        return ["paper_bigcodebench_hard", "paper_canitedit_descriptive"]
    if track in {"bigcodebench_hard", "canitedit_descriptive"}:
        return [f"paper_{track}"]
    raise ValueError(
        f"Unknown track {track!r}. Use all, math, coding, fresh_coding, repeatability, bigcodebench_hard, canitedit_descriptive, or replacement."
    )


def _full_tasks(name: str, *, bfcl_path: str | None) -> list[TaskRecord]:
    if name == "paper_math_core":
        gsm8k = _load_adapter("gsm8k", limit=500, budget_grid=BUDGETS["gsm8k"])
        math = _load_adapter("math", limit=500, budget_grid=BUDGETS["math"])
        return _retag(gsm8k + math, split_name=name, role="paper")
    if name == "paper_evalplus_humaneval_full":
        tasks = _load_adapter("evalplus", budget_grid=BUDGETS["humaneval"], kwargs={"dataset": "humaneval"})
        return _retag(tasks, split_name=name, role="paper")
    if name == "paper_evalplus_mbpp_full":
        tasks = _load_adapter("evalplus", budget_grid=BUDGETS["mbpp"], kwargs={"dataset": "mbpp"})
        return _retag(tasks, split_name=name, role="paper")
    if name == "paper_livecodebench_fresh_small":
        tasks = _load_adapter(
            "livecodebench",
            limit=50,
            budget_grid=BUDGETS["livecodebench"],
            kwargs={"dataset": "livecodebench/code_generation"},
        )
        return _retag(tasks, split_name=name, role="configured_fresh_coding")
    if name == "paper_livecodebench_fresh_300":
        tasks = _load_adapter(
            "livecodebench",
            limit=300,
            budget_grid=BUDGETS["livecodebench"],
            kwargs={"dataset": "livecodebench/code_generation"},
        )
        return _retag(tasks, split_name=name, role="freshness_appendix")
    if name == "paper_repeatability_small":
        return _repeatability_tasks(output_role="repeatability_audit")
    if name == "paper_bfcl_lite":
        if not bfcl_path:
            raise RuntimeError(
                "BFCL-lite requires --bfcl-path pointing to a local BFCL export. "
                "Clone the official BFCL repo/export first, then rerun this command."
            )
        tasks = _load_adapter("bfcl", limit=50, budget_grid=BUDGETS["bfcl"], kwargs={"path": bfcl_path})
        return _retag(tasks, split_name=name, role="future_work")
    if name == "swe_verified_smoke":
        tasks = _load_adapter(
            "swebench",
            limit=5,
            budget_grid=BUDGETS["swe"],
            kwargs={"dataset": "SWE-bench/SWE-bench_Verified"},
        )
        return _retag(tasks, split_name=name, role="future_work")
    if name == "paper_bigcodebench_hard":
        tasks = _load_adapter("bigcodebench_hard", limit=148, budget_grid=BUDGETS["bigcodebench_hard"])
        return _retag(tasks, split_name=name, role="paper_extension")
    if name == "paper_canitedit_descriptive":
        existing = _task_records_from_existing("data/tasks/paper_canitedit_descriptive.jsonl", limit=105)
        if existing:
            return _retag(existing, split_name=name, role="paper_extension_or_appendix")
        tasks = _load_adapter(
            "canitedit",
            limit=105,
            budget_grid=BUDGETS["canitedit"],
            kwargs={"instruction_style": "descriptive"},
        )
        return _retag(tasks, split_name=name, role="paper_extension_or_appendix")
    raise KeyError(name)


def _load_adapter(
    source: str,
    *,
    limit: int | None = None,
    budget_grid: list[int],
    kwargs: dict | None = None,
) -> list[TaskRecord]:
    cfg = AdapterConfig(name=source, split="test", limit=limit, budget_grid=budget_grid, kwargs=kwargs or {})
    adapter = get_adapter(source, cfg)
    try:
        return adapter.load_tasks()
    except RuntimeError as exc:
        raise RuntimeError(f"Could not build {source} paper split. {exc}") from exc


def _small_tasks(name: str) -> list[TaskRecord]:
    if name == "paper_math_core":
        rows = _sample_existing_by_source("data/processed/heldout_math_mix_60.jsonl", per_source=3)
        if rows:
            tasks = [TaskRecord.model_validate(row) for row in rows]
            for task in tasks:
                task.budget_grid = BUDGETS["gsm8k"] if task.source == "gsm8k" else BUDGETS["math"]
            return _retag(tasks, split_name=name, role="small_smoke")
        return _retag(_toy_math_tasks(), split_name=name, role="small_smoke")
    if name == "paper_evalplus_humaneval_full":
        rows = _read_existing("data/processed/evalplus_humaneval_20.jsonl", limit=3)
        if rows:
            tasks = [TaskRecord.model_validate(row) for row in rows]
            for task in tasks:
                task.budget_grid = BUDGETS["humaneval"]
            return _retag(tasks, split_name=name, role="small_smoke")
        return _retag(_toy_coding_tasks("evalplus_humaneval_smoke", BUDGETS["humaneval"]), split_name=name, role="small_smoke")
    if name == "paper_evalplus_mbpp_full":
        return _retag(_toy_coding_tasks("evalplus_mbpp_smoke", BUDGETS["mbpp"]), split_name=name, role="small_smoke")
    if name == "paper_livecodebench_fresh_small":
        return _retag(_toy_coding_tasks("livecodebench_smoke", BUDGETS["livecodebench"]), split_name=name, role="configured_only_smoke")
    if name == "paper_livecodebench_fresh_300":
        return _retag(_toy_coding_tasks("livecodebench_300_smoke", BUDGETS["livecodebench"]), split_name=name, role="configured_only_smoke")
    if name == "paper_repeatability_small":
        return _retag(_repeatability_tasks(output_role="repeatability_smoke", small=True), split_name=name, role="repeatability_smoke")
    if name == "paper_bfcl_lite":
        return _retag(_toy_agentic_tasks(), split_name=name, role="small_smoke")
    if name == "swe_verified_smoke":
        return _retag(_toy_swe_tasks(), split_name=name, role="small_smoke")
    if name == "paper_bigcodebench_hard":
        return _retag(_toy_bigcodebench_hard_tasks(), split_name=name, role="small_smoke")
    if name == "paper_canitedit_descriptive":
        return _retag(_toy_canitedit_tasks(), split_name=name, role="small_smoke")
    raise KeyError(name)


def _retag(tasks: Iterable[TaskRecord], *, split_name: str, role: str) -> list[TaskRecord]:
    result: list[TaskRecord] = []
    for index, task in enumerate(tasks):
        task.metadata = {
            **task.metadata,
            "paper_split": split_name,
            "paper_role": role,
            "paper_order": index,
        }
        result.append(task)
    return result


def _repeatability_tasks(*, output_role: str, small: bool = False) -> list[TaskRecord]:
    math_limit = 4 if small else 40
    humaneval_limit = 3 if small else 30
    mbpp_limit = 3 if small else 30
    tasks: list[TaskRecord] = []
    tasks.extend(_load_repeatability_math(math_limit))
    tasks.extend(_load_repeatability_coding("data/processed/paper_evalplus_humaneval_full.jsonl", humaneval_limit))
    tasks.extend(_load_repeatability_coding("data/processed/paper_evalplus_mbpp_full.jsonl", mbpp_limit))
    if not tasks and small:
        tasks.extend(_toy_math_tasks())
        tasks.extend(_toy_coding_tasks("evalplus_humaneval_smoke", BUDGETS["repeatability"]))
        tasks.extend(_toy_coding_tasks("evalplus_mbpp_smoke", BUDGETS["repeatability"]))
    for index, task in enumerate(tasks):
        task.budget_grid = BUDGETS["repeatability"]
        task.metadata = {
            **task.metadata,
            "paper_split": "paper_repeatability_small",
            "paper_role": output_role,
            "paper_order": index,
            "repeatability_budget_grid": BUDGETS["repeatability"],
        }
    return tasks


def _load_repeatability_math(limit: int) -> list[TaskRecord]:
    rows = _read_existing("data/processed/paper_math_core.jsonl", limit=100000)
    tasks: list[TaskRecord] = []
    for row in rows:
        task = TaskRecord.model_validate(row)
        if task.source == "gsm8k" or classify_math_answer(task.answer) in {"numeric", "fraction"}:
            tasks.append(task)
        if len(tasks) >= limit:
            break
    return tasks


def _load_repeatability_coding(path: str | Path, limit: int) -> list[TaskRecord]:
    rows = _read_existing(path, limit=limit)
    tasks = [TaskRecord.model_validate(row) for row in rows]
    for task in tasks:
        task.budget_grid = BUDGETS["repeatability"]
    return tasks


def _read_existing(path: str | Path, *, limit: int) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return read_jsonl(path)[:limit]


def _task_records_from_existing(path: str | Path, *, limit: int) -> list[TaskRecord]:
    return [TaskRecord.model_validate(row) for row in _read_existing(path, limit=limit)]


def _sample_existing_by_source(path: str | Path, *, per_source: int) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    grouped: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        source = str(row.get("source") or "unknown")
        grouped.setdefault(source, [])
        if len(grouped[source]) < per_source:
            grouped[source].append(row)
    rows: list[dict] = []
    for source in sorted(grouped):
        rows.extend(grouped[source])
    return rows


def _toy_math_tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="paper_math_smoke_001",
            track="math",
            source="local_smoke",
            source_version="local_smoke_v1",
            prompt="Compute 2+2.",
            answer="4",
            verifier="numeric_exact",
            budget_grid=BUDGETS["gsm8k"],
        )
    ]


def _toy_coding_tasks(source: str, budget_grid: list[int]) -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id=f"{source}_001",
            track="coding",
            source=source,
            source_version="local_smoke_v1",
            prompt="Write a Python function add_one(x) that returns x + 1.",
            verifier="python_unit_test",
            budget_grid=budget_grid,
            metadata={"tests": "assert add_one(0) == 1\nassert add_one(41) == 42"},
            external_eval={"harness": "local_smoke_only", "source": source},
        )
    ]


def _toy_bigcodebench_hard_tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="bigcodebench_hard_smoke_001",
            track="coding",
            source="bigcodebench_hard",
            source_version="local_smoke_v1",
            external_id="smoke_001",
            prompt="Write a Python function add_one(x) that returns x + 1.",
            verifier="bigcodebench",
            budget_grid=BUDGETS["bigcodebench_hard"],
            metadata={
                "original_task_id": "smoke_001",
                "split_name": "small_smoke",
                "tests": "assert add_one(0) == 1\nassert add_one(41) == 42",
                "chat_completion_compatible": True,
                "requires_docker": False,
            },
            external_eval={"harness": "bigcodebench", "task_id": "smoke_001"},
        )
    ]


def _toy_canitedit_tasks() -> list[TaskRecord]:
    before = "def add_one(x):\n    return x\n"
    instruction = "Change add_one so it returns x + 1."
    return [
        TaskRecord(
            task_id="canitedit_descriptive_smoke_001",
            track="code_editing",
            source="canitedit",
            source_version="local_smoke_v1",
            external_id="smoke_001",
            prompt=(
                "Original code:\n"
                f"{before}\n"
                "Edit instruction:\n"
                f"{instruction}\n"
                "Return only the complete edited Python program."
            ),
            verifier="canitedit",
            budget_grid=BUDGETS["canitedit"],
            metadata={
                "instruction_style": "descriptive",
                "before_code": before,
                "instruction": instruction,
                "after_code": "def add_one(x):\n    return x + 1\n",
                "tests": "assert add_one(0) == 1\nassert add_one(41) == 42",
                "paper_role": "small_smoke",
                "chat_completion_compatible": True,
                "requires_docker": False,
                "official_harness_status": "provided_tests_available",
            },
            external_eval={"harness": "canitedit", "task_id": "smoke_001", "instruction_style": "descriptive"},
        )
    ]


def _toy_aider_polyglot_tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="aider_polyglot_python_add_one",
            track="coding_edit",
            source="aider_polyglot",
            source_version="local_smoke_v1",
            external_id="python/add_one",
            prompt=(
                "Edit add_one.py so tests pass. Output JSON with files containing complete replacement contents."
            ),
            verifier="aider_polyglot_tests",
            budget_grid=BUDGETS["aider_polyglot"],
            metadata={"language": "python", "exercise": "add_one"},
            external_eval={
                "harness": "aider_polyglot_tests",
                "language": "python",
                "source_root": "external/polyglot-benchmark-smoke",
                "exercise_dir": "python/add_one",
                "allowed_source_files": ["add_one.py"],
                "test_files": ["test_add_one.py"],
                "test_command": ["pytest", "-q"],
            },
        )
    ]


def _toy_agentic_tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="bfcl_lite_smoke_001",
            track="agentic",
            source="bfcl_smoke",
            source_version="local_smoke_v1",
            prompt="Call the add tool with a=2 and b=2.",
            answer="4",
            verifier="numeric_exact",
            budget_grid=BUDGETS["bfcl"],
            external_eval={"harness": "local_smoke_only", "source": "bfcl"},
        )
    ]


def _toy_swe_tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="swe_verified_smoke_001",
            track="swe",
            source="swebench_smoke",
            source_version="local_smoke_v1",
            prompt="Repository: local/example\nIssue: Update add_one so it returns x + 1.\nProduce a patch.",
            verifier="swebench",
            budget_grid=BUDGETS["swe"],
            external_eval={"harness": "swebench_export_only", "source": "swebench"},
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen TokenCapBench paper task splits.")
    parser.add_argument(
        "--track",
        default="all",
        choices=[
            "all",
            "math",
            "coding",
            "fresh_coding",
            "fresh_coding_300",
            "livecodebench",
            "livecodebench_300",
            "repeatability",
            "agentic",
            "swe",
            "replacement",
            "replacement_benchmarks",
            "provider_gateway",
            "bigcodebench_hard",
            "canitedit_descriptive",
        ],
    )
    parser.add_argument("--small", action="store_true", help="Create cheap local smoke splits.")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--bfcl-path", default=None, help="Local BFCL JSONL/JSON export for the full BFCL split.")
    args = parser.parse_args()
    counts = build_splits(track=args.track, small=args.small, output_dir=args.output_dir, bfcl_path=args.bfcl_path)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
