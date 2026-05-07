from __future__ import annotations

from pathlib import Path

from budget2success.data.task_schema import TaskRecord


def build_forecast_prompt(template_path: str | Path, task: TaskRecord, budget_grid: list[int], scaffold: str) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    return (
        f"{template}\n\n"
        f"Solver scaffold: {scaffold}\n"
        f"Token budgets: {budget_grid}\n\n"
        f"Task ID: {task.task_id}\n"
        f"Track: {task.track}\n"
        f"Source: {task.source or 'unknown'}\n"
        f"Verifier: {task.verifier}\n\n"
        f"Task:\n{task.prompt}\n"
    )


def build_solver_prompt(template_path: str | Path, task: TaskRecord, scaffold: str) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    return (
        f"{template}\n\n"
        f"Solver scaffold: {scaffold}\n"
        f"Task ID: {task.task_id}\n"
        f"Track: {task.track}\n\n"
        f"Task:\n{task.prompt}\n"
    )
