from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from budget2success.datasets.base import BenchmarkSourceAdapter
from budget2success.schemas.records import TaskRecord


DEFAULT_BUDGET_GRID = [512, 1024, 2048, 4096]
DEFAULT_LANGUAGES = ["python", "javascript", "rust"]
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".h"],
}
SKIP_DIRS = {".git", "__pycache__", "node_modules", "target", "build", "dist", ".venv", "venv"}


class AiderPolyglotAdapter(BenchmarkSourceAdapter):
    """Scan a local Aider Polyglot checkout into edit-with-tests tasks."""

    source_name = "aider_polyglot"

    def load_tasks(self) -> list[TaskRecord]:
        root = Path(str(self.config.kwargs.get("source_root") or self.config.kwargs.get("path") or "external/polyglot-benchmark"))
        if not root.exists():
            raise RuntimeError(
                f"Aider Polyglot source root not found: {root}. "
                "Clone https://github.com/Aider-AI/polyglot-benchmark.git or pass --source-root."
            )
        languages = _languages(self.config.kwargs.get("languages"))
        limit = self.config.limit
        tasks: list[TaskRecord] = []
        for language in languages:
            for exercise_dir in _discover_exercises(root, language):
                descriptor = _describe_exercise(root, exercise_dir, language)
                if descriptor is None:
                    continue
                task = TaskRecord(
                    task_id=f"aider_polyglot_{language}_{_safe_id(exercise_dir.name)}",
                    source="aider_polyglot",
                    source_version=str(root),
                    track="coding_edit",
                    verifier="aider_polyglot_tests",
                    budget_grid=self.config.budget_grid or DEFAULT_BUDGET_GRID,
                    prompt=_format_prompt(descriptor),
                    external_id=f"{language}/{exercise_dir.name}",
                    metadata={
                        "language": language,
                        "exercise": exercise_dir.name,
                        "source_root": str(root),
                        "exercise_dir": str(exercise_dir.relative_to(root)),
                        "allowed_source_files": descriptor["source_files"],
                        "test_files": descriptor["test_files"],
                        "test_command": descriptor["test_command"],
                    },
                    external_eval={
                        "harness": "aider_polyglot_tests",
                        "language": language,
                        "source_root": str(root),
                        "exercise_dir": str(exercise_dir.relative_to(root)),
                        "allowed_source_files": descriptor["source_files"],
                        "test_files": descriptor["test_files"],
                        "test_command": descriptor["test_command"],
                    },
                )
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    return tasks
        if not tasks:
            raise RuntimeError(f"No Aider Polyglot exercises found under {root} for languages={languages}.")
        return tasks


def _languages(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return DEFAULT_LANGUAGES
    if isinstance(raw, str):
        values = [piece.strip().lower() for piece in raw.split(",")]
    else:
        values = [str(piece).strip().lower() for piece in raw]
    return [value for value in values if value]


def _discover_exercises(root: Path, language: str) -> list[Path]:
    candidates: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_dir() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if language not in {part.lower() for part in path.relative_to(root).parts} and path.name.lower() != language:
            continue
        if _has_tests(path, language):
            candidates.add(path)
    leaves = {path for path in candidates if not any(path != other and path in other.parents for other in candidates)}
    return sorted(leaves, key=lambda item: str(item.relative_to(root)))


def _describe_exercise(root: Path, exercise_dir: Path, language: str) -> dict[str, Any] | None:
    source_files = _source_files(exercise_dir, language)
    test_files = _test_files(exercise_dir, language)
    if not source_files or not test_files:
        return None
    return {
        "root": root,
        "exercise_dir": exercise_dir,
        "language": language,
        "instructions": _read_instructions(exercise_dir),
        "source_files": [str(path.relative_to(exercise_dir)) for path in source_files],
        "test_files": [str(path.relative_to(exercise_dir)) for path in test_files],
        "source_contents": [(str(path.relative_to(exercise_dir)), _read_limited(path)) for path in source_files[:8]],
        "test_command": _test_command(exercise_dir, language),
    }


def _format_prompt(descriptor: dict[str, Any]) -> str:
    lines = [
        "You are editing a programming exercise so that its existing tests pass.",
        "",
        f"Language: {descriptor['language']}",
        "",
        "Instructions:",
        descriptor["instructions"] or "Use the tests and stubs to infer the required behavior.",
        "",
        "Editable source files:",
    ]
    for path, content in descriptor["source_contents"]:
        lines.extend([f"--- {path} ---", content])
    lines.extend(
        [
            "",
            "Output format:",
            '{"files":[{"path":"relative/path/to/file.ext","content":"complete replacement content"}]}',
            "",
            "Output only JSON. Do not include markdown. Include complete file contents for modified files. Do not modify tests.",
        ]
    )
    return "\n".join(lines)


def _read_instructions(exercise_dir: Path) -> str:
    names = ["README.md", "README.rst", "instructions.md", ".docs/instructions.md"]
    chunks: list[str] = []
    for name in names:
        path = exercise_dir / name
        if path.exists() and path.is_file():
            chunks.append(_read_limited(path, limit=6000))
    if chunks:
        return "\n\n".join(chunks)
    return ""


def _source_files(exercise_dir: Path, language: str) -> list[Path]:
    extensions = set(LANGUAGE_EXTENSIONS.get(language, []))
    files = [
        path
        for path in exercise_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not _is_test_file(path, language)
        and not any(part in SKIP_DIRS for part in path.parts)
    ]
    return sorted(files, key=lambda item: str(item.relative_to(exercise_dir)))


def _test_files(exercise_dir: Path, language: str) -> list[Path]:
    return sorted(
        [
            path
            for path in exercise_dir.rglob("*")
            if path.is_file() and _is_test_file(path, language) and not any(part in SKIP_DIRS for part in path.parts)
        ],
        key=lambda item: str(item.relative_to(exercise_dir)),
    )


def _has_tests(path: Path, language: str) -> bool:
    return bool(_test_files(path, language))


def _is_test_file(path: Path, language: str) -> bool:
    name = path.name.lower()
    if language == "python":
        return path.suffix == ".py" and (name.startswith("test_") or name.endswith("_test.py") or "/tests/" in path.as_posix())
    if language == "javascript":
        return any(token in name for token in (".test.", ".spec.", "_test.", "-test."))
    if language == "rust":
        return name.endswith("_test.rs") or "/tests/" in path.as_posix()
    if language == "go":
        return name.endswith("_test.go")
    if language == "java":
        return name.endswith("test.java") or "/test/" in path.as_posix()
    if language == "cpp":
        return "test" in name and path.suffix.lower() in {".cpp", ".cc", ".cxx"}
    return "test" in name


def _test_command(exercise_dir: Path, language: str) -> list[str]:
    if language == "python":
        return ["pytest", "-q"]
    if language == "javascript":
        return ["npm", "test", "--", "--runInBand"] if (exercise_dir / "package.json").exists() else ["npm", "test"]
    if language == "rust":
        return ["cargo", "test", "--quiet"]
    if language == "go":
        return ["go", "test", "./..."]
    if language == "java":
        if (exercise_dir / "gradlew").exists():
            return ["./gradlew", "test"]
        return ["mvn", "test"]
    return []


def _read_limited(path: Path, *, limit: int = 12000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "exercise"
