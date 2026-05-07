from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from budget2success.execution.external_harness import run_command
from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class FileEdit:
    path: str
    content: str


class CodeEditJSONParser:
    """Parse complete-file JSON edits from solver output."""

    def parse(self, text: str) -> list[FileEdit]:
        payload = _extract_json_object(text)
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("solver JSON must contain a non-empty files array")
        edits: list[FileEdit] = []
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                raise ValueError(f"files[{index}] must be an object")
            raw_path = item.get("path")
            content = item.get("content")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"files[{index}].path must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError(f"files[{index}].content must be a string")
            path = _normalize_relative_path(raw_path)
            edits.append(FileEdit(path=path, content=content))
        return edits


class AiderPolyglotVerifier(Verifier):
    """Verify JSON file edits against an Aider Polyglot exercise's tests."""

    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout_seconds = timeout_seconds
        self.parser = CodeEditJSONParser()

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        bridge = AiderPolyglotBridge(timeout_seconds=self.timeout_seconds)
        return bridge.verify(task, solution)


class AiderPolyglotBridge:
    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout_seconds = timeout_seconds
        self.parser = CodeEditJSONParser()

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        metadata = task.external_eval or task.metadata or {}
        try:
            source_root, exercise_dir = _exercise_paths(metadata)
            allowed = {str(path) for path in metadata.get("allowed_source_files") or []}
            if not allowed:
                raise ValueError("Aider Polyglot task is missing allowed_source_files metadata")
            test_command = [str(part) for part in metadata.get("test_command") or []]
            if not test_command:
                raise ValueError("Aider Polyglot task is missing test_command metadata")
            edits = self.parser.parse(solution)
            edited_paths = {edit.path for edit in edits}
            disallowed = sorted(edited_paths - allowed)
            if disallowed:
                return VerificationResult.error(
                    error="disallowed_file_edit",
                    disallowed_paths=disallowed,
                    allowed_source_files=sorted(allowed),
                    metadata={"label_source": "aider_polyglot_tests"},
                )
            with tempfile.TemporaryDirectory(prefix="ttg_aider_polyglot_") as tmp:
                workdir = Path(tmp) / "exercise"
                shutil.copytree(exercise_dir, workdir, dirs_exist_ok=True)
                for edit in edits:
                    target = _resolve_under(workdir, edit.path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(edit.content, encoding="utf-8")
                result = run_command(test_command, cwd=workdir, timeout_seconds=self.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - verifier failures should be explicit rows.
            return VerificationResult.error(
                error="aider_polyglot_verifier_error",
                message=str(exc),
                metadata={"label_source": "aider_polyglot_tests"},
            )

        details = {
            "harness": "aider_polyglot_tests",
            "command": test_command,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "stdout_snippet": _snippet(result.stdout),
            "stderr_snippet": _snippet(result.stderr),
        }
        if result.success:
            return VerificationResult.ok(**details, metadata={"label_source": "aider_polyglot_tests"})
        return VerificationResult.fail(**details, metadata={"label_source": "aider_polyglot_tests"})


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [*list(_FENCED_BLOCK_RE.findall(stripped)), stripped]
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                payload, _end = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if isinstance(payload, dict):
                return payload
    detail = f": {'; '.join(errors[:3])}" if errors else ""
    raise ValueError(f"No JSON object found in solver response{detail}")


def _normalize_relative_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"edit path must be relative: {raw_path}")
    parts = path.parts
    if any(part in {"..", ""} for part in parts):
        raise ValueError(f"edit path cannot traverse directories: {raw_path}")
    return path.as_posix()


def _exercise_paths(metadata: dict[str, Any]) -> tuple[Path, Path]:
    source_root = Path(str(metadata.get("source_root") or ""))
    exercise_rel = Path(str(metadata.get("exercise_dir") or ""))
    if not source_root.exists():
        raise FileNotFoundError(f"Aider Polyglot source_root not found: {source_root}")
    exercise_dir = _resolve_under(source_root, exercise_rel.as_posix())
    if not exercise_dir.exists():
        raise FileNotFoundError(f"Aider Polyglot exercise_dir not found: {exercise_dir}")
    return source_root, exercise_dir


def _resolve_under(root: Path, relative_path: str) -> Path:
    target = (root / _normalize_relative_path(relative_path)).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"path escapes exercise directory: {relative_path}")
    return target


def _snippet(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
