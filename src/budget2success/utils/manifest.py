from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from budget2success.utils.jsonl import read_jsonl


SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
)
SECRET_KEY_EXACT = {
    "access_token",
    "auth_token",
    "bearer_token",
    "client_secret",
    "refresh_token",
    "token",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in SECRET_KEY_EXACT or any(part in key_text for part in SECRET_KEY_PARTS):
                redacted[str(key)] = "<redacted>"
            else:
                redacted[str(key)] = redact_secrets(child)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(child) for child in value]
    return value


def load_redacted_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return redact_secrets(data)


def write_redacted_config_snapshot(config: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(redact_secrets(config), sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def prompt_hashes(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    forecast_prompt = config.get("forecast_prompt")
    if forecast_prompt and Path(forecast_prompt).exists():
        path = Path(forecast_prompt)
        hashes["forecast"] = {"path": str(path), "sha256": sha256_file(path)}
    solver_prompts = config.get("solver_prompts") or {}
    if isinstance(solver_prompts, dict):
        for track, prompt_path in sorted(solver_prompts.items()):
            if prompt_path and Path(prompt_path).exists():
                path = Path(prompt_path)
                hashes[f"solver_{track}"] = {"path": str(path), "sha256": sha256_file(path)}
    return hashes


def dataset_summary(task_file: str | Path | None) -> dict[str, Any]:
    if not task_file:
        return {}
    path = Path(task_file)
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary
    summary["sha256"] = sha256_file(path)
    try:
        rows = read_jsonl(path)
    except Exception as exc:
        summary["read_error"] = str(exc)
        return summary
    sources: dict[str, dict[str, Any]] = {}
    budget_grids: dict[str, set[tuple[int, ...]]] = {}
    for row in rows:
        source = str(row.get("source") or "unknown")
        source_entry = sources.setdefault(
            source,
            {"tasks": 0, "source_versions": set(), "tracks": set(), "verifiers": set()},
        )
        source_entry["tasks"] += 1
        if row.get("source_version"):
            source_entry["source_versions"].add(str(row["source_version"]))
        if row.get("track"):
            source_entry["tracks"].add(str(row["track"]))
        if row.get("verifier"):
            source_entry["verifiers"].add(str(row["verifier"]))
        grid = tuple(int(b) for b in row.get("budget_grid") or [])
        if grid:
            budget_grids.setdefault(source, set()).add(grid)
    normalized_sources: dict[str, dict[str, Any]] = {}
    for source, entry in sorted(sources.items()):
        normalized_sources[source] = {
            "tasks": entry["tasks"],
            "source_versions": sorted(entry["source_versions"]),
            "tracks": sorted(entry["tracks"]),
            "verifiers": sorted(entry["verifiers"]),
        }
    summary["tasks"] = len(rows)
    summary["sources"] = normalized_sources
    summary["budget_grids_by_source"] = {
        source: [list(grid) for grid in sorted(grids)] for source, grids in sorted(budget_grids.items())
    }
    return summary


def artifact_file_hashes(run_dir: str | Path) -> dict[str, str]:
    run_path = Path(run_dir)
    hashes: dict[str, str] = {}
    for path in sorted(run_path.iterdir()) if run_path.exists() else []:
        if path.is_file() and path.name not in {"run_manifest.json"}:
            hashes[path.name] = sha256_file(path)
    return hashes


def write_run_manifest(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    command_line_arguments: list[str] | None = None,
    phase: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    task_file = config.get("task_file")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config.get("run_id") or run_path.name,
        "phase": phase,
        "code_commit": git_commit(),
        "command_line_arguments": command_line_arguments or [],
        "provider": config.get("provider"),
        "model_ids": _model_ids(config),
        "scaffold": config.get("scaffold"),
        "budget_grids": config.get("budget_grid") or config.get("budget_grids") or {},
        "dataset": dataset_summary(task_file),
        "prompt_hashes": prompt_hashes(config),
        "artifact_hashes": artifact_file_hashes(run_path),
        "config": redact_secrets(config),
    }
    if extra:
        manifest.update(redact_secrets(extra))
    out_path = run_path / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _model_ids(config: dict[str, Any]) -> list[str]:
    if config.get("model"):
        return [str(config["model"])]
    models = config.get("models") or []
    if isinstance(models, list):
        result: list[str] = []
        for model in models:
            if isinstance(model, dict):
                value = model.get("name") or model.get("model")
            else:
                value = model
            if value:
                result.append(str(value))
        return result
    return []
