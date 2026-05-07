import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.package_run_artifacts import package_release_archive, package_run_artifacts


def test_package_run_artifacts_writes_manifest_and_hashes(tmp_path):
    run_dir = tmp_path / "reports" / "runs" / "tiny_run"
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "p_success_by_budget": {"64": 0.5}, "median_budget2success": 64}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "t1", "budget": 64, "success": True, "model": "m", "metadata": {"track": "math"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"brier": 0.25}), encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "run_id": "tiny_run",
                "task_file": "missing.jsonl",
                "provider": "mock",
                "model": "mock-model",
                "budget_grid": {"math": [64]},
                "api_key": "do-not-copy",
            }
        ),
        encoding="utf-8",
    )

    artifact_dir = package_run_artifacts(run_dir, artifact_root=tmp_path / "reports" / "artifacts")

    manifest = json.loads((artifact_dir / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((artifact_dir / "sha256_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "tiny_run"
    assert manifest["config"]["api_key"] == "<redacted>"
    assert "forecasts.jsonl" in hashes
    assert (artifact_dir / "config_snapshot.yaml").exists()


def test_package_run_artifacts_names_nested_suite_artifacts_uniquely(tmp_path):
    first = _write_minimal_run(tmp_path / "reports" / "runs" / "suite_a" / "same_model")
    second = _write_minimal_run(tmp_path / "reports" / "runs" / "suite_b" / "same_model")
    artifact_root = tmp_path / "reports" / "artifacts"

    first_artifact = package_run_artifacts(first, artifact_root=artifact_root)
    second_artifact = package_run_artifacts(second, artifact_root=artifact_root)

    assert first_artifact.name == "suite_a__same_model"
    assert second_artifact.name == "suite_b__same_model"
    assert first_artifact != second_artifact


def test_package_release_archive_includes_required_bundle_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "reports" / "artifacts" / "run"
    artifact.mkdir(parents=True)
    for name in ["forecasts.jsonl", "outcomes.jsonl", "metrics.json"]:
        (artifact / name).write_text("{}\n", encoding="utf-8")
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tables / "paper_table_clean_evidence_scope.csv").write_text("track,source\nmath,GSM8K + MATH\n", encoding="utf-8")
    figures = tmp_path / "reports" / "figures"
    figures.mkdir(parents=True)
    (figures / "paper_figure1_pipeline.png").write_bytes(b"png")
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "paper.yaml").write_text("run_id: paper\n", encoding="utf-8")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "forecast_prompt.md").write_text("prompt\n", encoding="utf-8")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "croissant.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "reports" / "tokencapbench_release_archive.zip"
    archive = package_release_archive(
        artifact_root=artifact.parent,
        tables_dir=tables,
        figures_dir=figures,
        metadata_dir=metadata,
        output=output,
    )

    assert archive.exists()
    assert archive.with_suffix(archive.suffix + ".sha256_manifest.json").exists()
    listing = subprocess.run(
        [sys.executable, "-m", "zipfile", "-l", str(archive)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "reports/artifacts/run/forecasts.jsonl" in listing
    assert "reports/tables/paper_table_clean_evidence_scope.csv" in listing
    assert "metadata/croissant.json" in listing


def test_run_budget_grid_repeat_ids_land_in_outcomes_and_manifest(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "track": "math",
                "prompt": "Compute 2+2.",
                "verifier": "numeric_exact",
                "answer": "4",
                "source": "gsm8k",
                "budget_grid": [64],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run_id": "repeat_run",
                "task_file": str(task_file),
                "output_dir": str(tmp_path / "runs"),
                "provider": "mock",
                "model": "mock-model",
                "budget_grid": {"math": [64]},
                "metadata": {"repeat_ids": [1, 2]},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "scripts/run_budget_grid.py", "--config", str(config)], check=True)

    run_dir = tmp_path / "runs" / "repeat_run"
    outcomes = [json.loads(line) for line in (run_dir / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["metadata"]["repeat_id"] for row in outcomes} == {"1", "2"}
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["repeat_ids"] == ["1", "2"]
    assert manifest["repeats"] == 2


def _write_minimal_run(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "p_success_by_budget": {"64": 0.5}, "median_budget2success": 64}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "t1", "budget": 64, "success": True, "model": "m", "metadata": {"track": "math"}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"brier": 0.25}), encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({"run_id": run_dir.name, "provider": "mock", "model": "mock-model"}),
        encoding="utf-8",
    )
    return run_dir
