import csv
import json

from scripts.make_paper_tables import make_paper_tables


def test_fresh_coding_table_marks_placeholder_unverified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "reports" / "artifacts" / "paper_livecodebench_fresh_small__m"
    artifact.mkdir(parents=True)
    (artifact / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.5}}) + "\n",
        encoding="utf-8",
    )
    (artifact / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t1",
                "model": "m",
                "budget": 64,
                "success": False,
                "verification": {"status": "error", "success": False, "details": {}, "metadata": {}},
                "metadata": {"label_source": "official_harness_placeholder", "exclude_from_main_metrics": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (artifact / "metrics.json").write_text("{}", encoding="utf-8")
    (artifact / "config_snapshot.yaml").write_text(
        "suite_name: paper_livecodebench_fresh_small\nmodel: m\nrun_id: m\n",
        encoding="utf-8",
    )
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tables / "baseline_comparison.csv").write_text("", encoding="utf-8")

    make_paper_tables(
        suite="paper_livecodebench_fresh_small",
        artifact_root=tmp_path / "reports" / "artifacts",
        table_dir=tables,
    )

    with (tables / "paper_table11_fresh_coding.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["official_harness_status"] == "placeholder_unverified"


def test_fresh_coding_table_prefers_official_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    official = tmp_path / "reports" / "artifacts_livecodebench_official" / "m"
    official.mkdir(parents=True)
    (official / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.5}}) + "\n",
        encoding="utf-8",
    )
    (official / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t1",
                "model": "m",
                "budget": 64,
                "success": True,
                "verification": {"status": "success", "success": True, "details": {}, "metadata": {"label_source": "official_livecodebench"}},
                "metadata": {"source": "livecodebench", "label_source": "official_livecodebench"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (official / "metrics.json").write_text("{}", encoding="utf-8")
    (official / "config_snapshot.yaml").write_text(
        "suite_name: paper_livecodebench_fresh_small\nmodel: m\nrun_id: m\n",
        encoding="utf-8",
    )
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    (tables / "baseline_comparison.csv").write_text("", encoding="utf-8")

    make_paper_tables(
        suite="paper_livecodebench_fresh_small",
        artifact_root=tmp_path / "reports" / "artifacts",
        table_dir=tables,
        official_artifact_roots=[tmp_path / "reports" / "artifacts_livecodebench_official"],
    )

    with (tables / "paper_table11_fresh_coding.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["official_harness_status"] == "completed"
