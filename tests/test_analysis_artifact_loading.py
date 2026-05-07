import json

from budget2success.analysis.paper import discover_artifact_dirs, infer_suite_from_artifact_dir, load_paper_runs


def _write_run(path, suite="paper_math_core", run_id="tiny", model="m"):
    path.mkdir(parents=True)
    (path / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": model, "p_success_by_budget": {"64": 0.4}}) + "\n",
        encoding="utf-8",
    )
    (path / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": model, "budget": 64, "success": True}) + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text("{}", encoding="utf-8")
    (path / "config_snapshot.yaml").write_text(
        f"run_id: {run_id}\nmodel: {model}\nmetadata:\n  suite_name: {suite}\n",
        encoding="utf-8",
    )


def test_load_paper_runs_discovers_packaged_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__tiny"
    _write_run(artifact_dir)

    runs = load_paper_runs(
        suite="paper_math_core",
        run_root=tmp_path / "missing_runs",
        artifact_root=tmp_path / "artifacts",
    )

    assert len(runs) == 1
    assert runs[0].suite == "paper_math_core"
    assert runs[0].artifact_source == "artifacts"
    assert runs[0].run_dir == artifact_dir.resolve()


def test_discover_artifact_dirs_filters_by_inferred_suite(tmp_path):
    keep = tmp_path / "artifacts" / "paper_math_core__tiny"
    skip = tmp_path / "artifacts" / "paper_evalplus_mbpp_full__tiny"
    _write_run(keep, suite="paper_math_core")
    _write_run(skip, suite="paper_evalplus_mbpp_full")

    assert discover_artifact_dirs("paper_math_core", tmp_path / "artifacts") == [keep.resolve()]
    assert infer_suite_from_artifact_dir(skip) == "paper_evalplus_mbpp_full"


def test_load_paper_runs_strict_math_labels_replace_outcomes(tmp_path):
    artifact_dir = tmp_path / "artifacts" / "paper_math_core__tiny"
    corrected_dir = tmp_path / "corrected" / "paper_math_core__tiny"
    _write_run(artifact_dir)
    _write_run(corrected_dir)
    (corrected_dir / "outcomes.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "budget": 64, "success": False, "metadata": {"track": "math"}}) + "\n",
        encoding="utf-8",
    )

    runs = load_paper_runs(
        suite="paper_math_core",
        run_root=tmp_path / "missing_runs",
        artifact_root=tmp_path / "artifacts",
        corrected_artifact_root=tmp_path / "corrected",
        math_label_mode="strict",
    )

    assert runs[0].outcomes[0]["success"] is False
