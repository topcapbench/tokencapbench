import json

from budget2success.analysis.paper import load_paper_runs
from budget2success.execution.verifier_registry import OfficialHarnessRequiredVerifier
from budget2success.schemas.records import TaskRecord


def test_official_harness_placeholder_metadata_is_explicit():
    task = TaskRecord(
        task_id="lcb_1",
        track="coding",
        prompt="p",
        verifier="livecodebench",
        external_id="q1",
        external_eval={"harness": "livecodebench"},
    )

    result = OfficialHarnessRequiredVerifier("livecodebench").verify(task, "")

    assert result.success is False
    assert result.metadata["label_source"] == "official_harness_placeholder"
    assert result.metadata["exclude_from_main_metrics"] is True
    assert result.metadata["official_harness_required"] == "livecodebench"


def test_load_paper_runs_excludes_placeholder_outcomes_without_official_labels(tmp_path):
    run_dir = tmp_path / "artifacts" / "paper_livecodebench_fresh_small__m"
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps({"task_id": "t1", "model": "m", "p_success_by_budget": {"64": 0.5}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "outcomes.jsonl").write_text(
        json.dumps(
            {
                "task_id": "t1",
                "model": "m",
                "budget": 64,
                "success": False,
                "verification": {
                    "status": "error",
                    "success": False,
                    "details": {"error": "official_harness_required"},
                    "metadata": {"exclude_from_main_metrics": True},
                },
                "metadata": {"exclude_from_main_metrics": True, "label_source": "official_harness_placeholder"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "config_snapshot.yaml").write_text(
        "suite_name: paper_livecodebench_fresh_small\nmodel: m\nrun_id: m\n",
        encoding="utf-8",
    )

    runs = load_paper_runs(
        suite="paper_livecodebench_fresh_small",
        run_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts",
        include_artifacts=True,
    )

    assert len(runs) == 1
    assert runs[0].outcomes == []
