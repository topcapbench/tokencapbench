from budget2success.datasets.aider_polyglot import AiderPolyglotAdapter
from budget2success.datasets.base import AdapterConfig


def test_aider_polyglot_loader_scans_toy_python_exercise(tmp_path):
    exercise = tmp_path / "python" / "add-one"
    exercise.mkdir(parents=True)
    (exercise / "README.md").write_text("Implement add_one.", encoding="utf-8")
    (exercise / "add_one.py").write_text("def add_one(x):\n    pass\n", encoding="utf-8")
    (exercise / "test_add_one.py").write_text("from add_one import add_one\n\ndef test_add_one():\n    assert add_one(1) == 2\n", encoding="utf-8")
    adapter = AiderPolyglotAdapter(
        AdapterConfig(name="aider_polyglot", limit=5, kwargs={"source_root": str(tmp_path), "languages": "python"})
    )

    tasks = adapter.load_tasks()

    assert len(tasks) == 1
    assert tasks[0].track == "coding_edit"
    assert tasks[0].verifier == "aider_polyglot_tests"
    assert tasks[0].external_eval["allowed_source_files"] == ["add_one.py"]
    assert "Output only JSON" in tasks[0].prompt
