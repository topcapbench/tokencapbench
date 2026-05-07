from budget2success.execution.aider_polyglot_bridge import CodeEditJSONParser


def test_code_edit_json_parser_accepts_complete_file_edits():
    edits = CodeEditJSONParser().parse('{"files":[{"path":"src/app.py","content":"print(1)\\n"}]}')

    assert edits[0].path == "src/app.py"
    assert edits[0].content == "print(1)\n"


def test_code_edit_json_parser_rejects_path_traversal():
    try:
        CodeEditJSONParser().parse('{"files":[{"path":"../test.py","content":""}]}')
    except ValueError as exc:
        assert "traverse" in str(exc)
        return
    raise AssertionError("Expected path traversal rejection")
