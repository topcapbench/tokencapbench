from pathlib import Path


def _decode(hex_text: str) -> str:
    return bytes.fromhex(hex_text).decode("utf-8")


# Encoded to keep this guard test from reintroducing banned strings into the repo.
FORBIDDEN_TERMS = [
    _decode("5265736f75726365477265656e42656e6368"),
    _decode("7265736f75726365677265656e62656e6368"),
    _decode("477265656e42656e6368"),
    _decode("746f6b656e5f746f5f677265656e"),
    _decode("546f6b656e2d746f2d477265656e"),
    _decode("746f6b656e2d746f2d677265656e"),
    _decode("5374616e666f7264"),
    _decode("5354414e464f5244"),
    _decode("7374616e666f7264"),
]

TEXT_SUFFIXES = {".py", ".md", ".tex", ".bib", ".yml", ".yaml", ".toml", ".cff", ".csv", ".json", ".txt", ".svg"}
EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    # Unrelated untracked worktree present in some local workspaces.
    "SkipTranscoderSAEBench-transcoders",
}


def test_submission_text_has_no_stale_public_names():
    hits: list[str] = []
    for path in Path(".").rglob("*"):
        if path.is_dir() or any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in FORBIDDEN_TERMS:
            if term in text:
                hits.append(f"{path}: {term}")
    assert not hits, "Forbidden stale public names found:\n" + "\n".join(hits[:50])


def test_neurips_paper_uses_tokencapbench_and_eandd_style():
    paper = Path("paper/neurips2026_tokencapbench.tex").read_text(encoding="utf-8")
    assert "\\usepackage[eandd]{neurips_2026}" in paper
    assert "TokenCapBench" in paper
    for term in FORBIDDEN_TERMS:
        assert term not in paper
