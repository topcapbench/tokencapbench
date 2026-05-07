from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_NAME = "score_results.py"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / SCRIPT_NAME
    if not script.exists():
        raise SystemExit(f"Cannot find {script}. Run from a source checkout or use the scripts/ entry point directly.")
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
