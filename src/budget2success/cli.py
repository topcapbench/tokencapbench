from __future__ import annotations

import argparse
import sys
from pathlib import Path

from budget2success.validate_name import find_legacy_name_hits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tokencapbench", description="TokenCapBench benchmark utilities.")
    sub = parser.add_subparsers(dest="command")
    name_parser = sub.add_parser("validate-name", help="Check that the legacy benchmark name is absent from text files.")
    name_parser.add_argument("--root", default=".")
    sub.add_parser("version", help="Print package version.")
    args = parser.parse_args(argv)

    if args.command == "validate-name":
        hits = find_legacy_name_hits(Path(args.root))
        if hits:
            print("\n".join(hits))
            raise SystemExit(1)
        print("OK: no legacy benchmark name found")
        return
    if args.command == "version":
        try:
            from importlib.metadata import version
            print(version("budget2success"))
        except Exception:  # noqa: BLE001 - installed metadata may be absent in editable-free smoke tests.
            print("budget2success")
        return
    parser.print_help(sys.stderr)


if __name__ == "__main__":
    main()
