#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.clients.base import GenerationRequest
from budget2success.clients.chat_gateway import DEFAULT_GATEWAY_BASE_URL, ChatGatewayClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal provider gateway chat-completions smoke call.")
    parser.add_argument("--model", default="gemini-2.0-flash-lite-001")
    parser.add_argument("--base-url", default=DEFAULT_GATEWAY_BASE_URL)
    parser.add_argument(
        "--api-key-file",
        default="provider_gateway/api_key.txt",
        help="Optional local key file. BUDGET2SUCCESS_GATEWAY_API_KEY takes precedence.",
    )
    parser.add_argument("--prompt", default="Reply with exactly: ok")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    api_key = os.getenv("BUDGET2SUCCESS_GATEWAY_API_KEY") or _read_key(args.api_key_file)
    client = ChatGatewayClient(api_key=api_key, base_url=args.base_url)
    response = client.generate(
        GenerationRequest(model=args.model, prompt=args.prompt, max_tokens=args.max_tokens, temperature=0.0)
    )
    print(json.dumps({"status": "ok", "model": args.model, "finish_reason": response.finish_reason}))


def _read_key(path: str | Path) -> str | None:
    key_path = Path(path)
    if not key_path.exists():
        return None
    value = key_path.read_text(encoding="utf-8").strip()
    return value or None


if __name__ == "__main__":
    main()
