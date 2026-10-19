"""Codex hook entry point: append one privacy-minimized event per invocation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

POC_ROOT = Path(__file__).resolve().parents[1]
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from codex_telemetry.adapter import normalize_hook_event


DEFAULT_OUTPUT = POC_ROOT / "results" / "live" / "codex_events.jsonl"


def capture_payload(raw_json: str, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = json.loads(raw_json)
    record = normalize_hook_event(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)
    return record


def main() -> int:
    try:
        capture_payload(sys.stdin.read())
    except Exception as exc:
        # Do not echo the hook payload, which can contain private command data.
        print(f"Codex telemetry capture failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
