"""Summaries for captured, privacy-minimized Codex hook event JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    hooks = Counter(row.get("hook_event_name", "unknown") for row in rows)
    tools = Counter(
        row.get("tool_family", "none")
        for row in rows
        if row.get("hook_event_name") in {"PreToolUse", "PostToolUse"}
    )
    outcomes = Counter(
        row.get("outcome", "unknown")
        for row in rows
        if row.get("hook_event_name") == "PostToolUse"
    )

    ordered_times = sorted(_timestamp(row["observed_at"]) for row in rows)
    summary: dict[str, Any] = {
        "total_events": len(rows),
        "sessions": len({row["session_key"] for row in rows if row.get("session_key")}),
        "hook_events": dict(sorted(hooks.items())),
        "tool_families": dict(sorted(tools.items())),
        "post_tool_outcomes": dict(sorted(outcomes.items())),
        "first_event_at": ordered_times[0].isoformat() if ordered_times else None,
        "last_event_at": ordered_times[-1].isoformat() if ordered_times else None,
    }
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Expected an event object on line {line_number} of {path}.")
            events.append(event)
    return events
