"""Normalize Codex hook JSON without retaining prompt or tool content."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any


SUPPORTED_HOOKS = {"SessionStart", "SessionEnd", "PreToolUse", "PostToolUse"}
HOOK_TYPES = ("SessionStart", "SessionEnd", "PreToolUse", "PostToolUse")
TOOL_FAMILIES = ("shell", "file_edit", "mcp", "read", "search", "other", "none")
PERMISSION_MODES = ("default", "acceptEdits", "plan", "dontAsk", "bypassPermissions")
OUTCOMES = ("success", "error", "unknown")
SESSION_SOURCES = ("startup", "resume", "clear", "compact")

FEATURE_NAMES = (
    tuple(f"hook_{name}" for name in HOOK_TYPES)
    + tuple(f"tool_{name}" for name in TOOL_FAMILIES)
    + tuple(f"permission_{name}" for name in PERMISSION_MODES)
    + tuple(f"outcome_{name}" for name in OUTCOMES)
    + tuple(f"source_{name}" for name in SESSION_SOURCES)
    + ("input_size_log", "response_size_log")
    + (
        "has_session_id",
        "has_turn_id",
        "has_tool_use_id",
        "has_tool_input",
        "has_tool_response",
        "has_permission_mode",
        "has_session_source",
    )
)


def _identifier_hash(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _json_size(value: Any) -> int:
    if value is None:
        return 0
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _tool_family(tool_name: Any) -> str:
    if not isinstance(tool_name, str) or not tool_name:
        return "none"
    if tool_name == "Bash":
        return "shell"
    if tool_name in {"apply_patch", "Edit", "Write"}:
        return "file_edit"
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name in {"Read", "read_file", "open_file"}:
        return "read"
    if tool_name in {"Glob", "Grep", "Search", "rg"}:
        return "search"
    return "other"


def _outcome(tool_response: Any) -> str:
    """Use explicit machine-readable status only; do not inspect response text."""
    if not isinstance(tool_response, dict):
        return "unknown"

    for key in ("is_error", "isError"):
        value = tool_response.get(key)
        if isinstance(value, bool):
            return "error" if value else "success"

    for key in ("exit_code", "exitCode"):
        value = tool_response.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return "success" if value == 0 else "error"

    value = tool_response.get("success")
    if isinstance(value, bool):
        return "success" if value else "error"
    return "unknown"


def normalize_hook_event(payload: dict[str, Any], observed_at: str | None = None) -> dict[str, Any]:
    """Return a safe, stable record containing metadata and payload sizes only."""
    if not isinstance(payload, dict):
        raise ValueError("Hook input must be a JSON object.")

    hook_name = payload.get("hook_event_name")
    if not isinstance(hook_name, str) or hook_name not in SUPPORTED_HOOKS:
        raise ValueError("Unsupported or missing Codex hook_event_name.")

    permission_mode = payload.get("permission_mode")
    if permission_mode not in PERMISSION_MODES:
        permission_mode = "unknown"

    source = payload.get("source") if hook_name == "SessionStart" else None
    if source not in SESSION_SOURCES:
        source = "unknown" if hook_name == "SessionStart" else None

    tool_input = payload.get("tool_input")
    tool_response = payload.get("tool_response")
    recorded_at = observed_at or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if recorded_at.endswith("+00:00"):
        recorded_at = recorded_at[:-6] + "Z"

    return {
        "schema_version": 1,
        "observed_at": recorded_at,
        "hook_event_name": hook_name,
        "session_key": _identifier_hash(payload.get("session_id")),
        "turn_key": _identifier_hash(payload.get("turn_id")),
        "tool_call_key": _identifier_hash(payload.get("tool_use_id")),
        "tool_family": _tool_family(payload.get("tool_name")),
        "permission_mode": permission_mode,
        "session_source": source,
        "outcome": _outcome(tool_response) if hook_name == "PostToolUse" else "unknown",
        "tool_input_bytes": _json_size(tool_input),
        "tool_response_bytes": _json_size(tool_response),
    }


def to_feature_vector(record: dict[str, Any]) -> list[float]:
    """Encode one normalized hook record as a fixed-width 32-value event."""
    vector = [0.0] * len(FEATURE_NAMES)
    offset = 0

    for names, value in (
        (HOOK_TYPES, record.get("hook_event_name")),
        (TOOL_FAMILIES, record.get("tool_family")),
        (PERMISSION_MODES, record.get("permission_mode")),
        (OUTCOMES, record.get("outcome")),
        (SESSION_SOURCES, record.get("session_source")),
    ):
        if value in names:
            vector[offset + names.index(value)] = 1.0
        offset += len(names)

    size_scale = math.log1p(1_000_000)
    vector[offset] = min(1.0, math.log1p(max(0, record.get("tool_input_bytes", 0))) / size_scale)
    vector[offset + 1] = min(1.0, math.log1p(max(0, record.get("tool_response_bytes", 0))) / size_scale)
    offset += 2

    presence = (
        bool(record.get("session_key")),
        bool(record.get("turn_key")),
        bool(record.get("tool_call_key")),
        record.get("tool_input_bytes", 0) > 0,
        record.get("tool_response_bytes", 0) > 0,
        record.get("permission_mode") in PERMISSION_MODES,
        record.get("session_source") in SESSION_SOURCES,
    )
    vector[offset:] = [float(value) for value in presence]
    return vector
