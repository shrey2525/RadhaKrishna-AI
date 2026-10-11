"""Privacy-minimized ingestion for local Codex hook events."""

from .adapter import FEATURE_NAMES, normalize_hook_event, to_feature_vector

__all__ = ["FEATURE_NAMES", "normalize_hook_event", "to_feature_vector"]
