import json
import math
import tempfile
import unittest
from pathlib import Path

from codex_telemetry.adapter import FEATURE_NAMES, normalize_hook_event, to_feature_vector
from codex_telemetry.capture_hook import capture_payload
from codex_telemetry.report import read_jsonl, summarize_events


POC_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = POC_ROOT / "tests" / "fixtures" / "codex_hook_samples.jsonl"


class CodexTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.raw_events = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]

    def test_realistic_hook_payloads_normalize_without_persisting_content(self):
        pre_tool = normalize_hook_event(
            self.raw_events[1], observed_at="2026-10-06T10:00:00.000Z"
        )
        post_tool = normalize_hook_event(
            self.raw_events[2], observed_at="2026-10-06T10:00:00.250Z"
        )

        self.assertEqual(pre_tool["hook_event_name"], "PreToolUse")
        self.assertEqual(pre_tool["tool_family"], "shell")
        self.assertEqual(post_tool["outcome"], "success")
        self.assertEqual(pre_tool["tool_call_key"], post_tool["tool_call_key"])
        self.assertGreater(pre_tool["tool_input_bytes"], 0)
        self.assertNotIn("tool_input", pre_tool)
        self.assertNotIn("tool_response", post_tool)
        self.assertNotIn("cwd", pre_tool)
        self.assertNotIn("transcript_path", pre_tool)
        self.assertNotIn("session_id", pre_tool)

    def test_telemetry_features_have_fixed_finite_width(self):
        record = normalize_hook_event(self.raw_events[2])
        features = to_feature_vector(record)

        self.assertEqual(len(FEATURE_NAMES), 32)
        self.assertEqual(len(features), 32)
        self.assertTrue(all(math.isfinite(value) for value in features))
        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[3], 1.0)  # PostToolUse

    def test_report_counts_realistic_session_and_tool_events(self):
        start, pre, post, mcp_pre, mcp_post, end = [
            normalize_hook_event(event, observed_at=timestamp)
            for event, timestamp in zip(
                self.raw_events,
                (
                    "2026-10-06T10:00:00.000Z",
                    "2026-10-06T10:00:01.000Z",
                    "2026-10-06T10:00:01.250Z",
                    "2026-10-06T10:00:02.000Z",
                    "2026-10-06T10:00:02.400Z",
                    "2026-10-06T10:00:03.000Z",
                ),
            )
        ]
        summary = summarize_events([start, pre, post, mcp_pre, mcp_post, end])

        self.assertEqual(summary["total_events"], 6)
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["post_tool_outcomes"], {"error": 1, "success": 1})
        self.assertEqual(summary["tool_families"], {"mcp": 2, "shell": 2})

    def test_capture_writes_only_safe_jsonl_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            capture_payload(json.dumps(self.raw_events[1]), output)
            written_text = output.read_text(encoding="utf-8")
            record = read_jsonl(output)[0]

        self.assertEqual(record["tool_family"], "shell")
        self.assertNotIn("python -m pytest", written_text)
        self.assertNotIn("transcript.jsonl", written_text)
        self.assertNotIn("/Users/example/project", written_text)


if __name__ == "__main__":
    unittest.main()
