import argparse
import json
import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parents[1]
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from codex_telemetry.report import read_jsonl, summarize_events


def main():
    parser = argparse.ArgumentParser(description="Summarize captured Codex hook telemetry.")
    parser.add_argument(
        "--input",
        default=str(POC_ROOT / "results" / "live" / "codex_events.jsonl"),
        help="JSONL file written by the Codex hook listener.",
    )
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        parser.error(f"{path} does not exist yet. Review/trust the hook and use Codex in this repo first.")
    print(json.dumps(summarize_events(read_jsonl(path)), indent=2))


if __name__ == "__main__":
    main()
