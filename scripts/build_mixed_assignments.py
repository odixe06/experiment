"""Create/verify fixed assignments only; never train or push a notebook."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model_assignment import CASES, load_assignment, write_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/mixed-model-v1")
    parser.add_argument("--check", action="store_true", help="Read-only validation of frozen files")
    args = parser.parse_args()
    if not args.check:
        write_bundle(args.out)
    for case in CASES:
        load_assignment(args.out, case)
    meta = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": "PASS", "assignment_seed": meta["assignment_seed"],
                      "cases": meta["cases"]}, indent=2))


if __name__ == "__main__":
    main()
