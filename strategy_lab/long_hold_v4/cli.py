"""Command-line entry point for Long Hold Dividend V4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_config
from .pipeline import run_current


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "long_hold_v4.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--as-of", required=True, help="Research date in YYYY-MM-DD format")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = run_current(ROOT, config, args.as_of)
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    print(json.dumps({"status": readiness["system_status"], "outputs": {k: str(v) for k, v in paths.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
