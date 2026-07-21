"""Validate SSE ETF share-action candidates before registry promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pit_etf_share_action_validator_core import (
    ROOT,
    ShareActionValidationSpec,
    evaluate_candidates as evaluate_candidates_for_source,
    run_validation as run_source_validation,
)


CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "sse_etf_share_action_registry_candidates.csv"
MATCH_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "sse_etf_share_action_queue_matches.csv"
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_evidence_queue.csv"
COLLECTOR_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "sse_etf_share_action_collector_latest.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "sse_etf_share_actions"


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_candidates_for_source(
        frame,
        official_source_url_prefix="https://www.sse.com.cn/disclosure/fund/announcement/",
        expected_source_type="exchange_announcement",
    )


def run_validation() -> dict[str, Any]:
    return run_source_validation(
        ShareActionValidationSpec(
            source_label="SSE",
            asset_prefixes=("5",),
            candidate_path=CANDIDATE_PATH,
            match_path=MATCH_PATH,
            queue_path=QUEUE_PATH,
            collector_manifest_path=COLLECTOR_MANIFEST_PATH,
            output_dir=OUTPUT_DIR,
            validation_schema="sse_etf_share_action_cross_evidence_v2",
            official_source_url_prefix="https://www.sse.com.cn/disclosure/fund/announcement/",
            expected_source_type="exchange_announcement",
            official_document_source="Shanghai Stock Exchange fund announcement PDFs",
            validator_entrypoint_path=Path(__file__).resolve(),
        )
    )


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    manifest = run_validation()
    keys = (
        "qualification_status",
        "candidate_rows",
        "failed_check_rows",
        "independent_source_count",
        "registry_promotion_allowed",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
