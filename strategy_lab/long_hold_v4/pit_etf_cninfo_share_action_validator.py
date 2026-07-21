"""Validate CNInfo SZSE ETF share-action candidates before registry promotion."""

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


CANDIDATE_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "cninfo_etf_share_action_registry_candidates.csv"
MATCH_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "cninfo_etf_share_action_queue_matches.csv"
QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_share_action_evidence_queue.csv"
COLLECTOR_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "cninfo_etf_share_action_collector_latest.json"
OUTPUT_DIR = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "cninfo_etf_share_actions"


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return evaluate_candidates_for_source(
        frame,
        official_source_url_prefix="https://static.cninfo.com.cn/finalpage/",
        expected_source_type="regulatory_filing",
    )


def run_validation() -> dict[str, Any]:
    return run_source_validation(
        ShareActionValidationSpec(
            source_label="CNINFO",
            asset_prefixes=("1",),
            candidate_path=CANDIDATE_PATH,
            match_path=MATCH_PATH,
            queue_path=QUEUE_PATH,
            collector_manifest_path=COLLECTOR_MANIFEST_PATH,
            output_dir=OUTPUT_DIR,
            validation_schema="cninfo_etf_share_action_cross_evidence_v1",
            official_source_url_prefix="https://static.cninfo.com.cn/finalpage/",
            expected_source_type="regulatory_filing",
            official_document_source="CNInfo official fund disclosure PDFs",
            validator_entrypoint_path=Path(__file__).resolve(),
            required_collector_dependency_roles=("shared_share_action_parser",),
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
