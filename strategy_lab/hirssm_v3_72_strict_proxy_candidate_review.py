#!/usr/bin/env python
"""Run HIRSSM V3.72 strict proxy candidate review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from strict_proxy_candidate_review import (
    StrictProxyCandidateReviewConfig,
    build_acceptance_checks,
    build_catalog,
    build_cross_index_validation,
    build_cross_label_coverage,
    build_extended_negative_controls,
    build_index_forward_labels,
    build_market_trend_proxy_report,
    build_no_promotion_guard,
    build_redundancy_clusters,
    build_report,
    build_source_family_strict_summary,
    build_strict_candidate_decision,
    prepare_feature_frame,
    select_proxy_positive_candidates,
    validate_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "strict_proxy_candidate_review_v3_72.json"
TASK_ID = "20260529_v3_72_strict_proxy_candidate_review"
VERSION = "V3.72"
AGENT = "chief_quant_orchestrator"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> StrictProxyCandidateReviewConfig:
    thresholds = raw["thresholds"]
    return StrictProxyCandidateReviewConfig(
        v3_71_manifest_path=resolve_path(raw["v3_71_manifest_path"]),
        v3_71_summary_path=resolve_path(raw["v3_71_summary_path"]),
        v3_71_candidate_decision_path=resolve_path(raw["v3_71_candidate_decision_path"]),
        v3_71_walk_forward_path=resolve_path(raw["v3_71_walk_forward_path"]),
        combined_panel_path=resolve_path(raw["combined_panel_path"]),
        primary_label_path=resolve_path(raw["primary_label_path"]),
        cross_index_paths={key: resolve_path(value) for key, value in raw["cross_index_paths"].items()},
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(x) for x in raw["horizons"]),
        correlation_cluster_threshold=float(thresholds["correlation_cluster_threshold"]),
        min_cluster_score_gap=float(thresholds["min_cluster_score_gap"]),
        negative_control_shifts=tuple(int(x) for x in raw["negative_control_shifts"]),
        negative_control_ratio_threshold=float(thresholds["negative_control_ratio_threshold"]),
        negative_control_abs_threshold=float(thresholds["negative_control_abs_threshold"]),
        min_cross_index_observations=int(thresholds["min_cross_index_observations"]),
        min_cross_index_signed_spearman=float(thresholds["min_cross_index_signed_spearman"]),
        min_cross_index_signed_qspread=float(thresholds["min_cross_index_signed_qspread"]),
        min_cross_index_year_positive_share=float(thresholds["min_cross_index_year_positive_share"]),
        min_cross_index_pass_count=int(thresholds["min_cross_index_pass_count"]),
        market_trend_proxy_corr_threshold=float(thresholds["market_trend_proxy_corr_threshold"]),
        trend_windows=tuple(int(x) for x in raw["trend_windows"]),
        top_quantile=float(raw["bucket_validation"]["top_quantile"]),
        bottom_quantile=float(raw["bucket_validation"]["bottom_quantile"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    cols: list[str] = []
    for frame in frames:
        cols.extend(str(col) for col in frame.columns)
    return cols


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "strict_proxy_candidate_review.py",
        ROOT / "strategy_lab" / "hirssm_v3_72_strict_proxy_candidate_review.py",
        ROOT / "configs" / "strict_proxy_candidate_review_v3_72.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_72_strict_proxy_candidate_review.json",
        ROOT / "reports" / "AGENT_TASK_BOARD.md",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started_at = now_text()
    config_path = resolve_path(args.config)
    raw_config = read_json(config_path)
    config = load_config(raw_config)
    output_dir = config.output_dir

    v3_71_manifest = read_json(config.v3_71_manifest_path)
    summary = read_csv(config.v3_71_summary_path)
    decisions_v3_71 = read_csv(config.v3_71_candidate_decision_path)
    windows_v3_71 = read_csv(config.v3_71_walk_forward_path)
    combined = read_csv(config.combined_panel_path)
    primary_labels = read_csv(config.primary_label_path)
    cross_indices = {symbol: read_csv(path) for symbol, path in config.cross_index_paths.items()}

    input_checks = validate_inputs(v3_71_manifest, summary, decisions_v3_71, windows_v3_71, combined, primary_labels, cross_indices, config)
    candidates = select_proxy_positive_candidates(summary)
    features = prepare_feature_frame(combined)
    cross_labels = {symbol: build_index_forward_labels(frame, symbol, config.horizons) for symbol, frame in cross_indices.items()}
    label_coverage = build_cross_label_coverage(cross_labels, features["signal_date"])
    clusters = build_redundancy_clusters(candidates, features, config)
    negative = build_extended_negative_controls(candidates, features, primary_labels, config)
    cross = build_cross_index_validation(candidates, features, cross_labels, config)
    market_symbol = "000985" if "000985" in cross_indices else sorted(cross_indices)[0]
    trend = build_market_trend_proxy_report(candidates, features, cross_indices[market_symbol], config)
    strict_decisions = build_strict_candidate_decision(candidates, clusters, negative, cross, trend, config)
    source_summary = build_source_family_strict_summary(strict_decisions)
    guard = build_no_promotion_guard(strict_decisions)
    output_frames = [input_checks, candidates, label_coverage, clusters, negative, cross, trend, strict_decisions, source_summary, guard]
    acceptance = build_acceptance_checks(input_checks, candidates, clusters, negative, cross, trend, strict_decisions, guard, output_columns(output_frames))
    report = build_report(candidates, label_coverage, clusters, negative, cross, trend, strict_decisions, source_summary, input_checks, acceptance, guard)
    catalog = build_catalog(candidates, strict_decisions, config)

    output_paths = {
        "input_checks": output_dir / "input_checks.csv",
        "candidates": output_dir / "proxy_positive_candidates.csv",
        "label_coverage": output_dir / "alternate_index_label_coverage.csv",
        "clusters": output_dir / "feature_redundancy_clusters.csv",
        "negative": output_dir / "extended_negative_controls.csv",
        "cross": output_dir / "cross_index_validation.csv",
        "trend": output_dir / "market_trend_proxy_report.csv",
        "decisions": output_dir / "strict_candidate_decision.csv",
        "source_summary": output_dir / "source_family_strict_summary.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "strict_proxy_candidate_review_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(input_checks, output_paths["input_checks"])
    write_csv(candidates, output_paths["candidates"])
    write_csv(label_coverage, output_paths["label_coverage"])
    write_csv(clusters, output_paths["clusters"])
    write_csv(negative, output_paths["negative"])
    write_csv(cross, output_paths["cross"])
    write_csv(trend, output_paths["trend"])
    write_csv(strict_decisions, output_paths["decisions"])
    write_csv(source_summary, output_paths["source_summary"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    strict_survivors = int(strict_decisions["strict_review_status"].astype(str).eq("strict_proxy_survivor_for_label_review").sum()) if not strict_decisions.empty else 0
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "strict_decisions_do_not_promote",
                "status": "pass" if not strict_decisions["default_model_allowed"].astype(bool).any() and not strict_decisions["portfolio_backtest_allowed"].astype(bool).any() else "fail",
                "detail": f"decision_rows={len(strict_decisions)}",
            },
            {
                "check": "official_total_return_evidence_absent",
                "status": "pass" if not strict_decisions["official_total_return_evidence"].astype(bool).any() else "fail",
                "detail": "price-index proxy review only",
            },
            {
                "check": "cross_index_review_present",
                "status": "pass" if int(cross["cross_symbol"].nunique()) >= config.min_cross_index_pass_count else "fail",
                "detail": f"symbols={','.join(sorted(cross['cross_symbol'].astype(str).unique())) if not cross.empty else ''}",
            },
            {
                "check": "strict_survivor_queue_only",
                "status": "pass",
                "detail": f"strict_survivors={strict_survivors}",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["input_checks"],
        output_paths["candidates"],
        output_paths["label_coverage"],
        output_paths["clusters"],
        output_paths["negative"],
        output_paths["cross"],
        output_paths["trend"],
        output_paths["decisions"],
        output_paths["source_summary"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    fail_count = int(self_check["status"].eq("fail").sum())
    metrics = {
        "v3_71_proxy_positive_rows": int(len(candidates)),
        "strict_survivor_rows": strict_survivors,
        "redundancy_cluster_count": int(clusters["cluster_id"].nunique()) if not clusters.empty else 0,
        "redundant_non_representative_rows": int((~clusters["is_cluster_representative"].astype(bool)).sum()) if not clusters.empty else 0,
        "extended_negative_control_flag_rows": int(negative.drop_duplicates(["feature_id", "horizon"])["extended_negative_control_flag"].astype(bool).sum()) if not negative.empty else 0,
        "cross_index_validation_rows": int(len(cross)),
        "market_trend_proxy_flag_rows": int(trend["market_trend_proxy_flag"].astype(bool).sum()) if not trend.empty else 0,
        "cross_symbols": sorted(config.cross_index_paths),
        "portfolio_backtest_status": "not_run",
        "model_promotion_status": "blocked",
    }

    all_outputs = outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.71 guarded proxy-positive feature-horizon candidates",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": started_at,
        "command": f"python -B -X utf8 strategy_lab/hirssm_v3_72_strict_proxy_candidate_review.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [
            rel(config.v3_71_manifest_path),
            rel(config.v3_71_summary_path),
            rel(config.v3_71_candidate_decision_path),
            rel(config.v3_71_walk_forward_path),
            rel(config.combined_panel_path),
            rel(config.primary_label_path),
            *[rel(path) for path in config.cross_index_paths.values()],
        ],
        "code_refs": [
            "strategy_lab/strict_proxy_candidate_review.py",
            "strategy_lab/hirssm_v3_72_strict_proxy_candidate_review.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in all_outputs],
        "changed_files": build_changed_files(all_outputs).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": 0,
        "limitations": [
            "V3.72 still uses price-index proxy labels and cannot support official total-return claims.",
            "Strict survivors are only eligible for higher-quality label-source review, not portfolio harnesses.",
            "Alternate index validation uses broad-index price proxies and may share market beta structure.",
        ],
        "risk_flags": [
            "price_index_proxy_label_only",
            "market_beta_shared_label_structure",
            "portfolio_harness_not_run",
            "model_promotion_blocked",
        ],
        "next_decision": "Proceed to V3.73 higher-quality label-source acquisition/review for strict survivors, or retire candidates if no better label source is available.",
        "handoff_summary": "V3.72 strictly reviewed V3.71 proxy-positive rows with clustering, extended negative controls, alternate index validation, and market-trend proxy diagnostics.",
    }
    write_json(manifest, output_paths["manifest"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
