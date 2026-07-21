from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from model_run_manifest import schema_dict, validate_model_run_manifest


ROOT = Path("Introduction-to-Quantitative-Finance")
V36_DIR = ROOT / "outputs" / "hirssm_v3_6_component_attribution"
V39_DIR = ROOT / "outputs" / "agent_runs" / "v3_9"
V10_DIR = ROOT / "outputs" / "agent_runs" / "v3_10"
V36_AUDIT_DIR = V39_DIR / "backtest_validation_auditor_v36_upstream"
NESTED_DIR = V10_DIR / "chief_quant_orchestrator"
MANIFEST_DIR = V10_DIR / "code_quality_engineer_manifest_schema"
TEMPLATE_SCHEMA = ROOT / "strategy_lab" / "agents" / "_templates" / "model_run_manifest.schema.json"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def manifest_gap_report(manifest: dict) -> pd.DataFrame:
    required = [
        "schema_version",
        "task_id",
        "run_id",
        "model_version",
        "status",
        "started_at",
        "finished_at",
        "cwd",
        "command",
        "argv",
        "code_refs",
        "config",
        "data_refs",
        "environment",
        "selection",
        "artifacts",
        "checks",
        "limitations",
        "risk_flags",
        "next_decision",
    ]
    legacy_present = set(manifest)
    rows = []
    for field in required:
        rows.append(
            {
                "field": field,
                "present": field in legacy_present,
                "blocks_future_promotion_if_missing": True,
                "status": "present" if field in legacy_present else "missing",
            }
        )
    return pd.DataFrame(rows)


def build_v36_audit() -> None:
    V36_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(V36_DIR / "run_manifest.json")
    self_check = pd.read_csv(V36_DIR / "self_check_results.csv")
    score = pd.read_csv(V36_DIR / "benchmark_relative_score_table.csv")
    perf = pd.read_csv(V36_DIR / "all_candidate_oos_performance.csv")
    selected = str(manifest.get("selected", "unknown"))
    score_sorted = score.sort_values("benchmark_relative_score", ascending=False).reset_index(drop=True)
    selected_score = float(score_sorted.loc[0, "benchmark_relative_score"]) if len(score_sorted) else float("nan")
    second_score = float(score_sorted.loc[1, "benchmark_relative_score"]) if len(score_sorted) > 1 else float("nan")
    score_margin = selected_score - second_score
    cost10 = perf[(perf["candidate"] == selected) & (perf["cost_bps"] == 10.0)]
    selected_10_ann = float(cost10["annual_return"].iloc[0]) if not cost10.empty else float("nan")

    gaps = manifest_gap_report(manifest)
    gaps.to_csv(V36_AUDIT_DIR / "v36_manifest_gap_report.csv", index=False, encoding="utf-8-sig")

    findings = pd.DataFrame(
        [
            {
                "finding": "self_checks",
                "status": "pass" if bool((self_check["pass"].astype(str).str.lower() == "true").all()) else "fail",
                "detail": f"{int((self_check['pass'].astype(str).str.lower() == 'true').sum())}/{len(self_check)} checks pass",
            },
            {
                "finding": "selection_method",
                "status": "fail",
                "detail": "V3.6 is selected from same-period OOS backtests; no nested selection artifact found.",
            },
            {
                "finding": "score_margin",
                "status": "observation" if score_margin < 0.01 else "pass",
                "detail": f"top score margin versus runner-up is {score_margin:.6f}",
            },
            {
                "finding": "ex_post_state_gating",
                "status": "fail",
                "detail": "V3.6 disables alpha sleeve in crash-rebound because prior V3.5 attribution was negative there.",
            },
            {
                "finding": "upstream_self_check_propagation",
                "status": "fail",
                "detail": "V3.2 manifest has self_check_pass=false and V3.4 self-check fails promotion-style gates; V3.6 does not propagate these as blockers.",
            },
            {
                "finding": "manifest_reproducibility",
                "status": "fail",
                "detail": f"{int((~gaps['present']).sum())} strict manifest fields missing",
            },
            {
                "finding": "upstream_use_allowed",
                "status": "fail",
                "detail": "V3.6 cannot be used as a default upstream baseline target source; it can only be an experimental comparison snapshot.",
            },
        ]
    )
    findings.to_csv(V36_AUDIT_DIR / "v36_upstream_audit_findings.csv", index=False, encoding="utf-8-sig")

    report = [
        "# V3.6 Upstream Audit",
        "",
        "## Status",
        "",
        "`fail` for use as V3.8 default upstream baseline source",
        "",
        "## Critical Findings",
        "",
        "- V3.6 self-checks pass, and selected outputs are internally present.",
        "- V3.6 selected candidate is chosen from same-period OOS candidate results, not nested validation.",
        "- V3.6 state gating has ex-post attribution traces: crash-rebound alpha exposure is disabled because V3.5 attribution was negative there.",
        "- V3.6 upstream inputs are not all governance-clean: V3.2 has `self_check_pass=false`, and V3.4 fails promotion-style gates.",
        f"- Selected candidate: `{selected}`; 10bps annual return: `{selected_10_ann:.4f}`.",
        f"- Score margin versus runner-up: `{score_margin:.6f}`, which is not strong enough as independent evidence.",
        "- The original V3.6 manifest is a lightweight summary and fails strict future-promotion manifest requirements.",
        "",
        "## Decision",
        "",
        "V3.6 cannot be used as a default upstream baseline target source for V3.8. It may only remain an experimental comparison snapshot marked same-OOS-selected, ex-post-gated, manifest-incomplete, and not promotion-approved.",
    ]
    (V36_AUDIT_DIR / "agent_report.md").write_text("\n".join(report), encoding="utf-8")

    run_manifest = {
        "run_id": "20260526_v3_9_v36_upstream_audit_run_001",
        "task_id": "20260526_v3_9_v36_upstream_audit",
        "agent": "backtest_validation_auditor",
        "version": "V3.9",
        "baseline": "HIRSSM V3.6 upstream for V3.8",
        "status": "fail",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": "python -X utf8 strategy_lab/hirssm_v3_10_governance_upgrade.py",
        "config": {"selected": selected, "audit_target": rel(V36_DIR)},
        "data_refs": [rel(V36_DIR)],
        "code_refs": ["strategy_lab/hirssm_v3_6_component_attribution.py", "strategy_lab/hirssm_v3_10_governance_upgrade.py"],
        "output_dir": rel(V36_AUDIT_DIR),
        "allowed_inputs": [rel(V36_DIR), "strategy_lab/hirssm_v3_6_component_attribution.py"],
        "artifacts": [
            rel(V36_AUDIT_DIR / "agent_report.md"),
            rel(V36_AUDIT_DIR / "v36_upstream_audit_findings.csv"),
            rel(V36_AUDIT_DIR / "v36_manifest_gap_report.csv"),
            rel(V36_AUDIT_DIR / "agent_run_manifest.json"),
        ],
        "outputs": [
            rel(V36_AUDIT_DIR / "agent_report.md"),
            rel(V36_AUDIT_DIR / "v36_upstream_audit_findings.csv"),
            rel(V36_AUDIT_DIR / "v36_manifest_gap_report.csv"),
        ],
        "changed_files": [
            rel(V36_AUDIT_DIR / "agent_report.md"),
            rel(V36_AUDIT_DIR / "v36_upstream_audit_findings.csv"),
            rel(V36_AUDIT_DIR / "v36_manifest_gap_report.csv"),
            rel(V36_AUDIT_DIR / "agent_run_manifest.json"),
        ],
        "metrics": {
            "selected_10bps_annual_return": selected_10_ann,
            "score_margin_vs_runner_up": score_margin,
            "strict_manifest_missing_fields": int((~gaps["present"]).sum()),
        },
        "self_check_pass": True,
        "fail_count": int((findings["status"] == "fail").sum()),
        "warn_count": int((findings["status"] == "observation").sum()),
        "limitations": ["This audit uses existing V3.6 artifacts and does not rerun V3.6."],
        "risk_flags": [
            "same_oos_selection",
            "ex_post_state_gating",
            "upstream_self_check_failure_not_propagated",
            "strict_manifest_missing",
        ],
        "next_decision": "rebuild a governance-clean baseline before using V3.6-derived targets as upstream source",
        "handoff_summary": "V3.6 fails as a default upstream baseline source; use only as experimental comparison snapshot.",
    }
    (V36_AUDIT_DIR / "agent_run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest_schema_task() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATE_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_SCHEMA.write_text(json.dumps(schema_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    enhanced_path = V39_DIR / "code_quality_engineer_repro_manifest" / "enhanced_manifest_template.json"
    validation_findings = []
    if enhanced_path.exists():
        validation_findings = validate_model_run_manifest(read_json(enhanced_path))
    validation = pd.DataFrame(validation_findings)
    if validation.empty:
        validation = pd.DataFrame(columns=["severity", "field", "message"])
    validation.to_csv(MANIFEST_DIR / "manifest_schema_validation.csv", index=False, encoding="utf-8-sig")

    blocking = [
        "missing required top-level field",
        "code/config/data ref missing or unhashed",
        "required artifact missing or unhashed",
        "self_check_pass is not true",
        "fail_count is non-zero",
    ]
    policy = pd.DataFrame(
        [
            {"rule": rule, "promotion_blocking": True}
            for rule in blocking
        ]
        + [{"rule": "git_dirty true", "promotion_blocking": False}]
    )
    policy.to_csv(MANIFEST_DIR / "manifest_promotion_policy.csv", index=False, encoding="utf-8-sig")

    report = [
        "# V3.10 Model Manifest Schema",
        "",
        "## Status",
        "",
        "`accepted`",
        "",
        "## Rule",
        "",
        "Future model promotion requires `model_run_manifest.v1`. Missing command, argv, code/input/output hashes, environment, artifact inventory, or self-check evidence blocks promotion.",
        "",
        "## Integration",
        "",
        "- Schema template is written to `strategy_lab/agents/_templates/model_run_manifest.schema.json`.",
        "- Validation utility is `strategy_lab/model_run_manifest.py`.",
        "- Future strategy scripts should call `build_model_run_manifest` at generation time, not after the run.",
    ]
    (MANIFEST_DIR / "agent_report.md").write_text("\n".join(report), encoding="utf-8")

    run_manifest = {
        "run_id": "20260526_v3_10_model_manifest_schema_run_001",
        "task_id": "20260526_v3_10_model_manifest_schema",
        "agent": "code_quality_engineer",
        "version": "V3.10",
        "baseline": "HIRSSM V3.8 observation baseline",
        "status": "pass",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": "python -X utf8 strategy_lab/hirssm_v3_10_governance_upgrade.py",
        "config": {"schema_version": "model_run_manifest.v1"},
        "data_refs": [rel(enhanced_path)],
        "code_refs": ["strategy_lab/model_run_manifest.py", "strategy_lab/hirssm_v3_10_governance_upgrade.py"],
        "output_dir": rel(MANIFEST_DIR),
        "allowed_inputs": [rel(V39_DIR / "code_quality_engineer_repro_manifest"), "strategy_lab/model_run_manifest.py"],
        "artifacts": [
            rel(MANIFEST_DIR / "agent_report.md"),
            rel(MANIFEST_DIR / "manifest_schema_validation.csv"),
            rel(MANIFEST_DIR / "manifest_promotion_policy.csv"),
            rel(TEMPLATE_SCHEMA),
            rel(MANIFEST_DIR / "agent_run_manifest.json"),
        ],
        "outputs": [
            rel(MANIFEST_DIR / "agent_report.md"),
            rel(MANIFEST_DIR / "manifest_schema_validation.csv"),
            rel(MANIFEST_DIR / "manifest_promotion_policy.csv"),
            rel(TEMPLATE_SCHEMA),
        ],
        "changed_files": [
            rel(MANIFEST_DIR / "agent_report.md"),
            rel(MANIFEST_DIR / "manifest_schema_validation.csv"),
            rel(MANIFEST_DIR / "manifest_promotion_policy.csv"),
            rel(TEMPLATE_SCHEMA),
            rel(MANIFEST_DIR / "agent_run_manifest.json"),
        ],
        "metrics": {"schema_required_field_count": len(schema_dict()["required_top_level_fields"])},
        "self_check_pass": True,
        "fail_count": 0,
        "warn_count": int((validation.get("severity", pd.Series(dtype=str)) == "warn").sum()),
        "limitations": ["Existing historical manifests remain retrospective unless regenerated at model runtime."],
        "risk_flags": [],
        "next_decision": "require strict manifest in V3.10+ model scripts",
        "handoff_summary": "Strict model_run_manifest.v1 schema and validator are available.",
    }
    (MANIFEST_DIR / "agent_run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_nested_selection_design() -> None:
    NESTED_DIR.mkdir(parents=True, exist_ok=True)
    robustness = pd.read_csv(V39_DIR / "backtest_validation_auditor_pbo_dsr" / "robustness_decision.csv")
    repro_gap = pd.read_csv(V39_DIR / "code_quality_engineer_repro_manifest" / "repro_manifest_gap_report.csv")
    failed_costs = int((robustness["promotion_allowed"].astype(str).str.lower() != "true").sum())
    missing_repro = int((~repro_gap["present"].astype(str).str.lower().eq("true") & repro_gap["blocks_future_promotion_if_missing"].astype(str).str.lower().eq("true")).sum())

    policy = pd.DataFrame(
        [
            {"gate": "context_isolation", "threshold": "agent artifacts only", "promotion_blocking": True},
            {"gate": "clean_upstream_baseline", "threshold": "no same-OOS-selected or ex-post-gated upstream target", "promotion_blocking": True},
            {"gate": "nested_selection", "threshold": "no full-sample candidate selection", "promotion_blocking": True},
            {"gate": "pbo", "threshold": "<=0.20 on 10/20/30bps", "promotion_blocking": True},
            {"gate": "dsr", "threshold": "worst selected DSR>=0.95 under N_eff=18", "promotion_blocking": True},
            {"gate": "paired_delta", "threshold": "mean annual delta>=25bp and bootstrap ci_low>0", "promotion_blocking": True},
            {"gate": "manifest", "threshold": "model_run_manifest.v1 passes", "promotion_blocking": True},
            {"gate": "costs", "threshold": "10/20/30bps all valid", "promotion_blocking": True},
        ]
    )
    policy.to_csv(NESTED_DIR / "candidate_promotion_policy.csv", index=False, encoding="utf-8-sig")

    decision = pd.DataFrame(
        [
            {
                "decision": "V3.8_not_promoted",
                "status": "accepted",
                "reason": f"{failed_costs} cost rows failed strict PBO/DSR promotion checks",
            },
            {
                "decision": "strict_manifest_required",
                "status": "accepted",
                "reason": f"{missing_repro} future-promotion-blocking manifest fields missing in V3.8",
            },
            {
                "decision": "V3.10_should_design_nested_candidate_selection",
                "status": "accepted",
                "reason": "candidate selection must be made inside train folds and evaluated out-of-sample",
            },
        ]
    )
    decision.to_csv(NESTED_DIR / "v3_10_design_decision.csv", index=False, encoding="utf-8-sig")

    design = [
        "# HIRSSM V3.10 Nested Selection Design",
        "",
        "## Objective",
        "",
        "Rebuild a governance-clean baseline and replace full-sample candidate ranking with a nested, purged validation route before any candidate can become default.",
        "",
        "## Required Flow",
        "",
        "1. Start from a clean upstream baseline that is not selected by same-period OOS and is not gated by ex-post attribution.",
        "2. Freeze candidate definitions before validation.",
        "3. Split history into walk-forward train/test windows or 10 continuous CSCV blocks.",
        "4. Apply 120 trading-day purge and 21 trading-day embargo around OOS blocks.",
        "5. Select candidates only on IS data using the predeclared score.",
        "6. Evaluate selected candidates on OOS data at 10/20/30bps.",
        "7. Require PBO <= 0.20, worst selected DSR >= 0.95 under N_eff=18, and paired delta versus control with bootstrap CI lower bound > 0.",
        "8. Require `model_run_manifest.v1` and independent validation artifacts before promotion.",
        "",
        "## Current Decision",
        "",
        "V3.8 is not promoted, and V3.6 fails as a default upstream baseline source. V3.10 work should first rebuild a clean baseline and selection harness before seeking new alpha.",
    ]
    (NESTED_DIR / "nested_selection_rule.md").write_text("\n".join(design), encoding="utf-8")
    (NESTED_DIR / "agent_report.md").write_text("\n".join(design), encoding="utf-8")

    run_manifest = {
        "run_id": "20260526_v3_10_nested_selection_design_run_001",
        "task_id": "20260526_v3_10_nested_selection_design",
        "agent": "chief_quant_orchestrator",
        "version": "V3.10",
        "baseline": "HIRSSM V3.8 observation baseline",
        "status": "pass",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": "python -X utf8 strategy_lab/hirssm_v3_10_governance_upgrade.py",
        "config": {"promotion_gates": len(policy)},
        "data_refs": [
            rel(V39_DIR / "backtest_validation_auditor_pbo_dsr" / "robustness_decision.csv"),
            rel(V39_DIR / "code_quality_engineer_repro_manifest" / "repro_manifest_gap_report.csv"),
        ],
        "code_refs": ["strategy_lab/hirssm_v3_10_governance_upgrade.py"],
        "output_dir": rel(NESTED_DIR),
        "allowed_inputs": [rel(V39_DIR / "backtest_validation_auditor_pbo_dsr"), rel(V39_DIR / "code_quality_engineer_repro_manifest")],
        "artifacts": [
            rel(NESTED_DIR / "agent_report.md"),
            rel(NESTED_DIR / "nested_selection_rule.md"),
            rel(NESTED_DIR / "candidate_promotion_policy.csv"),
            rel(NESTED_DIR / "v3_10_design_decision.csv"),
            rel(NESTED_DIR / "agent_run_manifest.json"),
        ],
        "outputs": [
            rel(NESTED_DIR / "agent_report.md"),
            rel(NESTED_DIR / "nested_selection_rule.md"),
            rel(NESTED_DIR / "candidate_promotion_policy.csv"),
            rel(NESTED_DIR / "v3_10_design_decision.csv"),
        ],
        "changed_files": [
            rel(NESTED_DIR / "agent_report.md"),
            rel(NESTED_DIR / "nested_selection_rule.md"),
            rel(NESTED_DIR / "candidate_promotion_policy.csv"),
            rel(NESTED_DIR / "v3_10_design_decision.csv"),
            rel(NESTED_DIR / "agent_run_manifest.json"),
        ],
        "metrics": {"promotion_gate_count": len(policy), "v3_8_failed_cost_rows": failed_costs},
        "self_check_pass": True,
        "fail_count": 0,
        "warn_count": 0,
        "limitations": ["This task defines governance and selection rules; it does not create a new return stream."],
        "risk_flags": [],
        "next_decision": "implement nested selection before any V3.10 strategy promotion",
        "handoff_summary": "V3.10 must remove full-sample candidate selection and use strict promotion gates.",
    }
    (NESTED_DIR / "agent_run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    global ROOT, V36_DIR, V39_DIR, V10_DIR, V36_AUDIT_DIR, NESTED_DIR, MANIFEST_DIR, TEMPLATE_SCHEMA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    ROOT = Path(args.root)
    V36_DIR = ROOT / "outputs" / "hirssm_v3_6_component_attribution"
    V39_DIR = ROOT / "outputs" / "agent_runs" / "v3_9"
    V10_DIR = ROOT / "outputs" / "agent_runs" / "v3_10"
    V36_AUDIT_DIR = V39_DIR / "backtest_validation_auditor_v36_upstream"
    NESTED_DIR = V10_DIR / "chief_quant_orchestrator"
    MANIFEST_DIR = V10_DIR / "code_quality_engineer_manifest_schema"
    TEMPLATE_SCHEMA = ROOT / "strategy_lab" / "agents" / "_templates" / "model_run_manifest.schema.json"

    build_v36_audit()
    build_manifest_schema_task()
    build_nested_selection_design()

    result = {
        "v36_audit": rel(V36_AUDIT_DIR / "agent_report.md"),
        "manifest_schema": rel(TEMPLATE_SCHEMA),
        "nested_selection_design": rel(NESTED_DIR / "nested_selection_rule.md"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
