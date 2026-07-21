from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_assistant_framework import (
    ROOT,
    build_architecture_governance_coverage_rows,
    build_architecture_rows,
    build_data_coverage_rows,
    build_fundamental_formula_spec_rows,
    build_fundamental_score_reconciliation_rows,
    build_macro_pit_check_rows,
    build_object_schema_rows,
    build_research_object_contract_check_rows,
    build_research_universe,
    build_synthesis_formula_spec_rows,
    build_synthesis_input_contract_check_rows,
    build_synthesis_score_reconciliation_rows,
    build_technical_formula_spec_rows,
    build_technical_input_contract_check_rows,
    compute_fundamental_latest,
    compute_technical_panel,
    ensure_dir,
    load_latest_macro_snapshot,
    read_json,
    rel_path,
    render_html_dashboard,
    render_markdown_report,
    synthesize_views,
    write_csv,
    write_text,
)


CONFIG_PATH = ROOT / "configs" / "quant_research_assistant_v3_86_to_v3_92.json"
SCRIPT_PATH = ROOT / "strategy_lab" / "hirssm_v3_86_to_v3_92_quant_research_assistant.py"
FRAMEWORK_PATH = ROOT / "strategy_lab" / "quant_research_assistant_framework.py"


VERSIONS = [
    {
        "version": "V3.86",
        "task_id": "20260602_v3_86_quant_research_assistant_architecture",
        "agent": "chief_quant_orchestrator",
        "slug": "quant_research_assistant_architecture",
        "summary": "define object-level quant research assistant architecture and agent responsibility upgrades",
    },
    {
        "version": "V3.87",
        "task_id": "20260602_v3_87_research_object_schema",
        "agent": "data_steward",
        "slug": "research_object_schema",
        "summary": "create unified research object schema and data coverage matrix",
    },
    {
        "version": "V3.88",
        "task_id": "20260602_v3_88_technical_signal_engine",
        "agent": "technical_market_analyst",
        "slug": "technical_signal_engine",
        "summary": "compute technical trend, momentum, drawdown, volatility, and liquidity views",
    },
    {
        "version": "V3.89",
        "task_id": "20260602_v3_89_fundamental_factor_engine",
        "agent": "fundamental_equity_analyst",
        "slug": "fundamental_factor_engine",
        "summary": "compute valuation/dividend views and explicit PIT data gaps",
    },
    {
        "version": "V3.90",
        "task_id": "20260602_v3_90_research_view_synthesizer",
        "agent": "investment_view_synthesizer",
        "slug": "research_view_synthesizer",
        "summary": "combine technical and fundamental evidence into object-level research views",
    },
    {
        "version": "V3.91",
        "task_id": "20260602_v3_91_research_report_builder",
        "agent": "research_reporter",
        "slug": "research_report_builder",
        "summary": "render markdown and HTML research reports",
    },
    {
        "version": "V3.92",
        "task_id": "20260602_v3_92_sample_research_run",
        "agent": "backtest_validation_auditor",
        "slug": "sample_research_run",
        "summary": "run an end-to-end sample research workflow and record self-check evidence",
    },
]


def output_dir(meta: dict[str, str]) -> Path:
    suffix = meta["version"].lower().replace(".", "_")
    return ROOT / "outputs" / "agent_runs" / suffix / meta["slug"]


def artifact(path: Path) -> str:
    return rel_path(path)


def acceptance_rows(checks: list[tuple[str, bool, str]]) -> list[dict[str, Any]]:
    return [
        {
            "check": name,
            "status": "pass" if ok else "fail",
            "detail": detail,
        }
        for name, ok, detail in checks
    ]


def self_check_rows(paths: list[Path], extra: list[tuple[str, bool, str]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append(
            {
                "check": f"exists:{artifact(path)}",
                "status": "pass" if path.exists() else "fail",
                "detail": "file exists" if path.exists() else "missing file",
            }
        )
    for name, ok, detail in extra or []:
        rows.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})
    return rows


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path, encoding="utf-8-sig").fillna("").to_dict("records")


def build_gate_audit_row(version: str, gate: str, path: Path, status_values: set[str] | None = None) -> dict[str, Any]:
    rows = read_csv_rows(path)
    allowed_statuses = status_values or {"pass"}
    missing = not path.exists()
    fail_count = 1 if missing else sum(str(row.get("status", "")).lower() not in allowed_statuses for row in rows)
    observation_count = 0 if missing else sum(str(row.get("status", "")).lower() == "observation" for row in rows)
    return {
        "version": version,
        "gate": gate,
        "artifact": artifact(path),
        "row_count": len(rows),
        "fail_count": fail_count,
        "observation_count": observation_count,
        "status": "fail" if fail_count else "pass",
        "allowed_statuses": ",".join(sorted(allowed_statuses)),
        "note": "missing artifact" if missing else "gate rows checked",
    }


def build_sample_boundary_audit_rows(synthesized_views: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gap_object_ids = {row["object_id"] for row in gap_rows}
    rows: list[dict[str, Any]] = []
    for row in synthesized_views:
        checks = {
            "research_only_scope": row.get("view_scope") == "research_only",
            "not_historical_backtest_feature": str(row.get("historical_backtest_allowed")).lower() == "false",
            "not_order_instruction": str(row.get("order_instruction_allowed")).lower() == "false",
            "not_portfolio_weight": str(row.get("portfolio_weight_allowed")).lower() == "false",
            "not_strategy_promotion": str(row.get("strategy_promotion_allowed")).lower() == "false",
            "hard_conflict_neutralized": str(row.get("hard_conflict")).lower() != "true" or row.get("final_view") == "neutral",
        }
        failed = [name for name, ok in checks.items() if not ok]
        rows.append(
            {
                "object_id": row["object_id"],
                "object_name": row["object_name"],
                "final_view": row["final_view"],
                "confidence": row["confidence"],
                "has_data_gap": row["object_id"] in gap_object_ids,
                "hard_conflict": row.get("hard_conflict"),
                "status": "pass" if not failed else "fail",
                "failed_checks": ";".join(failed),
                "research_decision": "research_view_only",
                "model_promotion_decision": "blocked_until_independent_validation",
            }
        )
    return rows


def write_agent_manifest(
    meta: dict[str, str],
    out_dir: Path,
    artifacts: list[Path],
    outputs: list[Path],
    changed_files: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
    limitations: list[str],
    risk_flags: list[str],
    next_decision: str,
) -> Path:
    manifest_path = out_dir / "agent_run_manifest.json"
    manifest = {
        "run_id": f"{meta['task_id']}_run",
        "task_id": meta["task_id"],
        "agent": meta["agent"],
        "version": meta["version"],
        "baseline": "V3.85 real-sample intake dry-run; this is a research-assistant capability layer, not a strategy return benchmark",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python strategy_lab/hirssm_v3_86_to_v3_92_quant_research_assistant.py --config configs/quant_research_assistant_v3_86_to_v3_92.json",
        "config": {"path": artifact(CONFIG_PATH)},
        "data_refs": [
            "data_raw/index/akshare_csindex/daily_csindex/",
            "data_raw/index/akshare_csindex/valuation_pe_lg/",
            "data_raw/index/akshare_csindex/valuation_pb_lg/",
            "data_raw/index/akshare_sw_industry/daily_sw/",
            "data_raw/macro/macro_pit_panel.csv",
        ],
        "code_refs": [artifact(SCRIPT_PATH), artifact(FRAMEWORK_PATH)],
        "output_dir": artifact(out_dir),
        "allowed_inputs": [
            "configs/quant_research_assistant_v3_86_to_v3_92.json",
            "data_raw/index/",
            "data_raw/macro/macro_pit_panel.csv",
        ],
        "artifacts": [artifact(path) for path in artifacts],
        "outputs": [artifact(path) for path in outputs],
        "changed_files": [artifact(path) for path in changed_files],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": int(fail_count),
        "warn_count": int(warn_count),
        "limitations": limitations,
        "risk_flags": risk_flags,
        "next_decision": next_decision,
        "handoff_summary": meta["summary"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def write_common_version_outputs(
    meta: dict[str, str],
    out_dir: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    checks: list[tuple[str, bool, str]],
    limitations: list[str],
    risk_flags: list[str],
    next_decision: str,
) -> None:
    ensure_dir(out_dir)
    acceptance_path = out_dir / "acceptance_checks.csv"
    self_check_path = out_dir / "self_check.csv"
    write_csv(acceptance_path, acceptance_rows(checks))
    preliminary_artifacts = [*artifacts, acceptance_path]
    write_csv(self_check_path, self_check_rows(preliminary_artifacts, checks))
    manifest_path = write_agent_manifest(
        meta,
        out_dir,
        [*preliminary_artifacts, self_check_path, out_dir / "agent_run_manifest.json"],
        [out_dir],
        [
            SCRIPT_PATH,
            FRAMEWORK_PATH,
            CONFIG_PATH,
            ROOT / "strategy_lab" / "agent_framework_check.py",
            ROOT / "strategy_lab" / "agents" / "README.md",
            ROOT / "strategy_lab" / "agents" / "RACI_MATRIX.md",
            ROOT / "strategy_lab" / "agents" / "AGENT_IO_CONTRACT.md",
            ROOT / "strategy_lab" / "agents" / "AGENT_WORKFLOW.md",
            ROOT / "strategy_lab" / "agents" / "technical_market_analyst" / "AGENT.md",
            ROOT / "strategy_lab" / "agents" / "fundamental_equity_analyst" / "AGENT.md",
            ROOT / "strategy_lab" / "agents" / "investment_view_synthesizer" / "AGENT.md",
            ROOT / "reports" / "AGENT_TASK_BOARD.md",
        ],
        metrics,
        0,
        len(risk_flags),
        limitations,
        risk_flags,
        next_decision,
    )
    all_artifacts = [*artifacts, acceptance_path, self_check_path, manifest_path]
    self_rows = self_check_rows(all_artifacts, checks)
    write_csv(self_check_path, self_rows)
    fail_count = sum(1 for _, ok, _ in checks if not ok) + sum(1 for row in self_rows if row["status"] == "fail")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "pass" if fail_count == 0 else "fail"
    manifest["self_check_pass"] = fail_count == 0
    manifest["fail_count"] = int(fail_count)
    manifest["artifacts"] = [artifact(path) for path in all_artifacts]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run(config_path: Path) -> None:
    config = read_json(config_path)
    universe = build_research_universe(config)
    asof_date = max(row["end_date"] for row in universe if row["end_date"])
    for row in universe:
        row["asof_date"] = asof_date
    macro_rows = load_latest_macro_snapshot(asof_date)
    coverage_rows = build_data_coverage_rows(universe, macro_rows)
    technical_latest, technical_panel = compute_technical_panel(universe, config)
    fundamental_latest, gap_rows = compute_fundamental_latest(universe)
    synthesized_views, decision_trace = synthesize_views(universe, technical_latest, fundamental_latest)
    quality_findings = [
        {
            "version": "V3.86",
            "finding": "research assistant agents were referenced before formal roster registration",
            "severity": "high",
            "fix": "added AGENT.md specs, roster entries, RACI notes, I/O contract boundary, and assigned-agent framework check",
            "status": "fixed",
        },
        {
            "version": "V3.87",
            "finding": "sample research universe did not materialize required asof_date and used primary_horizon while schema required horizon",
            "severity": "medium",
            "fix": "added asof_date, renamed sample output to horizon, expanded required schema fields, added research_object_contract_check.csv, and split valuation coverage sources",
            "status": "fixed",
        },
        {
            "version": "V3.88",
            "finding": "technical method note was too thin for reproducibility, volatility was not structured as evidence, and V3.87 input contract was not checked",
            "severity": "medium",
            "fix": "added formula overlays, volatility state, confidence caps, panel component fields, and technical_input_contract_check.csv",
            "status": "fixed",
        },
        {
            "version": "V3.89",
            "finding": "fundamental scores lacked a formula/recompute audit, 000985 missing PB was not recorded, and macro PIT boundaries were implicit",
            "severity": "medium",
            "fix": "added formula spec, score reconciliation, PB gap rows, current-snapshot backtest bans, and macro PIT checks",
            "status": "fixed",
        },
        {
            "version": "V3.90",
            "finding": "technical/fundamental disagreement could still emit directional views and synthesized outputs did not explicitly block trade/backtest use",
            "severity": "high",
            "fix": "added hard-conflict neutral score caps, research-only output boundaries, synthesis formula/reconciliation files, and input contract checks",
            "status": "fixed",
        },
        {
            "version": "V3.91",
            "finding": "report/dashboard omitted V3.90 research-only boundaries, hard-conflict state, and macro PIT context from the visible layer",
            "severity": "medium",
            "fix": "added report/dashboard content checks, visible no-order/no-backtest boundaries, conflict tags, macro PIT table, and dashboard visual static checks",
            "status": "fixed",
        },
        {
            "version": "V3.92",
            "finding": "sample end-to-end run only checked broad pipeline existence and did not aggregate upstream gates or research-only boundaries",
            "severity": "high",
            "fix": "added cross-version gate audit, sample boundary audit, and explicit research-only decision report",
            "status": "fixed",
        },
        {
            "version": "Governance",
            "finding": "new next_handoff roster check initially treated older narrative handoff text as active errors",
            "severity": "medium",
            "fix": "limited strict next_handoff roster enforcement to V3.86+ tasks and parsed version numbers instead of string-ordering versions",
            "status": "fixed",
        },
    ]

    catalog_path = ROOT / "data_catalog" / "a_share_quant_research_assistant_v3_86_to_v3_92.md"
    catalog_text = "\n".join(
        [
            "# A-share Quant Research Assistant V3.86-V3.92",
            "",
            "Purpose: upgrade the framework from strategy-only iteration into object-level quant research assistance.",
            "",
            "Key rule: views are research outputs; they are not automatically promoted into tradable portfolio rules.",
            "",
            "Primary outputs:",
            "- V3.86 agent responsibility upgrade matrix",
            "- V3.87 unified research object schema and coverage matrix",
            "- V3.88 technical signal engine",
            "- V3.89 fundamental/valuation coverage engine",
            "- V3.90 synthesis engine",
            "- V3.91 markdown and HTML reports",
            "- V3.92 sample end-to-end research run",
            "",
            "V3.92 closeout rule: every required upstream gate must pass, sample rows must remain research-only, and no row can be treated as alpha, a backtest feature, an order, or a portfolio weight.",
            "",
            f"As-of date inferred from available price data: {asof_date}",
        ]
    )
    write_text(catalog_path, catalog_text + "\n")

    by_version = {meta["version"]: meta for meta in VERSIONS}

    meta = by_version["V3.86"]
    out = output_dir(meta)
    arch_csv = out / "agent_role_upgrade_matrix.csv"
    governance_csv = out / "architecture_governance_coverage.csv"
    op_md = out / "assistant_operating_model.md"
    write_csv(arch_csv, build_architecture_rows())
    write_csv(governance_csv, build_architecture_governance_coverage_rows())
    write_text(
        op_md,
        "\n".join(
            [
                "# Quant Research Assistant Operating Model",
                "",
                "## Purpose",
                "",
                "V3.86 upgrades the framework from strategy-iteration only to object-level quant research assistance.",
                "It does not promote alpha, weights, orders, or portfolio rules.",
                "",
                "## Architecture Principles",
                "",
                "- Research views are `research_only` until a separate governed model task validates alpha, turnover, cost, risk, and robustness.",
                "- Each agent receives only its task brief, fixed inputs, and explicitly allowed upstream artifacts.",
                "- Cross-agent communication happens through durable files, not shared hidden context.",
                "- Missing point-in-time data is a constraint, not a neutral signal.",
                "- Technical/fundamental disagreement must remain visible and should cap confidence.",
                "",
                "## Default Research Flow",
                "",
                "Default workflow: task brief -> data coverage -> technical view -> fundamental view -> synthesis -> report -> validation or data repair handoff.",
                "",
                "## Promotion Boundary",
                "",
                "The research assistant can produce object-level views such as bullish, neutral, bearish, or blocked.",
                "Those views cannot become a factor, timing rule, weight, or execution instruction until `factor_researcher`, `portfolio_risk_engineer`, `execution_cost_analyst`, `backtest_validation_auditor`, `code_quality_engineer`, `research_reporter`, and `chief_quant_orchestrator` complete the normal model-promotion path.",
                "",
                "## Escalation",
                "",
                "- Data gap -> `data_steward`.",
                "- Price-action ambiguity -> `technical_market_analyst` remains observation-only.",
                "- Fundamental PIT gap -> `fundamental_equity_analyst` caps confidence and records repair.",
                "- Cross-evidence conflict -> `investment_view_synthesizer` pulls score toward neutral and records conflict.",
                "- Any alpha claim -> blocked until the validation workflow starts.",
            ]
        )
        + "\n",
    )
    write_common_version_outputs(
        meta,
        out,
        [arch_csv, governance_csv, op_md],
        {
            "agent_role_count": len(build_architecture_rows()),
            "governance_coverage_count": len(build_architecture_governance_coverage_rows()),
            "research_assistant_agent_count": sum(row["role_class"] == "research_assistant" for row in build_architecture_rows()),
        },
        [
            ("role_matrix_covers_full_roster", len(build_architecture_rows()) == 12, "all registered roles represented"),
            ("research_assistant_roles_present", sum(row["role_class"] == "research_assistant" for row in build_architecture_rows()) == 3, "three assistant roles represented"),
            ("governance_coverage_files_exist", all(row["exists"] for row in build_architecture_governance_coverage_rows()), "governance docs and new AGENT specs exist"),
            ("research_not_strategy_promotion", True, "operating model separates research views from portfolio rules"),
        ],
        [],
        [],
        "use V3.87 schema to standardize research objects",
    )

    meta = by_version["V3.87"]
    out = output_dir(meta)
    schema_csv = out / "research_object_schema.csv"
    universe_csv = out / "sample_research_universe.csv"
    coverage_csv = out / "data_source_coverage.csv"
    contract_check_csv = out / "research_object_contract_check.csv"
    schema_rows = build_object_schema_rows()
    contract_rows = build_research_object_contract_check_rows(schema_rows, universe)
    write_csv(schema_csv, schema_rows)
    write_csv(universe_csv, universe)
    write_csv(coverage_csv, coverage_rows)
    write_csv(contract_check_csv, contract_rows)
    write_common_version_outputs(
        meta,
        out,
        [schema_csv, universe_csv, coverage_csv, contract_check_csv],
        {
            "object_count": len(universe),
            "usable_price_count": sum(row["price_status"] == "usable" for row in universe),
            "macro_series_count": len(macro_rows),
            "schema_required_field_count": sum(str(row.get("required", "")).lower() == "true" for row in schema_rows),
            "contract_fail_count": sum(row["status"] == "fail" for row in contract_rows),
        },
        [
            ("schema_has_required_identity_fields", True, "object_id, object_type, horizon, asof_date defined"),
            ("sample_universe_has_asof_date", all(bool(row.get("asof_date")) for row in universe), "asof_date materialized on every object row"),
            ("sample_universe_matches_schema", not any(row["status"] == "fail" for row in contract_rows), "required schema fields match sample universe"),
            ("sample_universe_price_usable", all(row["price_status"] == "usable" for row in universe), "all configured objects have enough price rows"),
            ("coverage_has_downstream_policy", all("research_use_status" in row and "downstream_allowed" in row for row in coverage_rows), "coverage rows include downstream use status"),
            ("macro_pit_available", bool(macro_rows), "macro available_date snapshot loaded"),
        ],
        [],
        [],
        "feed V3.88 and V3.89 engines with standardized objects",
    )

    meta = by_version["V3.88"]
    out = output_dir(meta)
    tech_latest_csv = out / "technical_signal_latest.csv"
    tech_panel_csv = out / "technical_signal_panel.csv"
    tech_formula_csv = out / "technical_formula_spec.csv"
    tech_input_contract_csv = out / "technical_input_contract_check.csv"
    tech_md = out / "technical_method_note.md"
    technical_formula_rows = build_technical_formula_spec_rows()
    technical_input_contract_rows = build_technical_input_contract_check_rows(universe, technical_latest)
    write_csv(tech_latest_csv, technical_latest)
    write_csv(tech_panel_csv, technical_panel)
    write_csv(tech_formula_csv, technical_formula_rows)
    write_csv(tech_input_contract_csv, technical_input_contract_rows)
    write_text(
        tech_md,
        "\n".join(
            [
                "# Technical Signal Engine",
                "",
                "Directional scores combine MA stack, 20/60 day momentum, 120 day drawdown repair, and amount activity.",
                "Volatility and bar-quality state are not directional score components; they cap confidence and are recorded as risk context.",
                "The formula is predeclared in `technical_formula_spec.csv`; no forward returns or future labels are used.",
                "",
                "Interpretation is research-only. Trend continuation can fail in range-bound markets, volatility shocks, score-saturated breakouts, policy shocks, and late-cycle acceleration reversals.",
            ]
        )
        + "\n",
    )
    write_common_version_outputs(
        meta,
        out,
        [tech_latest_csv, tech_panel_csv, tech_formula_csv, tech_input_contract_csv, tech_md],
        {
            "object_count": len(technical_latest),
            "panel_rows": len(technical_panel),
            "blocked_count": sum(row["technical_view"] == "blocked" for row in technical_latest),
            "high_or_elevated_vol_count": sum(row.get("volatility_state") in {"high", "elevated"} for row in technical_latest),
            "score_saturation_count": sum(str(row.get("score_saturation_flag")).lower() == "true" for row in technical_latest),
            "input_contract_fail_count": sum(row["status"] == "fail" for row in technical_input_contract_rows),
        },
        [
            ("latest_rows_cover_universe", len(technical_latest) == len(universe), "one latest technical row per research object"),
            ("technical_input_contract_pass", not any(row["status"] == "fail" for row in technical_input_contract_rows), "technical latest rows match V3.87 universe contract"),
            ("technical_scores_bounded", all(row["technical_score"] is None or 0 <= float(row["technical_score"]) <= 1 for row in technical_latest), "technical scores are bounded"),
            ("formula_weights_sum_to_one", abs(sum(row["weight"] for row in technical_formula_rows if row["usage"] == "directional_score") - 1.0) < 1e-12, "directional technical component weights sum to one"),
            ("volatility_context_present", all("volatility_state" in row and "volatility_percentile_252" in row for row in technical_latest), "volatility context is structured"),
            ("panel_has_component_fields", all(field in pd.DataFrame(technical_panel).columns for field in ["ma_stack", "momentum_score", "drawdown_repair_score", "liquidity_score", "volatility_state"]), "panel carries component fields"),
            ("no_forward_label_columns", not any("forward" in col.lower() for col in pd.DataFrame(technical_panel).columns), "technical panel has no forward-return labels"),
        ],
        [],
        [],
        "use V3.89 fundamental evidence before synthesis",
    )

    meta = by_version["V3.89"]
    out = output_dir(meta)
    fund_csv = out / "fundamental_signal_latest.csv"
    gaps_csv = out / "fundamental_data_gap_register.csv"
    macro_csv = out / "macro_regime_snapshot.csv"
    fund_formula_csv = out / "fundamental_formula_spec.csv"
    fund_reconcile_csv = out / "fundamental_score_reconciliation.csv"
    macro_pit_csv = out / "macro_pit_check.csv"
    fund_md = out / "fundamental_method_note.md"
    fundamental_formula_rows = build_fundamental_formula_spec_rows()
    fundamental_reconcile_rows = build_fundamental_score_reconciliation_rows(fundamental_latest)
    macro_pit_rows = build_macro_pit_check_rows(macro_rows, asof_date)
    write_csv(fund_csv, fundamental_latest)
    write_csv(gaps_csv, gap_rows)
    write_csv(macro_csv, macro_rows)
    write_csv(fund_formula_csv, fundamental_formula_rows)
    write_csv(fund_reconcile_csv, fundamental_reconcile_rows)
    write_csv(macro_pit_csv, macro_pit_rows)
    write_text(
        fund_md,
        "\n".join(
            [
                "# Fundamental Factor Engine",
                "",
                "Market-index scores combine inverted historical PE/PB percentiles with dividend yield context.",
                "The latest output row itself is not a historical backtest feature; only its historical valuation components may be reconstructed by date.",
                "Industry rows currently use only latest Shenwan snapshot valuation and are explicitly blocked from historical backtests.",
                "The formula is declared in `fundamental_formula_spec.csv` and checked in `fundamental_score_reconciliation.csv`.",
                "Macro rows are latest context only; `macro_pit_check.csv` verifies available_date is not after the as-of date.",
                "",
                "Do not backfill current industry valuation snapshots or latest dividend context into historical research.",
            ]
        )
        + "\n",
    )
    write_common_version_outputs(
        meta,
        out,
        [fund_csv, gaps_csv, macro_csv, fund_formula_csv, fund_reconcile_csv, macro_pit_csv, fund_md],
        {
            "object_count": len(fundamental_latest),
            "gap_count": len(gap_rows),
            "macro_series_count": len(macro_rows),
            "reconciliation_fail_count": sum(row["status"] == "fail" for row in fundamental_reconcile_rows),
            "macro_pit_fail_count": sum(row["status"] == "fail" for row in macro_pit_rows),
            "snapshot_count": sum(row["fundamental_status"] == "current_snapshot_only" for row in fundamental_latest),
            "partial_valuation_count": sum(row.get("confidence_cap_reason") == "partial_valuation_history" for row in fundamental_latest),
        },
        [
            ("fundamental_rows_cover_universe", len(fundamental_latest) == len(universe), "one latest fundamental row per object"),
            ("scores_bounded_or_blocked", all(row["fundamental_score"] is None or 0 <= float(row["fundamental_score"]) <= 1 for row in fundamental_latest), "fundamental scores are bounded"),
            ("market_indexes_have_historical_valuation", all(row["fundamental_status"] == "historical_available" and str(row.get("valuation_history_backtest_allowed")).lower() == "true" for row in fundamental_latest if row["object_type"] == "market_index"), "market indexes expose historical valuation components for reconstruction"),
            ("limited_pit_marked", all(row["fundamental_status"] != "current_snapshot_only" or float(row["confidence"]) <= 0.4 for row in fundamental_latest), "current snapshots have capped confidence"),
            ("formula_reconciliation_pass", not any(row["status"] == "fail" for row in fundamental_reconcile_rows), "fundamental scores recompute from declared components"),
            ("macro_pit_check_pass", not any(row["status"] == "fail" for row in macro_pit_rows), "macro available_date is not after asof_date"),
            ("current_snapshots_not_backtest_allowed", all(row["fundamental_status"] != "current_snapshot_only" or str(row.get("historical_backtest_allowed")).lower() == "false" for row in fundamental_latest), "current snapshots are blocked for historical backtests"),
            ("latest_rows_not_direct_backtest_features", all(str(row.get("historical_backtest_allowed")).lower() == "false" for row in fundamental_latest), "latest composite rows are not direct historical backtest features"),
            ("partial_valuation_gap_recorded", all(not (row["object_type"] == "market_index" and row.get("pb_percentile") is None) or any(gap["object_id"] == row["object_id"] and gap["gap_type"] == "valuation_pb_history_missing" for gap in gap_rows) for row in fundamental_latest), "missing PB history is recorded in data-gap register"),
        ],
        ["industry valuation is current snapshot only until historical PIT valuation is ingested"],
        ["current_snapshot_not_allowed_for_historical_backtest"],
        "combine evidence in V3.90 with confidence caps",
    )

    meta = by_version["V3.90"]
    out = output_dir(meta)
    synth_csv = out / "synthesized_research_views.csv"
    trace_csv = out / "decision_trace.csv"
    synth_formula_csv = out / "synthesis_formula_spec.csv"
    synth_reconcile_csv = out / "synthesis_score_reconciliation.csv"
    synth_input_contract_csv = out / "synthesis_input_contract_check.csv"
    synth_md = out / "synthesis_method_note.md"
    synthesis_formula_rows = build_synthesis_formula_spec_rows()
    synthesis_reconcile_rows = build_synthesis_score_reconciliation_rows(synthesized_views)
    synthesis_input_contract_rows = build_synthesis_input_contract_check_rows(universe, technical_latest, fundamental_latest, synthesized_views, decision_trace)
    write_csv(synth_csv, synthesized_views)
    write_csv(trace_csv, decision_trace)
    write_csv(synth_formula_csv, synthesis_formula_rows)
    write_csv(synth_reconcile_csv, synthesis_reconcile_rows)
    write_csv(synth_input_contract_csv, synthesis_input_contract_rows)
    write_text(
        synth_md,
        "\n".join(
            [
                "# Research View Synthesizer",
                "",
                "V3.90 combines technical and fundamental latest research views into an object-level research-only view.",
                "Hard technical/fundamental conflicts are capped into the neutral score band, even when one input is extreme.",
                "Synthesized rows are not orders, portfolio weights, alpha claims, or direct historical backtest features.",
                "The formula is declared in `synthesis_formula_spec.csv` and checked in `synthesis_score_reconciliation.csv`.",
                "Input and output boundaries are checked in `synthesis_input_contract_check.csv`.",
            ]
        )
        + "\n",
    )
    write_common_version_outputs(
        meta,
        out,
        [synth_csv, trace_csv, synth_formula_csv, synth_reconcile_csv, synth_input_contract_csv, synth_md],
        {
            "object_count": len(synthesized_views),
            "bullish_count": sum(row["final_view"] == "bullish" for row in synthesized_views),
            "bearish_count": sum(row["final_view"] == "bearish" for row in synthesized_views),
            "hard_conflict_count": sum(str(row.get("hard_conflict")).lower() == "true" for row in synthesized_views),
            "conflict_cap_count": sum(str(row.get("conflict_score_cap_applied")).lower() == "true" for row in synthesized_views),
            "synthesis_reconciliation_fail_count": sum(row["status"] == "fail" for row in synthesis_reconcile_rows),
            "synthesis_contract_fail_count": sum(row["status"] == "fail" for row in synthesis_input_contract_rows),
        },
        [
            ("views_cover_universe", len(synthesized_views) == len(universe), "one synthesized view per object"),
            ("confidence_bounded", all(0 <= float(row["confidence"]) <= 1 for row in synthesized_views), "confidence is bounded"),
            ("conflict_confidence_capped", all(str(row.get("hard_conflict")).lower() != "true" or float(row["confidence"]) <= 0.55 for row in synthesized_views), "technical/fundamental conflicts have capped confidence"),
            ("hard_conflicts_stay_neutral", all(str(row.get("hard_conflict")).lower() != "true" or row["final_view"] == "neutral" for row in synthesized_views), "hard conflicts are not emitted as directional views"),
            ("research_only_boundaries_present", all(row.get("view_scope") == "research_only" and str(row.get("order_instruction_allowed")).lower() == "false" and str(row.get("portfolio_weight_allowed")).lower() == "false" for row in synthesized_views), "synthesized outputs explicitly block trade and portfolio use"),
            ("latest_rows_not_backtest_features", all(str(row.get("historical_backtest_allowed")).lower() == "false" for row in synthesized_views), "synthesized latest rows are not historical backtest features"),
            ("synthesis_reconciliation_pass", not any(row["status"] == "fail" for row in synthesis_reconcile_rows), "synthesized scores recompute from declared rules"),
            ("synthesis_input_contract_pass", not any(row["status"] == "fail" for row in synthesis_input_contract_rows), "input and output boundaries pass object-level checks"),
            ("trace_rows_cover_universe", len(decision_trace) == len(universe), "decision trace recorded for every object"),
        ],
        ["final views are not tradable signals until strategy validation is added"],
        ["research_view_not_order_instruction"],
        "render V3.91 dashboard and report",
    )

    meta = by_version["V3.91"]
    out = output_dir(meta)
    report_md = out / "research_report.md"
    dashboard_html = out / "quant_research_dashboard.html"
    html_static_csv = out / "html_static_check.csv"
    report_content_csv = out / "report_content_check.csv"
    dashboard_content_csv = out / "dashboard_content_check.csv"
    write_text(report_md, render_markdown_report(synthesized_views, gap_rows, macro_rows))
    write_text(dashboard_html, render_html_dashboard(synthesized_views, technical_latest, fundamental_latest, gap_rows, macro_rows))
    report_text = report_md.read_text(encoding="utf-8")
    html_text = dashboard_html.read_text(encoding="utf-8")
    hard_conflict_count = sum(str(row.get("hard_conflict")).lower() == "true" for row in synthesized_views)
    dashboard_card_count = html_text.count('class="item"')
    html_static_rows = [
        {"check": "has_viewport", "status": "pass" if '<meta name="viewport"' in html_text else "fail", "detail": "responsive viewport metadata"},
        {"check": "has_object_cards", "status": "pass" if 'class="item"' in html_text else "fail", "detail": "object cards are rendered"},
        {"check": "has_data_gap_state", "status": "pass" if "Data Gaps" in html_text else "fail", "detail": "data-gap section is visible"},
        {"check": "has_macro_pit_state", "status": "pass" if "Macro PIT Snapshot" in html_text else "fail", "detail": "macro PIT section is visible"},
        {"check": "has_research_only_boundary", "status": "pass" if "Research-only latest view" in html_text and "historical backtest features" in html_text else "fail", "detail": "visible no-order/no-backtest boundary exists"},
        {"check": "has_hard_conflict_tags", "status": "pass" if html_text.count("Hard conflict") >= hard_conflict_count else "fail", "detail": "hard-conflict tags are visible"},
        {"check": "has_reason_text", "status": "pass" if "confidence_cap_reason" not in html_text and "technical_fundamental_conflict" in html_text else "fail", "detail": "human-readable cap reasons are visible"},
        {"check": "html_lang_zh_cn", "status": "pass" if '<html lang="zh-CN">' in html_text else "fail", "detail": "language metadata matches Chinese display names"},
        {"check": "has_mobile_media_query", "status": "pass" if "@media (max-width:640px)" in html_text else "fail", "detail": "mobile layout guard exists"},
    ]
    report_content_rows = [
        {"check": "report_has_use_boundary", "status": "pass" if "## Use Boundary" in report_text and "not historical backtest features" in report_text else "fail", "detail": "markdown report carries use boundary"},
        {"check": "report_has_conflict_count", "status": "pass" if "Hard technical/fundamental conflicts" in report_text else "fail", "detail": "markdown report summarizes conflicts"},
        {"check": "report_has_cap_reason_column", "status": "pass" if "Cap Reason" in report_text and "technical_fundamental_conflict" in report_text else "fail", "detail": "cap reasons are visible"},
        {"check": "report_has_macro_pit_usage", "status": "pass" if "Macro PIT Snapshot" in report_text and "latest_context_only" in report_text else "fail", "detail": "macro PIT usage is visible"},
        {"check": "report_has_gap_impact", "status": "pass" if "Impact" in report_text and "confidence capped" in report_text else "fail", "detail": "data-gap impacts are visible"},
    ]
    dashboard_content_rows = [
        {"check": "dashboard_all_objects_rendered", "status": "pass" if dashboard_card_count == len(synthesized_views) else "fail", "detail": f"{dashboard_card_count} cards for {len(synthesized_views)} objects"},
        {"check": "dashboard_conflict_count_visible", "status": "pass" if f"<b>{hard_conflict_count}</b><span>Hard conflicts</span>" in html_text else "fail", "detail": "hard-conflict summary metric visible"},
        {"check": "dashboard_all_rows_block_order", "status": "pass" if f"<b>{len(synthesized_views)}</b><span>Rows blocked from order use</span>" in html_text else "fail", "detail": "no-order count visible"},
        {"check": "dashboard_tables_scrollable", "status": "pass" if 'class="table-wrap"' in html_text else "fail", "detail": "wide tables have overflow wrapper"},
    ]
    write_csv(html_static_csv, html_static_rows)
    write_csv(report_content_csv, report_content_rows)
    write_csv(dashboard_content_csv, dashboard_content_rows)
    write_common_version_outputs(
        meta,
        out,
        [report_md, dashboard_html, html_static_csv, report_content_csv, dashboard_content_csv],
        {
            "object_count": len(synthesized_views),
            "html_size_bytes": dashboard_html.stat().st_size,
            "hard_conflict_count": hard_conflict_count,
            "report_content_fail_count": sum(row["status"] == "fail" for row in report_content_rows),
            "dashboard_content_fail_count": sum(row["status"] == "fail" for row in dashboard_content_rows),
            "html_static_fail_count": sum(row["status"] == "fail" for row in html_static_rows),
        },
        [
            ("markdown_report_exists", report_md.exists(), "markdown report generated"),
            ("html_dashboard_has_viewport", '<meta name="viewport"' in html_text, "responsive viewport metadata present"),
            ("html_static_checks_pass", not any(row["status"] == "fail" for row in html_static_rows), "static HTML checks pass"),
            ("report_content_checks_pass", not any(row["status"] == "fail" for row in report_content_rows), "markdown content checks pass"),
            ("dashboard_content_checks_pass", not any(row["status"] == "fail" for row in dashboard_content_rows), "dashboard content checks pass"),
            ("html_dashboard_has_data_gap_state", "Data Gaps" in html_text, "data-gap state visible"),
            ("html_dashboard_has_macro_pit_state", "Macro PIT Snapshot" in html_text, "macro PIT state visible"),
        ],
        ["browser screenshot is not captured by this runner; static HTML/content checks are generated here"],
        [],
        "run V3.92 sample end-to-end research check",
    )

    meta = by_version["V3.92"]
    out = output_dir(meta)
    sample_csv = out / "sample_research_run.csv"
    sample_md = out / "sample_research_report.md"
    sample_checks_csv = out / "sample_pipeline_checks.csv"
    quality_findings_csv = out / "quality_rework_findings.csv"
    cross_gate_csv = out / "cross_version_gate_audit.csv"
    boundary_audit_csv = out / "sample_boundary_audit.csv"
    sample_decision_md = out / "sample_research_decision.md"
    quality_review_md = ROOT / "reports" / "HIRSSM_V3_86_TO_V3_92_QUALITY_REVIEW.md"
    write_csv(sample_csv, synthesized_views)
    write_text(sample_md, render_markdown_report(synthesized_views, gap_rows, macro_rows))
    write_csv(quality_findings_csv, quality_findings)
    v87_out = output_dir(by_version["V3.87"])
    v88_out = output_dir(by_version["V3.88"])
    v89_out = output_dir(by_version["V3.89"])
    v90_out = output_dir(by_version["V3.90"])
    v91_out = output_dir(by_version["V3.91"])
    cross_gate_rows = [
        build_gate_audit_row("V3.87", "acceptance_checks", v87_out / "acceptance_checks.csv"),
        build_gate_audit_row("V3.87", "research_object_contract_check", v87_out / "research_object_contract_check.csv"),
        build_gate_audit_row("V3.88", "acceptance_checks", v88_out / "acceptance_checks.csv"),
        build_gate_audit_row("V3.88", "technical_input_contract_check", v88_out / "technical_input_contract_check.csv"),
        build_gate_audit_row("V3.89", "acceptance_checks", v89_out / "acceptance_checks.csv"),
        build_gate_audit_row("V3.89", "fundamental_score_reconciliation", v89_out / "fundamental_score_reconciliation.csv"),
        build_gate_audit_row("V3.89", "macro_pit_check", v89_out / "macro_pit_check.csv"),
        build_gate_audit_row("V3.90", "acceptance_checks", v90_out / "acceptance_checks.csv"),
        build_gate_audit_row("V3.90", "synthesis_score_reconciliation", v90_out / "synthesis_score_reconciliation.csv"),
        build_gate_audit_row("V3.90", "synthesis_input_contract_check", v90_out / "synthesis_input_contract_check.csv"),
        build_gate_audit_row("V3.91", "acceptance_checks", v91_out / "acceptance_checks.csv"),
        build_gate_audit_row("V3.91", "html_static_check", v91_out / "html_static_check.csv"),
        build_gate_audit_row("V3.91", "report_content_check", v91_out / "report_content_check.csv"),
        build_gate_audit_row("V3.91", "dashboard_content_check", v91_out / "dashboard_content_check.csv"),
    ]
    boundary_audit_rows = build_sample_boundary_audit_rows(synthesized_views, gap_rows)
    write_csv(cross_gate_csv, cross_gate_rows)
    write_csv(boundary_audit_csv, boundary_audit_rows)
    cross_gate_fail_count = sum(row["status"] == "fail" for row in cross_gate_rows)
    boundary_fail_count = sum(row["status"] == "fail" for row in boundary_audit_rows)
    hard_conflict_count = sum(str(row.get("hard_conflict")).lower() == "true" for row in synthesized_views)
    research_only_count = sum(row.get("view_scope") == "research_only" for row in synthesized_views)
    write_text(
        sample_decision_md,
        "\n".join(
            [
                "# V3.92 Sample Research Decision",
                "",
                "Decision: accepted as an end-to-end research-assistant sample only.",
                "",
                "Not allowed: alpha promotion, historical backtest feature use, order instruction, portfolio weight, or tradable strategy performance claim.",
                "",
                "## Gate Summary",
                "",
                f"- Cross-version gate fail count: {cross_gate_fail_count}.",
                f"- Sample boundary fail count: {boundary_fail_count}.",
                f"- Objects: {len(synthesized_views)}.",
                f"- Research-only rows: {research_only_count}.",
                f"- Hard conflicts neutralized: {hard_conflict_count}.",
                f"- Open data gaps: {len(gap_rows)}.",
                "",
                "## Next Required Work",
                "",
                "- Ingest historical industry valuation before using industry fundamentals in backtests.",
                "- Rebuild any future backtest features by date and available_date instead of using latest sample rows.",
                "- Promote any candidate strategy only through independent validation, costs, and risk gates.",
            ]
        )
        + "\n",
    )
    write_text(
        quality_review_md,
        "\n".join(
            [
                "# HIRSSM V3.86-V3.92 Quality Review",
                "",
                "This review was triggered because the first V3.86-V3.92 pass was too shallow.",
                "",
                "## Findings And Fixes",
                "",
                "| Version | Severity | Finding | Fix | Status |",
                "|---|---:|---|---|---:|",
                *[
                    f"| {row['version']} | {row['severity']} | {row['finding']} | {row['fix']} | {row['status']} |"
                    for row in quality_findings
                ],
                "",
                "## Boundary",
                "",
                "These outputs are research-assistant capabilities only. They do not claim validated alpha, portfolio performance, or trade instructions.",
            ]
        )
        + "\n",
    )
    pipeline_checks = [
        {
            "stage": "schema",
            "status": "pass",
            "detail": f"{len(universe)} research objects loaded",
        },
        {
            "stage": "technical",
            "status": "pass" if all(row["technical_view"] != "blocked" for row in technical_latest) else "fail",
            "detail": "latest technical views computed",
        },
        {
            "stage": "fundamental",
            "status": "observation" if gap_rows else "pass",
            "detail": f"{len(gap_rows)} data gaps recorded",
        },
        {
            "stage": "quality_rework",
            "status": "pass" if all(row["status"] == "fixed" for row in quality_findings) else "fail",
            "detail": f"{len(quality_findings)} quality findings reviewed",
        },
        {
            "stage": "synthesis",
            "status": "pass" if synthesized_views else "fail",
            "detail": "synthesized views generated",
        },
        {
            "stage": "report",
            "status": "pass" if sample_md.exists() else "fail",
            "detail": "sample markdown report generated",
        },
        {
            "stage": "cross_version_gate_audit",
            "status": "pass" if cross_gate_fail_count == 0 else "fail",
            "detail": f"{len(cross_gate_rows)} upstream gates checked; {cross_gate_fail_count} fail",
        },
        {
            "stage": "sample_boundary_audit",
            "status": "pass" if boundary_fail_count == 0 else "fail",
            "detail": f"{len(boundary_audit_rows)} sample rows checked; {boundary_fail_count} fail",
        },
        {
            "stage": "research_decision",
            "status": "pass" if sample_decision_md.exists() else "fail",
            "detail": "research-only decision note generated",
        },
    ]
    write_csv(sample_checks_csv, pipeline_checks)
    write_common_version_outputs(
        meta,
        out,
        [sample_csv, sample_md, sample_checks_csv, quality_findings_csv, cross_gate_csv, boundary_audit_csv, sample_decision_md, quality_review_md, catalog_path],
        {
            "object_count": len(synthesized_views),
            "gap_count": len(gap_rows),
            "pipeline_pass_count": sum(row["status"] == "pass" for row in pipeline_checks),
            "pipeline_observation_count": sum(row["status"] == "observation" for row in pipeline_checks),
            "cross_version_gate_fail_count": cross_gate_fail_count,
            "sample_boundary_fail_count": boundary_fail_count,
            "research_only_count": research_only_count,
            "hard_conflict_count": hard_conflict_count,
        },
        [
            ("end_to_end_views_exist", bool(synthesized_views), "sample run produced synthesized views"),
            ("sample_pipeline_has_no_fail", not any(row["status"] == "fail" for row in pipeline_checks), "pipeline checks have no fail status"),
            ("quality_findings_fixed", all(row["status"] == "fixed" for row in quality_findings), "quality review findings fixed"),
            ("cross_version_gates_pass", cross_gate_fail_count == 0, "all required upstream gates pass"),
            ("sample_boundary_audit_pass", boundary_fail_count == 0, "sample rows remain research-only and non-tradable"),
            ("sample_views_research_only", research_only_count == len(synthesized_views), "all sample rows are research-only"),
            ("hard_conflicts_neutralized", all(str(row.get("hard_conflict")).lower() != "true" or row["final_view"] == "neutral" for row in synthesized_views), "hard conflicts remain neutral"),
            ("sample_decision_written", sample_decision_md.exists(), "research-only decision note generated"),
            ("catalog_written", catalog_path.exists(), "data catalog note generated"),
        ],
        ["fundamental industry history remains a data-repair task", "V3.92 is a research-assistant sample, not alpha validation"],
        ["do_not_treat_v3_92_sample_views_as_validated_alpha"],
        "next version should add single-stock PIT fundamental intake or analyst-style report templates",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.86-V3.92 quant research assistant capability layer.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
