#!/usr/bin/env python
"""One-command quant model system orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import a_share_low_cost_factor_builder as lowcost
import csv_io
import data_quality_report as dq
import factor_factory_ledger as ledger
import factor_factory_walk_forward as wf
import factor_registry_audit as registry_audit
import model_run_report as run_report
import paper_trading_monitor as paper


DEFAULT_ROOT = Path("Introduction-to-Quantitative-Finance")


def validate_panel(panel: pd.DataFrame, registry: pd.DataFrame, config: wf.WalkForwardConfig) -> pd.DataFrame:
    """Validate minimum input contract before running a model."""
    rows = []
    required = [config.date_col, config.asset_col, config.forward_return_col]
    for col in required:
        rows.append({"gate": f"required_column:{col}", "status": "pass" if col in panel.columns else "fail", "detail": col})
    if config.industry_col:
        rows.append(
            {
                "gate": f"recommended_column:{config.industry_col}",
                "status": "pass" if config.industry_col in panel.columns else "warn",
                "detail": config.industry_col,
            }
        )
    for col in config.control_cols:
        rows.append({"gate": f"control_column:{col}", "status": "pass" if col in panel.columns else "warn", "detail": col})
    if {config.date_col, config.asset_col}.issubset(panel.columns):
        duplicated = int(panel.duplicated([config.date_col, config.asset_col]).sum())
        rows.append({"gate": "duplicate_date_asset", "status": "pass" if duplicated == 0 else "fail", "detail": str(duplicated)})
        rows.append({"gate": "date_count", "status": "pass" if panel[config.date_col].nunique() >= config.train_periods + config.test_periods else "fail", "detail": str(panel[config.date_col].nunique())})
        rows.append({"gate": "asset_count", "status": "pass" if panel[config.asset_col].nunique() >= config.min_assets else "fail", "detail": str(panel[config.asset_col].nunique())})
    available_factors = [col for col in registry["column"].dropna().astype(str).tolist() if col in panel.columns]
    rows.append({"gate": "registered_factor_columns", "status": "pass" if available_factors else "fail", "detail": str(len(available_factors))})
    if config.forward_return_col in panel.columns:
        coverage = float(pd.to_numeric(panel[config.forward_return_col], errors="coerce").notna().mean())
        rows.append({"gate": "label_coverage", "status": "pass" if coverage >= 0.5 else "warn", "detail": f"{coverage:.4f}"})
    return pd.DataFrame(rows)


def write_markdown_summary(output_dir: Path, title: str, tables: dict[str, pd.DataFrame]) -> None:
    lines = [f"# {title}", ""]
    for name, table in tables.items():
        lines.extend([f"## {name}", ""])
        if table.empty:
            lines.extend(["empty", ""])
        else:
            lines.extend(["```text", table.head(20).to_string(index=False), "```", ""])
    (output_dir / "SYSTEM_RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_data_quality(output_dir: Path, panel: pd.DataFrame, registry: pd.DataFrame, config: wf.WalkForwardConfig) -> dict[str, pd.DataFrame]:
    report = dq.build_quality_report(panel, registry, config)
    dq.save_report(report, output_dir / "data_quality")
    gates = report["gates"]
    if not gates.empty and (gates["status"] == "fail").any():
        failed = gates.loc[gates["status"] == "fail", "gate"].tolist()
        raise ValueError(f"Data-quality gates failed before model run: {failed}. See data_quality/gates.csv.")
    return report


def run_preflight_from_files(
    panel_csv: str,
    registry_csv: str,
    config_json: str,
    output_dir: str,
) -> Path:
    """Run data-quality, panel, and registry checks without running a backtest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = csv_io.read_csv_robust(panel_csv)
    registry = csv_io.read_csv_robust(registry_csv)
    config = wf.load_config(config_json)
    audit = registry_audit.audit_registry(registry)
    for name, table in audit.items():
        table.to_csv(out / f"registry_{name}.csv", index=False, encoding="utf-8-sig")
    quality_report = dq.build_quality_report(panel, registry, config)
    dq.save_report(quality_report, out / "data_quality")
    validation = validate_panel(panel, registry, config)
    validation.to_csv(out / "panel_validation.csv", index=False, encoding="utf-8-sig")
    write_markdown_summary(
        out,
        "Quant Model System Preflight",
        {
            "panel_validation": validation,
            "data_quality_summary": quality_report["summary"],
            "data_quality_gates": quality_report["gates"],
            "registry_audit": audit["summary"],
        },
    )
    failed = []
    if (validation["status"] == "fail").any():
        failed.extend(validation.loc[validation["status"] == "fail", "gate"].tolist())
    gates = quality_report["gates"]
    if not gates.empty and (gates["status"] == "fail").any():
        failed.extend(gates.loc[gates["status"] == "fail", "gate"].tolist())
    if not audit["summary"].empty and (audit["summary"]["status"] == "fail").any():
        failed.extend(audit["summary"].loc[audit["summary"]["status"] == "fail", "gate"].tolist())
    if failed:
        raise ValueError(f"Preflight failed: {failed}. See {out}.")
    return out


def run_demo(root: Path, output_name: str) -> Path:
    """Run a complete synthetic low-cost factor system demo."""
    output_dir = root / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_raw = lowcost.make_synthetic_low_cost_panel()
    panel = lowcost.build_low_cost_factors(panel_raw)
    panel = lowcost.add_forward_return(panel)
    panel_path = output_dir / "panel_with_factors.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")

    registry_path = root / "data_catalog" / "a_share_factor_registry_v0.csv"
    registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    audit = registry_audit.audit_registry(registry)
    for name, table in audit.items():
        table.to_csv(output_dir / f"registry_{name}.csv", index=False, encoding="utf-8-sig")

    config_path = root / "configs" / "factor_factory_smoke.json"
    config = wf.load_config(config_path)
    quality_report = write_data_quality(output_dir, panel, registry, config)
    validation = validate_panel(panel, registry, config)
    validation.to_csv(output_dir / "panel_validation.csv", index=False, encoding="utf-8-sig")

    wf_dir = output_dir / "walk_forward"
    results = wf.run_walk_forward(panel, registry, config)
    wf.save_results(results, wf_dir)
    ledger_path = root / "logs" / "factor_factory_experiment_ledger.csv"
    ledger_table = ledger.append_ledger(
        ledger_path,
        wf_dir,
        experiment_id=output_name,
        hypothesis="one-command quant model system demo should run low-cost factors through walk-forward governance",
    )
    ledger_tail = ledger_table.tail(1)
    ledger_tail.to_csv(output_dir / "ledger_row.csv", index=False, encoding="utf-8-sig")
    paper_state = paper.initialize_paper_state(wf_dir, output_dir / "paper_tracking", output_name, capital=1_000_000.0)
    paper_state_df = pd.DataFrame([paper_state])
    write_markdown_summary(
        output_dir,
        "Quant Model System Demo",
        {
            "panel_validation": validation,
            "data_quality_gates": quality_report["gates"],
            "registry_audit": audit["summary"],
            "walk_forward_meta": results["meta"],
            "walk_forward_performance": results["walk_forward_performance"],
            "ledger_decision": ledger_tail,
            "paper_state": paper_state_df,
        },
    )
    run_report.generate_model_run_report(output_dir)
    return output_dir


def run_walk_forward_from_files(
    root: Path,
    panel_csv: str,
    registry_csv: str,
    config_json: str,
    output_dir: str,
    experiment_id: str,
    hypothesis: str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = csv_io.read_csv_robust(panel_csv)
    registry = csv_io.read_csv_robust(registry_csv)
    config = wf.load_config(config_json)
    quality_report = write_data_quality(out, panel, registry, config)
    validation = validate_panel(panel, registry, config)
    validation.to_csv(out / "panel_validation.csv", index=False, encoding="utf-8-sig")
    if (validation["status"] == "fail").any():
        raise ValueError("Panel validation failed. See panel_validation.csv.")
    results = wf.run_walk_forward(panel, registry, config)
    wf.save_results(results, out / "walk_forward")
    ledger_table = ledger.append_ledger(root / "logs" / "factor_factory_experiment_ledger.csv", out / "walk_forward", experiment_id, hypothesis)
    ledger_table.tail(1).to_csv(out / "ledger_row.csv", index=False, encoding="utf-8-sig")
    paper_state = paper.initialize_paper_state(out / "walk_forward", out / "paper_tracking", experiment_id, capital=1_000_000.0)
    paper_state_df = pd.DataFrame([paper_state])
    write_markdown_summary(
        out,
        "Quant Model System Run",
        {
            "panel_validation": validation,
            "data_quality_gates": quality_report["gates"],
            "walk_forward_meta": results["meta"],
            "walk_forward_performance": results["walk_forward_performance"],
            "ledger_decision": ledger_table.tail(1),
            "paper_state": paper_state_df,
        },
    )
    run_report.generate_model_run_report(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--root", default=str(DEFAULT_ROOT))
    demo.add_argument("--output-name", default="quant_model_system_demo")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--panel-csv", required=True)
    preflight.add_argument("--registry-csv", required=True)
    preflight.add_argument("--config-json", required=True)
    preflight.add_argument("--output-dir", required=True)

    run = sub.add_parser("walk-forward")
    run.add_argument("--root", default=str(DEFAULT_ROOT))
    run.add_argument("--panel-csv", required=True)
    run.add_argument("--registry-csv", required=True)
    run.add_argument("--config-json", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--experiment-id", required=True)
    run.add_argument("--hypothesis", required=True)
    args = parser.parse_args()

    if args.command == "demo":
        out = run_demo(Path(args.root), args.output_name)
    elif args.command == "preflight":
        out = run_preflight_from_files(args.panel_csv, args.registry_csv, args.config_json, args.output_dir)
    elif args.command == "walk-forward":
        out = run_walk_forward_from_files(
            Path(args.root),
            args.panel_csv,
            args.registry_csv,
            args.config_json,
            args.output_dir,
            args.experiment_id,
            args.hypothesis,
        )
    else:
        raise ValueError(args.command)
    print(f"saved={out.resolve()}")


if __name__ == "__main__":
    main()
