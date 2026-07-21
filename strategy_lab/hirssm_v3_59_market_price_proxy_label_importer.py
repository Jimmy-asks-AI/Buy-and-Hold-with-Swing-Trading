#!/usr/bin/env python
"""Run HIRSSM V3.59 guarded MARKET price-proxy label importer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_price_proxy_label_importer import (
    PriceProxyLabelConfig,
    build_acceptance_checks,
    build_catalog,
    build_label_contract,
    build_label_diagnostics,
    build_no_promotion_guard,
    build_price_proxy_forward_labels,
    build_readiness_checks,
    build_report,
    validate_proxy_source,
    validate_signal_panel,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_price_proxy_label_importer_v3_59.json"
TASK_ID = "20260529_v3_59_market_price_proxy_label_importer"
VERSION = "V3.59"
AGENT = "backtest_validation_auditor"


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(raw: dict[str, Any]) -> PriceProxyLabelConfig:
    return PriceProxyLabelConfig(
        proxy_source_path=resolve_path(raw["proxy_source_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        v3_58_manifest_path=resolve_path(raw["v3_58_manifest_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        canonical_label_path=resolve_path(raw["canonical_label_path"]),
        official_total_return_source_path=resolve_path(raw["official_total_return_source_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        source_symbol=str(raw["source_symbol"]),
        min_proxy_rows=int(raw["minimums"]["proxy_rows"]),
        min_signal_rows=int(raw["minimums"]["signal_rows"]),
        min_label_rows=int(raw["minimums"]["label_rows"]),
        min_source_window_coverage_ratio=float(raw["minimums"]["source_window_coverage_ratio"]),
        min_all_signal_coverage_ratio=float(raw["minimums"]["all_signal_coverage_ratio"]),
    )


def output_columns(frames: list[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    for frame in frames:
        columns.extend(str(col) for col in frame.columns)
    return columns


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_price_proxy_label_importer.py",
        ROOT / "strategy_lab" / "hirssm_v3_59_market_price_proxy_label_importer.py",
        ROOT / "configs" / "market_price_proxy_label_importer_v3_59.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_59_market_price_proxy_label_importer.json",
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

    proxy = pd.read_csv(config.proxy_source_path, encoding="utf-8-sig", low_memory=False, dtype={"index_code": str})
    signal_panel = pd.read_csv(config.signal_panel_path, encoding="utf-8-sig", low_memory=False)
    v3_58_manifest = read_json(config.v3_58_manifest_path)

    proxy_checks = validate_proxy_source(proxy, config)
    signal_checks = validate_signal_panel(signal_panel, config)
    labels, coverage = build_price_proxy_forward_labels(signal_panel, proxy, config)
    label_contract = build_label_contract(labels)
    diagnostics = build_label_diagnostics(labels, config)
    readiness = build_readiness_checks(proxy_checks, signal_checks, labels, coverage, v3_58_manifest, config)
    guard = build_no_promotion_guard(labels)

    if len(labels) > 0:
        write_csv(labels, config.canonical_label_path)

    output_frames = [proxy_checks, signal_checks, labels, coverage, label_contract, diagnostics, readiness, guard]
    acceptance = build_acceptance_checks(
        readiness,
        labels,
        guard,
        config.official_total_return_source_path.exists(),
        output_columns(output_frames),
    )
    report = build_report(proxy, signal_panel, labels, coverage, diagnostics, readiness, acceptance, config)
    catalog = build_catalog(config, labels)

    output_paths = {
        "proxy_checks": output_dir / "proxy_source_checks.csv",
        "signal_checks": output_dir / "signal_contract_checks.csv",
        "labels": output_dir / "market_price_proxy_forward_labels.csv",
        "coverage": output_dir / "label_coverage.csv",
        "contract": output_dir / "price_proxy_label_contract.csv",
        "diagnostics": output_dir / "label_distribution_sanity.csv",
        "readiness": output_dir / "price_proxy_label_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "market_price_proxy_label_importer_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(proxy_checks, output_paths["proxy_checks"])
    write_csv(signal_checks, output_paths["signal_checks"])
    write_csv(labels, output_paths["labels"])
    write_csv(coverage, output_paths["coverage"])
    write_csv(label_contract, output_paths["contract"])
    write_csv(diagnostics, output_paths["diagnostics"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "canonical_price_proxy_labels_exist",
                "status": "pass" if config.canonical_label_path.exists() else "fail",
                "detail": rel(config.canonical_label_path),
            },
            {
                "check": "official_total_return_source_not_created",
                "status": "pass" if not config.official_total_return_source_path.exists() else "fail",
                "detail": rel(config.official_total_return_source_path),
            },
            {
                "check": "performance_validation_blocked",
                "status": "pass"
                if str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
                else "fail",
                "detail": "V3.59 labels only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["proxy_checks"],
        output_paths["signal_checks"],
        output_paths["labels"],
        config.canonical_label_path,
        output_paths["coverage"],
        output_paths["contract"],
        output_paths["diagnostics"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    label_dates = labels["signal_date"].astype(str) if not labels.empty else pd.Series(dtype=str)
    min_source_window_coverage = float(coverage["coverage_source_window_signal_dates"].min()) if not coverage.empty else 0.0
    min_all_signal_coverage = float(coverage["coverage_all_signal_dates"].min()) if not coverage.empty else 0.0
    warn_count = int((readiness["status"] == "warn").sum())
    metrics = {
        "proxy_rows": int(len(proxy)),
        "signal_rows": int(len(signal_panel)),
        "label_rows": int(len(labels)),
        "label_date_min": str(label_dates.min()) if not label_dates.empty else "",
        "label_date_max": str(label_dates.max()) if not label_dates.empty else "",
        "min_source_window_coverage": min_source_window_coverage,
        "min_all_signal_coverage": min_all_signal_coverage,
        "return_basis": "price_index_return",
        "official_total_return_labels_written": False,
        "price_proxy_labels_written": bool(config.canonical_label_path.exists()),
        "performance_validation_status": "blocked",
    }
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.58 long MARKET price proxy source",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -X utf8 strategy_lab/hirssm_v3_59_market_price_proxy_label_importer.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [rel(config.proxy_source_path), rel(config.signal_panel_path), rel(config.v3_58_manifest_path)],
        "code_refs": [
            "strategy_lab/market_price_proxy_label_importer.py",
            "strategy_lab/hirssm_v3_59_market_price_proxy_label_importer.py",
        ],
        "output_dir": rel(output_dir),
        "allowed_inputs": raw_config.get("allowed_inputs", []),
        "artifacts": [rel(path) for path in outputs_for_changed],
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]],
        "changed_files": build_changed_files(outputs_for_changed + [output_paths["changed_files"], output_paths["manifest"]]).splitlines(),
        "metrics": metrics,
        "metrics_summary": metrics,
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "fail_count": int((self_check["status"] != "pass").sum()),
        "warn_count": warn_count,
        "limitations": [
            "Labels are based on a price-index proxy and exclude dividend reinvestment.",
            "Labels are not official total-return labels and cannot support dividend-inclusive performance claims.",
            "V3.59 creates no IC, hit-rate, state validation, backtest, or model-promotion evidence.",
        ],
        "risk_flags": [
            "price_index_proxy_labels_not_total_return",
            "pre_2005_signal_dates_unlabelled",
            "dividend_exclusion_bias",
            "performance_validation_blocked",
        ],
        "next_decision": "V3.60 may run guarded state-stratified proxy validation, but all results must remain non-official price-index proxy evidence.",
        "handoff_summary": "Long MARKET price-proxy forward labels were generated with explicit non-official governance and no performance validation.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
