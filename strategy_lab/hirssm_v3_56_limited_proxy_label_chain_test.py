#!/usr/bin/env python
"""Run HIRSSM V3.56 limited proxy label-chain test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from limited_proxy_label_chain_test import (
    LimitedProxyChainConfig,
    build_acceptance_checks,
    build_catalog,
    build_chain_readiness,
    build_import_config,
    build_limited_signal_panel,
    build_no_promotion_guard,
    build_report,
    coverage_summary,
    source_summary,
)
from market_total_return_label_importer import build_forward_labels, validate_market_source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "limited_proxy_label_chain_test_v3_56.json"
TASK_ID = "20260529_v3_56_limited_proxy_label_chain_test"
VERSION = "V3.56"


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


def load_config(raw: dict[str, Any]) -> LimitedProxyChainConfig:
    return LimitedProxyChainConfig(
        proxy_source_path=resolve_path(raw["proxy_source_path"]),
        signal_panel_path=resolve_path(raw["signal_panel_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        horizons=tuple(int(item) for item in raw["horizons"]),
        min_source_rows=int(raw["min_source_rows"]),
        min_signal_rows=int(raw["min_signal_rows"]),
        min_label_rows=int(raw["min_label_rows"]),
        min_limited_coverage_ratio=float(raw["min_limited_coverage_ratio"]),
        market_proxy_codes=tuple(str(item) for item in raw["market_proxy_codes"]),
        source_asset_priority=tuple(str(item) for item in raw["source_asset_priority"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "limited_proxy_label_chain_test.py",
        ROOT / "strategy_lab" / "hirssm_v3_56_limited_proxy_label_chain_test.py",
        ROOT / "configs" / "limited_proxy_label_chain_test_v3_56.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_56_limited_proxy_label_chain_test.json",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(read_json(resolve_path(args.config)))
    output_dir = config.output_dir

    source = pd.read_csv(config.proxy_source_path, encoding="utf-8-sig", low_memory=False)
    signal_panel = pd.read_csv(config.signal_panel_path, encoding="utf-8-sig", low_memory=False)
    import_config = build_import_config(config)
    source_checks = validate_market_source(source, import_config)
    limited_signal_panel = build_limited_signal_panel(signal_panel, source, config.horizons)
    labels, importer_coverage = build_forward_labels(limited_signal_panel, source, import_config)
    coverage = coverage_summary(labels, limited_signal_panel, config.horizons)
    readiness = build_chain_readiness(source_checks, limited_signal_panel, labels, coverage, config)
    guard = build_no_promotion_guard(labels)
    acceptance = build_acceptance_checks(readiness, guard, labels, source)
    summary = source_summary(source)
    report = build_report(summary, source_checks, limited_signal_panel, labels, coverage, readiness, acceptance)
    catalog = build_catalog(readiness, labels)

    output_paths = {
        "limited_proxy_source": output_dir / "limited_proxy_source.csv",
        "source_summary": output_dir / "limited_proxy_source_summary.csv",
        "source_checks": output_dir / "limited_proxy_source_validation_checks.csv",
        "limited_signal_panel": output_dir / "limited_signal_panel.csv",
        "limited_labels": output_dir / "limited_market_forward_labels.csv",
        "importer_coverage": output_dir / "importer_label_coverage.csv",
        "coverage": output_dir / "limited_label_coverage.csv",
        "readiness": output_dir / "chain_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "limited_proxy_label_chain_report.md",
        "catalog": config.catalog_path,
        "changed_files": output_dir / "changed_files.txt",
        "self_check": output_dir / "self_check.csv",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(source, output_paths["limited_proxy_source"])
    write_csv(summary, output_paths["source_summary"])
    write_csv(source_checks, output_paths["source_checks"])
    write_csv(limited_signal_panel, output_paths["limited_signal_panel"])
    write_csv(labels, output_paths["limited_labels"])
    write_csv(importer_coverage, output_paths["importer_coverage"])
    write_csv(coverage, output_paths["coverage"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    official_source = ROOT / "data_raw" / "market_labels" / "market_total_return_index.csv"
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "limited_labels_nonempty",
                "status": "pass" if len(labels) > 0 else "fail",
                "detail": f"label_rows={len(labels)}",
            },
            {
                "check": "official_source_not_created",
                "status": "pass" if not official_source.exists() else "fail",
                "detail": rel(official_source),
            },
            {
                "check": "performance_validation_blocked",
                "status": "pass"
                if str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0]) == "blocked"
                else "fail",
                "detail": "limited chain test only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["limited_proxy_source"],
        output_paths["source_summary"],
        output_paths["source_checks"],
        output_paths["limited_signal_panel"],
        output_paths["limited_labels"],
        output_paths["importer_coverage"],
        output_paths["coverage"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    signal_dates = limited_signal_panel["signal_date"].astype(str) if not limited_signal_panel.empty else pd.Series(dtype=str)
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "generated_at": now_text(),
        "limited_chain_test_ready": True,
        "proxy_source_rows": int(len(source)),
        "limited_signal_rows": int(len(limited_signal_panel)),
        "limited_unique_signal_dates": int(signal_dates.nunique()),
        "limited_signal_date_min": str(signal_dates.min()) if not signal_dates.empty else "",
        "limited_signal_date_max": str(signal_dates.max()) if not signal_dates.empty else "",
        "label_rows": int(len(labels)),
        "labels_produced": bool(len(labels) > 0),
        "official_market_source_written": bool(official_source.exists()),
        "performance_validation_status": "blocked",
        "acceptance_pass": bool(acceptance["status"].eq("pass").all()),
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "data_decision": "limited_proxy_label_chain_verified" if len(labels) > 0 else "limited_proxy_label_chain_blocked",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"]]],
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["acceptance_pass"] and manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
