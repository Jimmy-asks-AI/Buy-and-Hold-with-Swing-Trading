#!/usr/bin/env python
"""Run HIRSSM V3.58 long MARKET price proxy source builder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from long_market_proxy_source import (
    CandidateIndexSource,
    LongMarketProxyConfig,
    audit_candidate_sources,
    build_acceptance_checks,
    build_catalog,
    build_label_contract_delta,
    build_no_promotion_guard,
    build_primary_proxy_source,
    build_report,
    build_source_contract_checks,
    build_write_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "long_market_proxy_source_v3_58.json"
TASK_ID = "20260529_v3_58_long_market_proxy_source"
VERSION = "V3.58"
AGENT = "data_steward"


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


def load_config(raw: dict[str, Any]) -> LongMarketProxyConfig:
    candidates = tuple(
        CandidateIndexSource(
            index_code=str(item["index_code"]),
            asset_or_index=str(item["asset_or_index"]),
            role=str(item["role"]),
            source_path=resolve_path(item["source_path"]),
        )
        for item in raw["candidate_sources"]
    )
    return LongMarketProxyConfig(
        candidate_sources=candidates,
        primary_index_code=str(raw["primary_index_code"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        raw_proxy_path=resolve_path(raw["raw_proxy_path"]),
        official_total_return_path=resolve_path(raw["official_total_return_path"]),
        as_of_date=str(raw["as_of_date"]),
        min_primary_rows=int(raw["minimums"]["primary_rows"]),
        min_regular_rows=int(raw["minimums"]["regular_rows"]),
        min_start_date=str(raw["minimums"]["start_date_lte"]),
        min_end_date=str(raw["minimums"]["end_date_gte"]),
        max_staleness_days=int(raw["minimums"]["max_staleness_days"]),
        max_return_diff_bps=float(raw["minimums"]["max_return_diff_bps"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "long_market_proxy_source.py",
        ROOT / "strategy_lab" / "hirssm_v3_58_long_market_proxy_source.py",
        ROOT / "configs" / "long_market_proxy_source_v3_58.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_58_long_market_proxy_source.json",
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

    audit = audit_candidate_sources(config)
    proxy = build_primary_proxy_source(config)
    checks = build_source_contract_checks(audit, proxy, config)
    write_policy = build_write_policy(config, checks)
    proxy_policy = write_policy.loc[write_policy["target_role"] == "long_history_market_price_index_proxy"].iloc[0]
    if bool(proxy_policy.write_allowed):
        write_csv(proxy, config.raw_proxy_path)
    label_delta = build_label_contract_delta()
    guard = build_no_promotion_guard()
    acceptance = build_acceptance_checks(checks, write_policy, guard, config)
    report = build_report(audit, proxy, checks, write_policy, acceptance, config)
    catalog = build_catalog(config, proxy)

    output_paths = {
        "audit": output_dir / "candidate_index_source_audit.csv",
        "proxy_copy": output_dir / "market_price_proxy_000985.csv",
        "checks": output_dir / "source_contract_checks.csv",
        "write_policy": output_dir / "write_policy.csv",
        "label_delta": output_dir / "label_contract_delta.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "report": output_dir / "long_market_proxy_source_report.md",
        "catalog": config.catalog_path,
        "self_check": output_dir / "self_check.csv",
        "changed_files": output_dir / "changed_files.txt",
        "manifest": output_dir / "agent_run_manifest.json",
    }
    write_csv(audit, output_paths["audit"])
    write_csv(proxy, output_paths["proxy_copy"])
    write_csv(checks, output_paths["checks"])
    write_csv(write_policy, output_paths["write_policy"])
    write_csv(label_delta, output_paths["label_delta"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    dates = proxy["date"].astype(str) if not proxy.empty else pd.Series(dtype=str)
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "long_proxy_file_exists",
                "status": "pass" if config.raw_proxy_path.exists() else "fail",
                "detail": rel(config.raw_proxy_path),
            },
            {
                "check": "official_total_return_source_not_created",
                "status": "pass" if not config.official_total_return_path.exists() else "fail",
                "detail": rel(config.official_total_return_path),
            },
            {
                "check": "performance_validation_blocked",
                "status": "pass",
                "detail": "V3.58 source construction only",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["audit"],
        output_paths["proxy_copy"],
        config.raw_proxy_path,
        output_paths["checks"],
        output_paths["write_policy"],
        output_paths["label_delta"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    metrics = {
        "primary_index_code": config.primary_index_code,
        "proxy_rows": int(len(proxy)),
        "proxy_date_min": str(dates.min()) if not dates.empty else "",
        "proxy_date_max": str(dates.max()) if not dates.empty else "",
        "candidate_count": int(len(audit)),
        "official_total_return_written": bool(config.official_total_return_path.exists()),
        "long_price_proxy_written": bool(config.raw_proxy_path.exists()),
        "performance_validation_status": "blocked",
    }
    manifest = {
        "run_id": TASK_ID,
        "task_id": TASK_ID,
        "agent": AGENT,
        "version": VERSION,
        "baseline": "V3.57 limited proxy state diagnostic",
        "status": "pass" if bool(self_check["status"].eq("pass").all()) else "fail",
        "started_at": started_at,
        "command": f"python -X utf8 strategy_lab/hirssm_v3_58_long_market_proxy_source.py --config {rel(config_path)}",
        "config": raw_config,
        "data_refs": [rel(candidate.source_path) for candidate in config.candidate_sources],
        "code_refs": [
            "strategy_lab/long_market_proxy_source.py",
            "strategy_lab/hirssm_v3_58_long_market_proxy_source.py",
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
        "warn_count": int((checks["status"] == "warn").sum()),
        "limitations": [
            "The source is a CSIndex price-index proxy and excludes dividend reinvestment.",
            "It is not an official total-return index and must not be written to market_total_return_index.csv.",
            "V3.58 creates no labels, IC, hit-rate, backtest, or model-promotion evidence.",
        ],
        "risk_flags": [
            "price_index_proxy_not_total_return",
            "dividend_exclusion_bias",
            "same_day_after_close_availability",
            "performance_validation_blocked",
        ],
        "next_decision": "Implement V3.59 guarded proxy-label importer only if price-index proxy evidence is acceptable as a separate non-official validation track.",
        "handoff_summary": "Local CSIndex 000985 was normalized into a governed long-history MARKET price proxy while the official total-return source remains untouched.",
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
