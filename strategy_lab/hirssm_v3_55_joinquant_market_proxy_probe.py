#!/usr/bin/env python
"""Run HIRSSM V3.55 JoinQuant MARKET proxy probe."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from joinquant_market_proxy_probe import (
    JoinQuantProbeConfig,
    build_acceptance_checks,
    build_catalog,
    build_no_promotion_guard,
    build_probe_plan,
    build_readiness_checks,
    build_report,
    credential_pair,
    credential_instructions,
    execute_joinquant_probe,
    maybe_write_proxy_source,
    readiness_frame,
    write_policy_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "joinquant_market_proxy_probe_v3_55.json"
TASK_ID = "20260529_v3_55_joinquant_market_proxy_probe"
VERSION = "V3.55"


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


def load_config(raw: dict[str, Any]) -> JoinQuantProbeConfig:
    return JoinQuantProbeConfig(
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        target_security=str(raw["target_security"]),
        target_asset_or_index=str(raw["target_asset_or_index"]),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        fields=tuple(str(item) for item in raw["fields"]),
        fq=str(raw["fq"]),
        execute_probe=bool(raw["execute_probe"]),
        allow_adjusted_proxy_write=bool(raw["allow_adjusted_proxy_write"]),
        approved_proxy_basis=str(raw["approved_proxy_basis"]),
        allow_overwrite_target=bool(raw["allow_overwrite_target"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "joinquant_market_proxy_probe.py",
        ROOT / "strategy_lab" / "hirssm_v3_55_joinquant_market_proxy_probe.py",
        ROOT / "configs" / "joinquant_market_proxy_probe_v3_55.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_55_joinquant_market_proxy_probe.json",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute-probe", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    raw = read_json(config_path)
    if args.execute_probe:
        raw["execute_probe"] = True
    config = load_config(raw)
    output_dir = config.output_dir

    readiness = readiness_frame(ROOT, config)
    probe_plan = build_probe_plan(config, readiness)
    try:
        probe_result, normalized = execute_joinquant_probe(ROOT, config, readiness)
    except Exception as exc:  # noqa: BLE001
        probe_result = pd.DataFrame(
            [
                {
                    "probe_id": "jq_get_price_market_proxy",
                    "status": "error",
                    "rows": 0,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ]
        )
        normalized = pd.DataFrame()
    write_policy = write_policy_frame(config, normalized)
    write_result = maybe_write_proxy_source(config, normalized, write_policy)
    checks = build_readiness_checks(readiness, probe_result, write_policy, write_result, config)
    guard = build_no_promotion_guard(write_result)
    acceptance = build_acceptance_checks(readiness, probe_plan, write_policy, guard)
    report = build_report(config, readiness, probe_plan, probe_result, write_policy, write_result, checks, acceptance)
    catalog = build_catalog(checks, write_result)

    output_paths = {
        "readiness": output_dir / "joinquant_readiness.csv",
        "probe_plan": output_dir / "probe_plan.csv",
        "probe_result": output_dir / "probe_result.csv",
        "normalized_series": output_dir / "normalized_proxy_series.csv",
        "normalized_preview": output_dir / "normalized_proxy_preview.csv",
        "write_policy": output_dir / "write_policy.csv",
        "write_result": output_dir / "write_result.csv",
        "checks": output_dir / "joinquant_probe_readiness_checks.csv",
        "guard": output_dir / "no_promotion_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "instructions": output_dir / "joinquant_credential_instructions.md",
        "report": output_dir / "joinquant_market_proxy_probe_report.md",
        "catalog": config.catalog_path,
        "changed_files": output_dir / "changed_files.txt",
        "self_check": output_dir / "self_check.csv",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(readiness, output_paths["readiness"])
    write_csv(probe_plan, output_paths["probe_plan"])
    write_csv(probe_result, output_paths["probe_result"])
    write_csv(normalized, output_paths["normalized_series"])
    write_csv(normalized.head(20), output_paths["normalized_preview"])
    write_csv(write_policy, output_paths["write_policy"])
    write_csv(write_result, output_paths["write_result"])
    write_csv(checks, output_paths["checks"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(credential_instructions(), output_paths["instructions"])
    write_text(report, output_paths["report"])
    write_text(catalog, output_paths["catalog"])

    source_written = bool((write_result["status"] == "written").any()) if not write_result.empty else False
    username, password = credential_pair(ROOT)
    combined_public_text = "\n".join(
        [
            readiness.to_csv(index=False),
            probe_plan.to_csv(index=False),
            probe_result.to_csv(index=False),
            write_policy.to_csv(index=False),
            write_result.to_csv(index=False),
            checks.to_csv(index=False),
            acceptance.to_csv(index=False),
            report,
            catalog,
        ]
    )
    leaked_secrets = [
        secret
        for secret in [username, password]
        if secret and secret in combined_public_text
    ]
    self_check = pd.DataFrame(
        [
            {
                "check": "acceptance_checks_pass",
                "status": "pass" if acceptance["status"].eq("pass").all() else "fail",
                "detail": ";".join(acceptance.loc[acceptance["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "credentials_not_written",
                "status": "pass" if not leaked_secrets else "fail",
                "detail": "checked actual configured JoinQuant secrets against public outputs",
            },
            {
                "check": "no_labels_or_backtest_outputs",
                "status": "pass"
                if not bool(guard.loc[guard["result_type"] == "labels_or_state_validation_or_backtest", "produced"].iloc[0])
                else "fail",
                "detail": "probe only",
            },
            {
                "check": "source_write_requires_policy",
                "status": "pass"
                if source_written
                == bool(write_policy.loc[write_policy["policy"] == "write_market_total_return_index_csv", "approved"].iloc[0] and len(normalized) > 0)
                else "fail",
                "detail": f"source_written={source_written}",
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["readiness"],
        output_paths["probe_plan"],
        output_paths["probe_result"],
        output_paths["normalized_series"],
        output_paths["normalized_preview"],
        output_paths["write_policy"],
        output_paths["write_result"],
        output_paths["checks"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["instructions"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    probe_ok = bool((probe_result["status"] == "ok").any()) if not probe_result.empty else False
    perf_status = str(checks.loc[checks["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "generated_at": now_text(),
        "joinquant_probe_harness_ready": True,
        "execute_probe": config.execute_probe,
        "target_security": config.target_security,
        "fq": config.fq,
        "probe_ok": probe_ok,
        "normalized_proxy_rows": int(len(normalized)),
        "source_written": source_written,
        "target_source_path": rel(config.target_source_path),
        "labels_produced": False,
        "performance_validation_status": perf_status,
        "acceptance_pass": bool(acceptance["status"].eq("pass").all()),
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "data_decision": "joinquant_proxy_source_written" if source_written else "joinquant_probe_ready_source_write_blocked",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"]]],
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["acceptance_pass"] and manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
