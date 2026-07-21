#!/usr/bin/env python
"""Run HIRSSM V3.54 MARKET total-return source acquisition router."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_total_return_source_acquirer import (
    SourceAcquirerConfig,
    build_acceptance_checks,
    build_catalog,
    build_no_source_guard,
    build_report,
    build_source_readiness,
    discover_total_return_index_candidates,
    execute_csindex_total_return_fetch,
    load_index_list,
    manual_instructions,
    provider_readiness,
    route_table,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "market_total_return_source_acquirer_v3_54.json"
TASK_ID = "20260529_v3_54_market_total_return_source_acquirer"
VERSION = "V3.54"


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


def load_config(raw: dict[str, Any]) -> SourceAcquirerConfig:
    return SourceAcquirerConfig(
        index_list_path=resolve_path(raw["index_list_path"]),
        target_source_path=resolve_path(raw["target_source_path"]),
        output_dir=resolve_path(raw["output_dir"]),
        catalog_path=resolve_path(raw["catalog_path"]),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        preferred_market_asset=str(raw["preferred_market_asset"]),
        execute=bool(raw["execute"]),
        allow_adjusted_proxy=bool(raw["allow_adjusted_proxy"]),
        allow_overwrite_target=bool(raw["allow_overwrite_target"]),
        route_priority=tuple(str(item) for item in raw["route_priority"]),
    )


def build_changed_files(outputs: list[Path]) -> str:
    static_files = [
        ROOT / "strategy_lab" / "market_total_return_source_acquirer.py",
        ROOT / "strategy_lab" / "hirssm_v3_54_market_total_return_source_acquirer.py",
        ROOT / "configs" / "market_total_return_source_acquirer_v3_54.json",
        ROOT / "strategy_lab" / "agents" / "task_briefs" / "20260529_v3_54_market_total_return_source_acquirer.json",
    ]
    return "\n".join(rel(path) for path in static_files + outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    raw = read_json(config_path)
    if args.execute:
        raw["execute"] = True
    config = load_config(raw)
    output_dir = config.output_dir

    providers = provider_readiness(ROOT)
    index_list, index_error = load_index_list(config.index_list_path)
    candidates = discover_total_return_index_candidates(index_list)
    target_exists_before = config.target_source_path.exists()
    routes = route_table(providers, candidates, config, target_exists_before)

    acquisition_rows: list[dict[str, Any]] = []
    if bool(routes.loc[routes["route_id"] == "csindex_total_return_code", "will_execute"].any()):
        try:
            acquired, data, detail = execute_csindex_total_return_fetch(candidates, config)
            acquisition_rows.append(
                {
                    "route_id": "csindex_total_return_code",
                    "status": "acquired" if acquired else "blocked",
                    "rows": int(len(data)),
                    "detail": detail,
                }
            )
        except Exception as exc:  # noqa: BLE001
            acquisition_rows.append(
                {
                    "route_id": "csindex_total_return_code",
                    "status": "error",
                    "rows": 0,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
    else:
        acquisition_rows.append(
            {
                "route_id": "all",
                "status": "not_executed",
                "rows": 0,
                "detail": "execute=false or no ready certified route",
            }
        )
    acquisition_log = pd.DataFrame(acquisition_rows)
    target_exists_after = config.target_source_path.exists()
    readiness = build_source_readiness(routes, acquisition_log, target_exists_after)
    guard = build_no_source_guard(target_exists_after)
    acceptance = build_acceptance_checks(routes, candidates, readiness, guard, config)
    report = build_report(providers, candidates, routes, acquisition_log, readiness, acceptance, config)
    catalog = build_catalog(readiness, routes, config.target_source_path)

    index_status = pd.DataFrame(
        [
            {
                "input": rel(config.index_list_path),
                "status": "pass" if not index_error else "blocked",
                "detail": index_error,
                "rows": int(len(index_list)),
            }
        ]
    )

    output_paths = {
        "provider_readiness": output_dir / "provider_readiness.csv",
        "index_list_status": output_dir / "index_list_status.csv",
        "candidate_discovery": output_dir / "total_return_candidate_discovery.csv",
        "routes": output_dir / "acquisition_routes.csv",
        "acquisition_log": output_dir / "acquisition_log.csv",
        "readiness": output_dir / "source_acquisition_readiness_checks.csv",
        "guard": output_dir / "no_source_guard.csv",
        "acceptance": output_dir / "acceptance_checks.csv",
        "manual": output_dir / "manual_total_return_source_instructions.md",
        "report": output_dir / "market_total_return_source_acquirer_report.md",
        "catalog": config.catalog_path,
        "changed_files": output_dir / "changed_files.txt",
        "self_check": output_dir / "self_check.csv",
        "manifest": output_dir / "agent_run_manifest.json",
    }

    write_csv(providers, output_paths["provider_readiness"])
    write_csv(index_status, output_paths["index_list_status"])
    write_csv(candidates, output_paths["candidate_discovery"])
    write_csv(routes, output_paths["routes"])
    write_csv(acquisition_log, output_paths["acquisition_log"])
    write_csv(readiness, output_paths["readiness"])
    write_csv(guard, output_paths["guard"])
    write_csv(acceptance, output_paths["acceptance"])
    write_text(manual_instructions(config), output_paths["manual"])
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
                "check": "no_labels_or_backtest_outputs",
                "status": "pass"
                if not bool(guard.loc[guard["result_type"].isin(["market_forward_labels", "state_stratified_validation_or_backtest"]), "produced"].any())
                else "fail",
                "detail": "source acquisition only",
            },
            {
                "check": "route_table_nonempty",
                "status": "pass" if len(routes) > 0 else "fail",
                "detail": f"routes={len(routes)}",
            },
            {
                "check": "manual_instructions_written",
                "status": "pass" if output_paths["manual"].exists() else "fail",
                "detail": rel(output_paths["manual"]),
            },
        ]
    )
    write_csv(self_check, output_paths["self_check"])

    outputs_for_changed = [
        output_paths["provider_readiness"],
        output_paths["index_list_status"],
        output_paths["candidate_discovery"],
        output_paths["routes"],
        output_paths["acquisition_log"],
        output_paths["readiness"],
        output_paths["guard"],
        output_paths["acceptance"],
        output_paths["manual"],
        output_paths["report"],
        output_paths["catalog"],
        output_paths["self_check"],
    ]
    write_text(build_changed_files(outputs_for_changed), output_paths["changed_files"])

    ready_routes = routes.loc[routes["ready"].astype(bool), "route_id"].astype(str).tolist()
    acquired = bool((acquisition_log["status"] == "acquired").any())
    manifest = {
        "task_id": TASK_ID,
        "version": VERSION,
        "generated_at": now_text(),
        "source_acquirer_ready": True,
        "execute": config.execute,
        "target_source_path": rel(config.target_source_path),
        "target_exists_before": target_exists_before,
        "target_exists_after": target_exists_after,
        "ready_routes": ready_routes,
        "csindex_total_return_candidates": int(len(candidates)),
        "csindex_equity_total_return_candidates": int((candidates["acceptance_status"] == "candidate").sum()) if not candidates.empty else 0,
        "source_acquired": acquired,
        "labels_produced": False,
        "performance_validation_status": "blocked",
        "acceptance_pass": bool(acceptance["status"].eq("pass").all()),
        "self_check_pass": bool(self_check["status"].eq("pass").all()),
        "data_decision": "market_total_return_source_ready" if target_exists_after else "market_total_return_source_routes_blocked",
        "model_decision": "no_model_promotion_no_performance_claims",
        "outputs": [rel(path) for path in outputs_for_changed + [output_paths["changed_files"]]],
    }
    write_json(manifest, output_paths["manifest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["acceptance_pass"] and manifest["self_check_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
