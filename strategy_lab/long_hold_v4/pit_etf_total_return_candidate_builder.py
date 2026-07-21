"""Build an ETF total-return candidate from official cash-event evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .etf_corporate_actions import (
    DEFAULT_REGISTRY_PATH,
    conversion_factors_for_asset,
    load_corporate_action_registry,
)
from .pit_etf_total_return_collector import (
    PRICE_COLUMNS,
    build_lifecycle_observation,
    load_lifecycles,
    load_terminal_cash_event_registry,
)


ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = ROOT / "data_raw" / "long_hold_v4" / "pit_history"
MASTER_PATH = PIT_ROOT / "etf_security_master.csv"
LIFECYCLE_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_lifecycle_observation_latest.json"
)
DIVIDEND_PATH = PIT_ROOT / "etf_dividend_events.csv"
DIVIDEND_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_dividend_events_builder_latest.json"
)
TERMINAL_PATH = PIT_ROOT / "etf_terminal_cash_events.csv"
TERMINAL_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "pit_etf_terminal_cash_events_builder_latest.json"
)
OBSERVATION_DIR = PIT_ROOT / "observations"
PRICE_OUTPUT_PATH = OBSERVATION_DIR / "etf_total_return_prices_official_event_candidate.csv.gz"
EVENT_OUTPUT_PATH = OBSERVATION_DIR / "etf_total_return_official_event_usage_candidate.csv"
STATUS_OUTPUT_PATH = OBSERVATION_DIR / "etf_total_return_official_event_candidate_status.csv"
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_total_return_official_event_candidate_latest.json"
)
QUALIFICATION_STATUS = "CANDIDATE_OFFICIAL_EVENTS_CURRENT_FINAL_PRICE"
EVENT_USAGE_COLUMNS = [
    "event_id",
    "asset",
    "event_type",
    "distribution_sequence",
    "holder_scope",
    "announcement_date",
    "available_trade_date",
    "available_date",
    "entitlement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "accounting_date",
    "cash_per_share",
    "termination_date",
    "is_final_distribution",
    "additional_distribution_expected",
    "extinguishes_position",
    "event_effective_by_cutoff",
    "applied_to_price_adjustment",
    "applied_to_cash_ledger",
    "data_source",
    "source_pdf_sha256_set",
    "source_vintage",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0} if gzip else None,
    )
    temporary.replace(path)


def _authenticate_file(item: dict[str, Any], role: str) -> Path:
    path = _resolve(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError(f"ETF total-return candidate input failed authentication: {role}")
    return path


def _load_raw_source_declarations() -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    manifest = json.loads(LIFECYCLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "COLLECTION_IN_PROGRESS_CURRENT_FINAL_SNAPSHOT"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
        or int(manifest.get("quarantined_assets", -1)) != 0
    ):
        raise ValueError("ETF lifecycle observation is not a complete governed raw-price source")
    declarations: dict[str, dict[str, dict[str, Any]]] = {"price": {}, "metadata": {}}
    role_map = {"etf_raw_price": "price", "etf_raw_metadata": "metadata"}
    for item in manifest.get("inputs", []):
        kind = role_map.get(str(item.get("role")))
        if kind is None or not item.get("asset") or not item.get("sha256"):
            continue
        asset = str(item["asset"]).zfill(6)
        if asset in declarations[kind]:
            raise ValueError(f"ETF lifecycle source has duplicate {kind} declarations: {asset}")
        declarations[kind][asset] = item
    expected = int(manifest.get("selected_assets", -1))
    if any(len(items) != expected for items in declarations.values()):
        raise ValueError("ETF lifecycle raw-price or metadata declarations are incomplete")
    return manifest, declarations


def load_official_dividend_events(
    event_path: Path = DIVIDEND_PATH,
    manifest_path: Path = DIVIDEND_MANIFEST_PATH,
    as_of: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "PROMOTED_FULL_UNIVERSE_OFFICIAL_EVENTS"
        or manifest.get("historical_backtest_allowed") is not True
        or manifest.get("model_promotion_allowed") is not False
        or manifest.get("current_final_snapshot") is not False
    ):
        raise ValueError("ETF official dividend manifest is not formally promoted")
    code_path = _resolve(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("ETF official dividend promoter code hash mismatch")
    outputs = [item for item in manifest.get("outputs", []) if item.get("role") == "pit_etf_dividend_events"]
    if len(outputs) != 1 or _authenticate_file(outputs[0], "pit_etf_dividend_events") != event_path.resolve():
        raise ValueError("ETF official dividend output declaration is invalid")
    events = pd.read_csv(event_path, dtype={"asset": str}, low_memory=False)
    required = {
        "asset",
        "announcement_date",
        "record_date",
        "ex_date",
        "pay_date",
        "cash_per_share",
        "available_date",
        "data_source",
        "source_vintage",
    }
    missing = sorted(required.difference(events.columns))
    if missing or events.empty:
        raise ValueError(f"ETF official dividend table is incomplete: {missing}")
    events = events.copy()
    events["asset"] = events["asset"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    dates = ["announcement_date", "record_date", "ex_date", "pay_date", "available_date"]
    for column in dates:
        events[column] = pd.to_datetime(events[column], errors="coerce").dt.normalize()
    events["cash_per_share"] = pd.to_numeric(events["cash_per_share"], errors="coerce")
    if events[["asset", *dates, "cash_per_share", "source_vintage"]].isna().any(axis=None):
        raise ValueError("ETF official dividend table contains missing formal values")
    if events["cash_per_share"].le(0).any() or not events["available_date"].eq(events["announcement_date"]).all():
        raise ValueError("ETF official dividend table contains invalid cash or availability")
    chronology = (
        events["announcement_date"].lt(events["record_date"])
        & events["record_date"].le(events["ex_date"])
        & events["ex_date"].le(events["pay_date"])
    )
    if not chronology.all() or events.duplicated(["asset", "ex_date"]).any():
        raise ValueError("ETF official dividend table contains invalid chronology or duplicate ex-dates")
    if len(events) != int(manifest.get("rows", -1)) or events["asset"].nunique() != int(manifest.get("assets", -1)):
        raise ValueError("ETF official dividend population does not match its manifest")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        events = events[events["available_date"].le(cutoff)].copy()
    metadata = {
        "table_sha256": _sha256(event_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_vintage_set_sha256": str(manifest.get("source_vintage_set_sha256", "")),
    }
    return events.sort_values(["asset", "ex_date"]).reset_index(drop=True), metadata


def build_event_usage(
    ordinary: pd.DataFrame,
    terminal: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    regular = ordinary.copy()
    regular["event_id"] = regular.apply(
        lambda row: "ordinary:"
        + hashlib.sha256(
            (
                f"{str(row['asset']).zfill(6)}|{pd.Timestamp(row['ex_date']).date().isoformat()}|"
                f"{float(row['cash_per_share']):.12g}|{row['source_vintage']}"
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    regular["event_type"] = "cash_distribution"
    regular["distribution_sequence"] = pd.NA
    regular["holder_scope"] = "record_date_holders"
    regular["available_trade_date"] = regular["available_date"]
    regular["entitlement_date"] = regular["record_date"]
    regular["accounting_date"] = regular["ex_date"]
    regular["termination_date"] = pd.NaT
    regular["is_final_distribution"] = False
    regular["additional_distribution_expected"] = False
    regular["extinguishes_position"] = False
    regular["event_effective_by_cutoff"] = regular["ex_date"].le(cutoff)
    regular["applied_to_price_adjustment"] = regular["event_effective_by_cutoff"]
    regular["applied_to_cash_ledger"] = regular["event_effective_by_cutoff"]
    regular["historical_backtest_allowed"] = True
    regular["model_promotion_allowed"] = False
    regular["source_pdf_sha256_set"] = ""

    terminal_usage = terminal.copy()
    if not terminal_usage.empty:
        is_v2 = "event_id" in terminal_usage.columns and "accounting_date" in terminal_usage.columns
        if not is_v2:
            terminal_usage["event_id"] = terminal_usage.apply(
                lambda row: "legacy-terminal:"
                + hashlib.sha256(
                    (
                        f"{str(row['asset']).zfill(6)}|{pd.Timestamp(row['ex_date']).date().isoformat()}|"
                        f"{float(row['cash_per_share']):.12g}|{row['source_vintage']}"
                    ).encode("utf-8")
                ).hexdigest(),
                axis=1,
            )
            terminal_usage["distribution_sequence"] = 1
            terminal_usage["holder_scope"] = "all_registered_holders"
            terminal_usage["available_trade_date"] = terminal_usage["available_date"]
            terminal_usage["entitlement_date"] = terminal_usage["record_date"]
            terminal_usage["accounting_date"] = terminal_usage["ex_date"]
            terminal_usage["is_final_distribution"] = terminal_usage["extinguishes_position"]
            terminal_usage["additional_distribution_expected"] = False
            terminal_usage["source_pdf_sha256_set"] = ""
        terminal_usage["event_effective_by_cutoff"] = (
            pd.to_datetime(terminal_usage["available_trade_date"], errors="coerce").dt.normalize().le(cutoff)
            & pd.to_datetime(terminal_usage["accounting_date"], errors="coerce").dt.normalize().le(cutoff)
        )
        terminal_usage["applied_to_price_adjustment"] = False
        terminal_usage["applied_to_cash_ledger"] = terminal_usage["event_effective_by_cutoff"]
        terminal_usage["data_source"] = "Official exchange liquidation distribution evidence"
    combined = pd.concat([regular, terminal_usage], ignore_index=True, sort=False)
    if combined["event_id"].astype(str).duplicated().any():
        raise ValueError("ETF official-event usage contains duplicate event IDs")
    return combined[EVENT_USAGE_COLUMNS].sort_values(
        ["asset", "accounting_date", "event_type", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _candidate_source_vintage(
    asset: str,
    price_sha256: str,
    metadata_sha256: str,
    dividend_sha256: str,
    action_sha256: str,
    terminal_bundle_sha256: str,
) -> str:
    material = {
        "asset": asset,
        "raw_price_sha256": price_sha256,
        "raw_metadata_sha256": metadata_sha256,
        "official_dividend_table_sha256": dividend_sha256,
        "corporate_action_registry_sha256": action_sha256,
        "terminal_event_bundle_sha256": terminal_bundle_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"etf_official_event_candidate_bundle_sha256:{digest}"


def build(as_of: str | pd.Timestamp) -> dict[str, Any]:
    cutoff = pd.Timestamp(as_of).normalize()
    created_at = datetime.now().astimezone()
    lifecycle_manifest, declarations = _load_raw_source_declarations()
    lifecycles = load_lifecycles(MASTER_PATH, cutoff)
    ordinary, dividend_metadata = load_official_dividend_events(as_of=cutoff)
    terminal, terminal_metadata = load_terminal_cash_event_registry(
        TERMINAL_PATH,
        TERMINAL_MANIFEST_PATH,
        cutoff,
    )
    actions = load_corporate_action_registry(DEFAULT_REGISTRY_PATH)
    action_sha256 = _sha256(DEFAULT_REGISTRY_PATH)
    effective_ordinary = ordinary[ordinary["ex_date"].le(cutoff)].copy()
    event_usage = build_event_usage(ordinary, terminal, cutoff)

    price_frames: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    authenticated_inputs: list[dict[str, Any]] = []
    for lifecycle in lifecycles.itertuples(index=False):
        asset = str(lifecycle.asset).zfill(6)
        status: dict[str, Any] = {
            "asset": asset,
            "asset_name": str(lifecycle.asset_name),
            "lifecycle_status": str(lifecycle.lifecycle_status),
            "build_status": "quarantined",
            "error": "",
            "price_rows": 0,
            "formal_dividend_events_available": int(ordinary["asset"].eq(asset).sum()),
            "formal_dividend_events_effective": int(effective_ordinary["asset"].eq(asset).sum()),
            "scheduled_future_dividend_events": int(
                (ordinary["asset"].eq(asset) & ordinary["ex_date"].gt(cutoff)).sum()
            ),
            "terminal_cash_events": int(terminal["asset"].eq(asset).sum()),
            "qualification_status": QUALIFICATION_STATUS,
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
        try:
            price_item = declarations["price"][asset]
            metadata_item = declarations["metadata"][asset]
            price_path = _authenticate_file(price_item, f"raw_price:{asset}")
            metadata_path = _authenticate_file(metadata_item, f"raw_metadata:{asset}")
            raw_prices = pd.read_csv(price_path, compression="gzip", low_memory=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            observed = metadata.get("fetched_at")
            cash_events = effective_ordinary[effective_ordinary["asset"].eq(asset)][
                ["ex_date", "cash_per_share"]
            ].rename(columns={"ex_date": "date", "cash_per_share": "cash"})
            terminal_matches = terminal[terminal["asset"].eq(asset)]
            if not terminal_matches.empty and "event_id" not in terminal_matches.columns:
                cash_events = pd.concat(
                    [
                        cash_events,
                        terminal_matches[["ex_date", "cash_per_share"]].rename(
                            columns={"ex_date": "date", "cash_per_share": "cash"}
                        ),
                    ],
                    ignore_index=True,
                )
            cash_events = cash_events.sort_values("date").reset_index(drop=True)
            synthetic_dividends = pd.DataFrame(columns=["date", "cumulative_dividend"])
            if not cash_events.empty:
                cash_events["cumulative_dividend"] = pd.to_numeric(cash_events["cash"], errors="raise").cumsum()
                synthetic_dividends = cash_events[["date", "cumulative_dividend"]]
            source_vintage = _candidate_source_vintage(
                asset,
                str(price_item["sha256"]),
                str(metadata_item["sha256"]),
                dividend_metadata["table_sha256"],
                action_sha256,
                terminal_metadata["bundle_sha256"],
            )
            factors = conversion_factors_for_asset(actions, asset, cutoff)
            prices, _, diagnostics = build_lifecycle_observation(
                raw_prices,
                synthetic_dividends,
                lifecycle,
                cutoff,
                observed,
                source_vintage,
                factors,
                terminal_matches if not terminal_matches.empty else None,
            )
            prices["data_source"] = (
                "Sina raw ETF OHLC current-final snapshot; official cash events; governed share actions"
            )
            prices["source_vintage"] = source_vintage
            prices["qualification_status"] = QUALIFICATION_STATUS
            prices["historical_backtest_allowed"] = False
            prices["model_promotion_allowed"] = False
            price_frames.append(prices[PRICE_COLUMNS])
            status.update(
                {
                    "build_status": "ready_candidate",
                    "price_rows": int(len(prices)),
                    **diagnostics,
                }
            )
            authenticated_inputs.extend(
                [
                    {"role": "raw_price", "asset": asset, "path": _relative(price_path), "sha256": str(price_item["sha256"])},
                    {"role": "raw_metadata", "asset": asset, "path": _relative(metadata_path), "sha256": str(metadata_item["sha256"])},
                ]
            )
        except Exception as exc:  # noqa: BLE001 - candidate assets fail closed
            status["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        statuses.append(status)

    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame(columns=PRICE_COLUMNS)
    status_frame = pd.DataFrame(statuses)
    if prices.duplicated(["date", "asset"]).any():
        raise ValueError("ETF official-event candidate contains duplicate price keys")
    _atomic_csv(prices, PRICE_OUTPUT_PATH, gzip=True)
    _atomic_csv(event_usage, EVENT_OUTPUT_PATH)
    _atomic_csv(status_frame, STATUS_OUTPUT_PATH)
    successful = status_frame[status_frame["build_status"].eq("ready_candidate")]
    input_records = [
        {"role": "lifecycle_observation_manifest", "path": _relative(LIFECYCLE_MANIFEST_PATH), "sha256": _sha256(LIFECYCLE_MANIFEST_PATH)},
        {"role": "etf_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)},
        {"role": "official_dividend_events", "path": _relative(DIVIDEND_PATH), "sha256": dividend_metadata["table_sha256"]},
        {"role": "official_dividend_manifest", "path": _relative(DIVIDEND_MANIFEST_PATH), "sha256": dividend_metadata["manifest_sha256"]},
        {"role": "terminal_cash_events", "path": _relative(TERMINAL_PATH), "sha256": terminal_metadata["table_sha256"]},
        {"role": "terminal_cash_event_manifest", "path": _relative(TERMINAL_MANIFEST_PATH), "sha256": terminal_metadata["manifest_sha256"]},
        {"role": "corporate_action_registry", "path": _relative(DEFAULT_REGISTRY_PATH), "sha256": action_sha256},
        *authenticated_inputs,
    ]
    outputs = [
        {"role": "etf_total_return_price_candidate", "path": _relative(PRICE_OUTPUT_PATH), "sha256": _sha256(PRICE_OUTPUT_PATH), "rows": len(prices)},
        {"role": "official_event_usage", "path": _relative(EVENT_OUTPUT_PATH), "sha256": _sha256(EVENT_OUTPUT_PATH), "rows": len(event_usage)},
        {"role": "candidate_asset_status", "path": _relative(STATUS_OUTPUT_PATH), "sha256": _sha256(STATUS_OUTPUT_PATH), "rows": len(status_frame)},
    ]
    payload = {
        "schema_version": 2,
        "created_at": created_at.isoformat(timespec="seconds"),
        "as_of_date": cutoff.date().isoformat(),
        "qualification_status": QUALIFICATION_STATUS,
        "inputs": input_records,
        "outputs": outputs,
        "selected_assets": int(len(lifecycles)),
        "ready_candidate_assets": int(len(successful)),
        "quarantined_assets": int(status_frame["build_status"].eq("quarantined").sum()),
        "quarantined_asset_codes": status_frame.loc[
            status_frame["build_status"].eq("quarantined"), "asset"
        ].tolist(),
        "price_rows": int(len(prices)),
        "official_dividend_events_available": int(len(ordinary)),
        "official_dividend_events_effective": int(len(effective_ordinary)),
        "scheduled_future_dividend_events": int(ordinary["ex_date"].gt(cutoff).sum()),
        "terminal_cash_event_rows": int(len(terminal)),
        "terminal_cash_event_assets": int(terminal["asset"].nunique()),
        "complete_terminal_event_chain_assets": int(terminal_metadata["complete_event_chain_assets"]),
        "incomplete_terminal_event_chain_assets": int(
            terminal["asset"].nunique() - terminal_metadata["complete_event_chain_assets"]
        ),
        "quarantined_terminal_event_candidate_rows": int(
            terminal_metadata["quarantined_candidate_rows"]
        ),
        "event_usage_rows": int(len(event_usage)),
        "governed_corporate_action_rows": int(len(actions)),
        "inferred_corporate_action_rows": int(successful["inferred_corporate_actions"].sum()),
        "contains_synthetic_market_rows": False,
        "current_final_price_snapshot": True,
        "formal_event_evidence_complete_for_registered_events": bool(
            len(successful) == len(lifecycles)
            and successful["inferred_corporate_actions"].eq(0).all()
            and len(event_usage) == len(ordinary) + len(terminal)
        ),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "source_quality_gate_passed": False,
        "limitations": [
            "official cash and share-action evidence replaces provider event inference",
            "raw market prices remain one current-final Sina snapshot per asset",
            "the independent JoinQuant comparison covers only a recent trial window",
            "full delisted history and source-version depth remain insufficient",
            "terminal cash events live in the event/cash ledger and never create synthetic OHLC rows",
            "only formally validated terminal-event rows are included; incomplete chains remain disclosed",
        ],
        "code_files": [
            {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
            {
                "path": _relative(Path(__file__).with_name("pit_etf_total_return_collector.py")),
                "sha256": _sha256(Path(__file__).with_name("pit_etf_total_return_collector.py")),
            },
            {
                "path": _relative(Path(__file__).with_name("etf_snapshot_builder.py")),
                "sha256": _sha256(Path(__file__).with_name("etf_snapshot_builder.py")),
            },
            {
                "path": _relative(Path(__file__).with_name("etf_corporate_actions.py")),
                "sha256": _sha256(Path(__file__).with_name("etf_corporate_actions.py")),
            },
        ],
    }
    _atomic_json(payload, MANIFEST_PATH)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.as_of)
    keys = (
        "qualification_status",
        "selected_assets",
        "ready_candidate_assets",
        "quarantined_assets",
        "price_rows",
        "official_dividend_events_available",
        "official_dividend_events_effective",
        "scheduled_future_dividend_events",
        "terminal_cash_event_rows",
        "contains_synthetic_market_rows",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
