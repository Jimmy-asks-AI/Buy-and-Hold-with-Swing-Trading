"""Build an event-sourced ETF lifecycle master for Long Hold V4 Gate E2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_master"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
SOURCE_URLS = {
    "joinquant_etf": "https://www.joinquant.com/help/api/doc?id=10029&name=JQDatadoc",
    "eastmoney_etf_spot": "https://quote.eastmoney.com/center/gridlist.html#fund_etf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _normalise_joinquant(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    required = {"code", "display_name", "start_date", "end_date", "type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"joinquant_etf missing columns: {missing}")

    code = frame["code"].astype(str).str.strip()
    out = pd.DataFrame(
        {
            "asset": code.str[:6],
            "asset_name": frame["display_name"].astype(str).str.strip(),
            "list_date": pd.to_datetime(frame["start_date"], errors="coerce"),
            "provider_end_date": pd.to_datetime(frame["end_date"], errors="coerce"),
            "exchange": code.str[-4:].map({"XSHG": "SSE", "XSHE": "SZSE"}),
            "fund_type": frame["type"].astype(str).str.lower().str.strip(),
        }
    )
    valid = out["asset"].str.fullmatch(r"\d{6}") & out["exchange"].notna() & out["fund_type"].eq("etf")
    out = out[valid & out["list_date"].notna() & out["provider_end_date"].notna()].copy()
    out = out[out["list_date"] <= as_of].copy()
    if out.empty:
        raise ValueError("joinquant_etf contains no as-of eligible ETF records")
    if out.duplicated("asset").any():
        duplicates = sorted(out.loc[out.duplicated("asset", keep=False), "asset"].unique())[:10]
        raise ValueError(f"joinquant_etf contains duplicate ETF codes: {duplicates}")
    if out["provider_end_date"].le(out["list_date"]).any():
        raise ValueError("joinquant_etf contains end dates on or before list dates")
    return out.sort_values("asset").reset_index(drop=True)


def build_etf_security_master(
    joinquant_frame: pd.DataFrame,
    source_vintage: str,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    """Convert all-status ETF records into as-of-safe listing and delisting events."""

    as_of_date = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of_date):
        raise ValueError("as_of must be a valid date")
    entities = _normalise_joinquant(joinquant_frame, as_of_date)
    rows: list[dict[str, Any]] = []
    for row in entities.itertuples(index=False):
        common = {
            "asset": row.asset,
            "asset_name": row.asset_name,
            "list_date": row.list_date,
            "exchange": row.exchange,
            "fund_type": row.fund_type,
            "data_source": "joinquant.get_all_securities(etf)",
            "source_vintage": source_vintage,
        }
        rows.append(
            {
                **common,
                "delist_date": pd.NaT,
                "list_status": "listed",
                "event_type": "listing",
                "available_date": row.list_date,
            }
        )
        if row.provider_end_date <= as_of_date:
            rows.append(
                {
                    **common,
                    "delist_date": row.provider_end_date,
                    "list_status": "delisted",
                    "event_type": "delisting",
                    "available_date": row.provider_end_date,
                }
            )

    columns = [
        "asset",
        "asset_name",
        "list_date",
        "delist_date",
        "list_status",
        "event_type",
        "exchange",
        "fund_type",
        "available_date",
        "data_source",
        "source_vintage",
    ]
    output = pd.DataFrame(rows, columns=columns).sort_values(["available_date", "asset", "event_type"]).reset_index(drop=True)
    if output.duplicated(["asset", "available_date", "list_status"]).any():
        raise ValueError("ETF lifecycle output contains duplicate event keys")
    listing = output["event_type"].eq("listing")
    delisting = output["event_type"].eq("delisting")
    if output.loc[listing, "delist_date"].notna().any():
        raise ValueError("listing events must not reveal a future delist date")
    if not (output.loc[delisting, "available_date"] == output.loc[delisting, "delist_date"]).all():
        raise ValueError("delisting events must become available on their event date")
    if (output["available_date"] > as_of_date).any():
        raise ValueError("ETF lifecycle output contains future events")
    return output


def _load_joinquant() -> pd.DataFrame:
    credential_path = ROOT / "configs" / "data_credentials.json"
    if not credential_path.is_file():
        raise ValueError("JoinQuant credentials are required for ETF lifecycle history")
    credentials = json.loads(credential_path.read_text(encoding="utf-8-sig")).get("joinquant", {})
    username = str(credentials.get("username", "")).strip()
    password = str(credentials.get("password", "")).strip()
    if not username or not password:
        raise ValueError("JoinQuant credentials are incomplete")

    import jqdatasdk as jq

    jq.auth(username, password)
    frame = jq.get_all_securities(types=["etf"])
    if frame.empty:
        raise ValueError("JoinQuant returned an empty ETF security master")
    return frame.rename_axis("code").reset_index()


def _load_eastmoney_current() -> pd.DataFrame:
    import akshare as ak

    frame = ak.fund_etf_spot_em()
    if frame.empty or "代码" not in frame.columns:
        raise ValueError("Eastmoney returned an empty or invalid current ETF list")
    return frame


def _snapshot(frame: pd.DataFrame, source_id: str, raw_dir: Path, sort_column: str) -> tuple[Path, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values(sort_column).reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    digest = hashlib.sha256(payload).hexdigest()
    path = raw_dir / f"{source_id}_{digest[:16]}.csv"
    if not path.exists():
        path.write_bytes(payload)
    return path, digest


def _current_reconciliation(master: pd.DataFrame, eastmoney: pd.DataFrame) -> dict[str, Any]:
    exits = set(master.loc[master["event_type"].eq("delisting"), "asset"])
    jq_active = set(master.loc[master["event_type"].eq("listing"), "asset"]).difference(exits)
    eastmoney_active = set(eastmoney["代码"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6))
    eastmoney_active = {code for code in eastmoney_active if len(code) == 6 and code.isdigit()}
    intersection = jq_active.intersection(eastmoney_active)
    if not jq_active or not eastmoney_active:
        raise ValueError("ETF current-list reconciliation has an empty population")
    jq_overlap = len(intersection) / len(jq_active)
    eastmoney_overlap = len(intersection) / len(eastmoney_active)
    if min(jq_overlap, eastmoney_overlap) < 0.90:
        raise ValueError(f"ETF current-list reconciliation is too weak: jq={jq_overlap:.3f};eastmoney={eastmoney_overlap:.3f}")
    return {
        "joinquant_active": len(jq_active),
        "eastmoney_active": len(eastmoney_active),
        "intersection": len(intersection),
        "joinquant_overlap": round(jq_overlap, 6),
        "eastmoney_overlap": round(eastmoney_overlap, 6),
        "joinquant_only": sorted(jq_active.difference(eastmoney_active)),
        "eastmoney_only": sorted(eastmoney_active.difference(jq_active)),
    }


def run_builder(as_of: str, raw_dir: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    jq_raw = _load_joinquant()
    eastmoney_raw = _load_eastmoney_current()
    jq_path, jq_hash = _snapshot(jq_raw, "joinquant_etf", raw_dir, "code")
    eastmoney_path, eastmoney_hash = _snapshot(eastmoney_raw, "eastmoney_etf_spot", raw_dir, "代码")
    combined_hash = hashlib.sha256(f"{jq_hash}:{eastmoney_hash}".encode()).hexdigest()
    source_vintage = f"etf_master_bundle_sha256:{combined_hash}"
    output = build_etf_security_master(jq_raw, source_vintage, as_of_date)
    reconciliation = _current_reconciliation(output, eastmoney_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    code_path = Path(__file__).resolve()
    inputs = [
        {"source_id": "joinquant_etf", "source_url": SOURCE_URLS["joinquant_etf"], "path": _relative(jq_path), "sha256": jq_hash},
        {
            "source_id": "eastmoney_etf_spot",
            "source_url": SOURCE_URLS["eastmoney_etf_spot"],
            "path": _relative(eastmoney_path),
            "sha256": eastmoney_hash,
        },
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(as_of_date.date()),
        "inputs": inputs,
        "source_vintage": source_vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "listed_events": int(output["event_type"].eq("listing").sum()),
        "delisted_events": int(output["event_type"].eq("delisting").sum()),
        "coverage_start": str(output["list_date"].min().date()),
        "coverage_end": str(output["list_date"].max().date()),
        "current_reconciliation": reconciliation,
        "limitations": [
            "Lifecycle dates come from one all-status provider; Eastmoney verifies only the current population",
            "Names are latest provider labels and must not be used as historical signals",
            "Benchmark history is intentionally excluded and governed by a separate PIT dataset",
        ],
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_builder(args.as_of, args.raw_dir, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
