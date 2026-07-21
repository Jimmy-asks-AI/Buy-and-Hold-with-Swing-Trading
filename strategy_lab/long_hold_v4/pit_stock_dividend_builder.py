"""Build conservative PIT stock cash-dividend events from implemented plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_dividends"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_dividend_events.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_dividend_builder_latest.json"
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
LINEAGE_PATH = ROOT / "configs" / "long_hold_v4_security_lineage.json"
SOURCE_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REQUIRED_RAW = {
    "SECURITY_CODE",
    "SECURITY_NAME_ABBR",
    "REPORT_DATE",
    "PLAN_NOTICE_DATE",
    "NOTICE_DATE",
    "EX_DIVIDEND_DATE",
    "PRETAX_BONUS_RMB",
    "ASSIGN_PROGRESS",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def report_periods(start_year: int, as_of: str | pd.Timestamp) -> list[str]:
    as_of_date = pd.Timestamp(as_of).normalize()
    periods = []
    for year in range(start_year, as_of_date.year + 1):
        for suffix in ("03-31", "06-30", "09-30", "12-31"):
            period = pd.Timestamp(f"{year}-{suffix}")
            if period <= as_of_date:
                periods.append(str(period.date()))
    return periods


def _lifecycle_table(master: pd.DataFrame) -> pd.DataFrame:
    listed = master[master["event_type"].eq("listing")][["asset", "asset_name", "list_date"]].copy()
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listed["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("security master contains duplicate lifecycle events")
    lifecycle = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    lifecycle["asset"] = lifecycle["asset"].astype(str).str.zfill(6)
    lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"], errors="coerce")
    lifecycle["delist_date"] = pd.to_datetime(lifecycle["delist_date"], errors="coerce")
    if lifecycle[["asset", "list_date"]].isna().any(axis=None):
        raise ValueError("security master contains incomplete lifecycle keys")
    return lifecycle


def reconcile_dividend_lifecycles(
    events: pd.DataFrame,
    master: pd.DataFrame,
    lineage: dict,
) -> tuple[pd.DataFrame, dict[str, int]]:
    lifecycle = _lifecycle_table(master)
    by_asset = lifecycle.set_index("asset")
    output = events.copy()
    output["security_code_resolution"] = "direct"
    reassigned = 0
    for item in lineage.get("code_migrations", []):
        predecessor = str(item["predecessor"]).zfill(6)
        successor = str(item["successor"]).zfill(6)
        if predecessor not in by_asset.index or successor not in by_asset.index:
            raise ValueError(f"security lineage is absent from master: {predecessor}->{successor}")
        predecessor_end = by_asset.loc[predecessor, "delist_date"]
        successor_start = by_asset.loc[successor, "list_date"]
        if pd.isna(predecessor_end) or pd.Timestamp(predecessor_end) >= pd.Timestamp(successor_start):
            raise ValueError(f"invalid security lineage interval: {predecessor}->{successor}")
        mask = (
            output["asset"].eq(successor)
            & output["ex_date"].le(pd.Timestamp(predecessor_end))
            & output["ex_date"].lt(pd.Timestamp(successor_start))
        )
        reassigned += int(mask.sum())
        output.loc[mask, "asset"] = predecessor
        output.loc[mask, "asset_name"] = str(by_asset.loc[predecessor, "asset_name"])
        output.loc[mask, "security_code_resolution"] = "reassigned_predecessor"

    before = len(output)
    output = output.merge(
        lifecycle[["asset", "list_date", "delist_date"]], on="asset", how="inner", validate="many_to_one"
    )
    within_lifecycle = output["ex_date"].ge(output["list_date"]) & (
        output["delist_date"].isna() | output["ex_date"].le(output["delist_date"])
    )
    output = output[within_lifecycle].drop(columns=["list_date", "delist_date"]).copy()
    return output, {"reassigned_events": reassigned, "dropped_outside_lifecycle": int(before - len(output))}


def _request_page(session: requests.Session, period: str, page: int, attempts: int = 3) -> dict:
    params = {
        "sortColumns": "PLAN_NOTICE_DATE",
        "sortTypes": "-1",
        "pageSize": "500",
        "pageNumber": str(page),
        "reportName": "RPT_SHAREBONUS_DET",
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
        "filter": f"(REPORT_DATE='{period}')",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(SOURCE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"failed dividend query period={period};page={page}") from last_error


def fetch_raw(periods: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    nonempty_periods: list[str] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for period in periods:
            first = _request_page(session, period, 1)
            result = first.get("result")
            if not result or not result.get("data"):
                continue
            pages = int(result.get("pages", 1))
            rows = list(result["data"])
            for page in range(2, pages + 1):
                page_result = _request_page(session, period, page).get("result") or {}
                rows.extend(page_result.get("data") or [])
                time.sleep(0.05)
            frame = pd.DataFrame(rows)
            frame["query_report_date"] = period
            frames.append(frame)
            nonempty_periods.append(period)
    if not frames:
        raise ValueError("dividend source returned no records")
    return pd.concat(frames, ignore_index=True), nonempty_periods


def build_stock_dividend_events(
    raw: pd.DataFrame,
    source_vintage: str,
    as_of: str | pd.Timestamp,
    master: pd.DataFrame | None = None,
    lineage: dict | None = None,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_RAW.difference(raw.columns))
    if missing:
        raise ValueError(f"dividend source missing columns: {missing}")
    out = pd.DataFrame(
        {
            "asset": raw["SECURITY_CODE"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6),
            "asset_name": raw["SECURITY_NAME_ABBR"].astype(str).str.strip(),
            "report_date": pd.to_datetime(raw["REPORT_DATE"], errors="coerce"),
            "plan_announcement_date": pd.to_datetime(raw["PLAN_NOTICE_DATE"], errors="coerce"),
            "announcement_date": pd.to_datetime(raw["NOTICE_DATE"], errors="coerce"),
            "ex_date": pd.to_datetime(raw["EX_DIVIDEND_DATE"], errors="coerce"),
            "cash_per_share": pd.to_numeric(raw["PRETAX_BONUS_RMB"], errors="coerce") / 10.0,
            "status": raw["ASSIGN_PROGRESS"].astype(str).str.strip(),
        }
    )
    as_of_date = pd.Timestamp(as_of).normalize()
    implemented = out["status"].eq("实施分配")
    valid_asset = out["asset"].str.fullmatch(r"\d{6}") & ~out["asset"].str.startswith(("200", "900"))
    valid = (
        implemented
        & valid_asset
        & out[["report_date", "announcement_date", "ex_date", "cash_per_share"]].notna().all(axis=1)
        & out["cash_per_share"].gt(0)
        & out["announcement_date"].le(as_of_date)
    )
    out = out.loc[valid].copy()
    lifecycle_stats = {"reassigned_events": 0, "dropped_outside_lifecycle": 0}
    if master is not None:
        out, lifecycle_stats = reconcile_dividend_lifecycles(out, master, lineage or {"code_migrations": []})
    else:
        out["security_code_resolution"] = "direct"
    out["pay_date"] = pd.NaT
    out["available_date"] = out["announcement_date"]
    out["data_source"] = "eastmoney.RPT_SHAREBONUS_DET.implemented"
    out["source_vintage"] = source_vintage

    key = ["asset", "announcement_date", "ex_date", "cash_per_share"]
    conflicts = out.groupby(key, dropna=False)["report_date"].nunique().gt(1)
    if conflicts.any():
        # Identical implemented events can be repeated under multiple report periods. This is
        # source duplication, not two cash flows; deterministic de-duplication is required.
        out = out.sort_values([*key, "report_date"]).drop_duplicates(key, keep="last")
    else:
        out = out.drop_duplicates(key)
    if out.empty or out.duplicated(key).any():
        raise ValueError("no unique implemented cash-dividend events")
    if (out["announcement_date"] < out["plan_announcement_date"]).any():
        raise ValueError("final dividend notice precedes the plan notice")

    columns = [
        "asset",
        "asset_name",
        "report_date",
        "plan_announcement_date",
        "announcement_date",
        "ex_date",
        "pay_date",
        "cash_per_share",
        "status",
        "security_code_resolution",
        "available_date",
        "data_source",
        "source_vintage",
    ]
    result = out[columns].sort_values(["announcement_date", "asset", "ex_date"]).reset_index(drop=True)
    result.attrs["lifecycle_reconciliation"] = lifecycle_stats
    return result


def run(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    as_of: str | pd.Timestamp,
    start_year: int = 2000,
    reuse_latest: bool = False,
) -> dict[str, object]:
    periods = report_periods(start_year, as_of)
    if reuse_latest:
        if not manifest_path.is_file():
            raise ValueError("cannot reuse dividend raw data without a prior manifest")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior_inputs = [item for item in prior.get("inputs", []) if "raw_stock_dividends" in str(item.get("path", ""))]
        if not prior_inputs:
            raise ValueError("prior dividend manifest has no reusable raw input")
        raw_path = ROOT / str(prior_inputs[0]["path"])
        raw_hash = str(prior_inputs[0]["sha256"])
        if not raw_path.is_file() or _sha256(raw_path) != raw_hash:
            raise ValueError("reusable dividend raw input failed hash validation")
        raw = pd.read_csv(raw_path, low_memory=False)
        nonempty_periods = list(prior.get("nonempty_periods", []))
    else:
        raw, nonempty_periods = fetch_raw(periods)
        raw_dir.mkdir(parents=True, exist_ok=True)
        temporary_raw = raw_dir / ".eastmoney_sharebonus.csv.tmp"
        raw.to_csv(temporary_raw, index=False, encoding="utf-8-sig")
        raw_hash = _sha256(temporary_raw)
        raw_path = raw_dir / f"eastmoney_sharebonus_{raw_hash[:16]}.csv"
        if raw_path.exists():
            temporary_raw.unlink()
        else:
            temporary_raw.replace(raw_path)

    if not MASTER_PATH.is_file() or not LINEAGE_PATH.is_file():
        raise ValueError("stock master and security lineage config are required")
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str})
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    master_hash = _sha256(MASTER_PATH)
    lineage_hash = _sha256(LINEAGE_PATH)
    bundle_hash = hashlib.sha256(f"{raw_hash}:{master_hash}:{lineage_hash}".encode()).hexdigest()
    vintage = f"eastmoney_sharebonus_lifecycle_bundle_sha256:{bundle_hash}"
    output = build_stock_dividend_events(raw, vintage, as_of, master, lineage)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary_output, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary_output.replace(output_path)

    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "source_url": SOURCE_URL,
        "query_periods": periods,
        "nonempty_periods": nonempty_periods,
        "inputs": [
            {"source_id": "eastmoney_sharebonus", "path": _relative(raw_path), "sha256": raw_hash},
            {"source_id": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": master_hash},
            {"source_id": "security_code_lineage", "path": _relative(LINEAGE_PATH), "sha256": lineage_hash},
        ],
        "input_sha256": bundle_hash,
        "source_vintage": vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "raw_rows": int(len(raw)),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "coverage_start": str(output["announcement_date"].min().date()),
        "coverage_end": str(output["announcement_date"].max().date()),
        "lifecycle_reconciliation": output.attrs.get("lifecycle_reconciliation", {}),
        "limitations": [
            "source exposes implemented final terms but not every historical plan revision",
            "pay_date is unavailable and remains null; ex_date drives total-return treatment",
            "code migrations are governed by configs/long_hold_v4_security_lineage.json",
        ],
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--reuse-latest", action="store_true")
    args = parser.parse_args()
    result = run(args.raw_dir, args.output, args.manifest, args.as_of, args.start_year, args.reuse_latest)
    summary_keys = (
        "rows",
        "assets",
        "coverage_start",
        "coverage_end",
        "lifecycle_reconciliation",
        "historical_backtest_allowed",
        "model_promotion_allowed",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
