"""Build conservative PIT stock fundamentals from final statement snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from strategy_lab.long_hold_v4.pit_stock_dividend_builder import report_periods


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_fundamentals"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_fundamentals_pit.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_fundamentals_builder_latest.json"
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
LINEAGE_PATH = ROOT / "configs" / "long_hold_v4_security_lineage.json"
SOURCE_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORT_SPECS = {
    "performance": {
        "report_name": "RPT_LICO_FN_CPD",
        "report_column": "REPORTDATE",
        "sort_columns": "UPDATE_DATE,SECURITY_CODE",
    },
    "balance": {
        "report_name": "RPT_DMSK_FN_BALANCE",
        "report_column": "REPORT_DATE",
        "sort_columns": "NOTICE_DATE,SECURITY_CODE",
    },
    "cashflow": {
        "report_name": "RPT_DMSK_FN_CASHFLOW",
        "report_column": "REPORT_DATE",
        "sort_columns": "NOTICE_DATE,SECURITY_CODE",
    },
}
REQUIRED_COLUMNS = {
    "performance": {
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "REPORTDATE",
        "NOTICE_DATE",
        "UPDATE_DATE",
        "TOTAL_OPERATE_INCOME",
        "PARENT_NETPROFIT",
        "WEIGHTAVG_ROE",
    },
    "balance": {"SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "TOTAL_ASSETS", "TOTAL_LIABILITIES"},
    "cashflow": {"SECURITY_CODE", "REPORT_DATE", "NOTICE_DATE", "NETCASH_OPERATE"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _filter(period: str, report_column: str) -> str:
    return (
        '(SECURITY_TYPE_CODE in ("058001001","058001008"))'
        '(TRADE_MARKET_CODE!="069001017")'
        f"({report_column}='{period}')"
    )


def _request_page(
    session: requests.Session, dataset: str, period: str, page: int, attempts: int = 3
) -> dict:
    spec = REPORT_SPECS[dataset]
    params = {
        "sortColumns": spec["sort_columns"],
        "sortTypes": "-1,-1",
        "pageSize": "500",
        "pageNumber": str(page),
        "reportName": spec["report_name"],
        "columns": "ALL",
        "filter": _filter(period, spec["report_column"]),
        "source": "WEB",
        "client": "WEB",
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
    raise RuntimeError(f"failed fundamentals query dataset={dataset};period={period};page={page}") from last_error


def _fetch_period(session: requests.Session, dataset: str, period: str) -> pd.DataFrame:
    first = _request_page(session, dataset, period, 1)
    result = first.get("result")
    if not result or not result.get("data"):
        return pd.DataFrame()
    rows = list(result["data"])
    for page in range(2, int(result.get("pages", 1)) + 1):
        page_result = _request_page(session, dataset, period, page).get("result") or {}
        rows.extend(page_result.get("data") or [])
        time.sleep(0.03)
    frame = pd.DataFrame(rows)
    frame["query_report_date"] = period
    return frame


def _snapshot(frame: pd.DataFrame, directory: Path, period: str) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{period}.csv.tmp"
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    digest = _sha256(temporary)
    destination = directory / f"{period}_{digest[:16]}.csv"
    if destination.exists():
        temporary.unlink()
    else:
        temporary.replace(destination)
    return destination, digest


def _cached_snapshot(directory: Path, period: str) -> Path | None:
    matches = list(directory.glob(f"{period}_*.csv")) if directory.is_dir() else []
    return max(matches, key=lambda path: path.stat().st_mtime_ns) if matches else None


def acquire(
    raw_dir: Path, periods: list[str], refresh: bool = False
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    frames: dict[str, list[pd.DataFrame]] = {dataset: [] for dataset in REPORT_SPECS}
    inputs: list[dict[str, object]] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        for dataset in REPORT_SPECS:
            directory = raw_dir / dataset
            for period in periods:
                cached = None if refresh else _cached_snapshot(directory, period)
                if cached is not None:
                    frame = pd.read_csv(cached, dtype={"SECURITY_CODE": str}, low_memory=False)
                    path, digest = cached, _sha256(cached)
                    source = "cache"
                else:
                    frame = _fetch_period(session, dataset, period)
                    if frame.empty:
                        continue
                    path, digest = _snapshot(frame, directory, period)
                    source = "network"
                frames[dataset].append(frame)
                inputs.append(
                    {
                        "dataset": dataset,
                        "period": period,
                        "path": _relative(path),
                        "sha256": digest,
                        "rows": int(len(frame)),
                        "loaded_from": source,
                    }
                )
    if any(not values for values in frames.values()):
        missing = [dataset for dataset, values in frames.items() if not values]
        raise ValueError(f"fundamental source returned no records for: {missing}")
    return {dataset: pd.concat(values, ignore_index=True) for dataset, values in frames.items()}, inputs


def _latest(frame: pd.DataFrame, dataset: str, date_column: str) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS[dataset].difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset} source missing columns: {missing}")
    work = frame.copy()
    work["asset"] = work["SECURITY_CODE"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    report_column = REPORT_SPECS[dataset]["report_column"]
    work[report_column] = pd.to_datetime(work[report_column], errors="coerce")
    work = work[work["asset"].str.fullmatch(r"\d{6}") & ~work["asset"].str.startswith(("200", "900"))]
    return (
        work.sort_values(["asset", report_column, date_column])
        .drop_duplicates(["asset", report_column], keep="last")
        .reset_index(drop=True)
    )


def _lifecycle_table(master: pd.DataFrame) -> pd.DataFrame:
    required = {"asset", "asset_name", "event_type", "list_date"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"stock security master missing lifecycle columns: {missing}")
    listed = master[master["event_type"].eq("listing")][["asset", "asset_name", "list_date"]].copy()
    exits = master[master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listed["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("stock security master contains duplicate lifecycle events")
    lifecycle = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    lifecycle["asset"] = lifecycle["asset"].astype(str).str.zfill(6)
    lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"], errors="coerce").dt.normalize()
    lifecycle["delist_date"] = pd.to_datetime(lifecycle["delist_date"], errors="coerce").dt.normalize()
    if lifecycle[["asset", "list_date"]].isna().any(axis=None):
        raise ValueError("stock security master contains incomplete lifecycle keys")
    return lifecycle


def reconcile_fundamental_lifecycles(
    records: pd.DataFrame, master: pd.DataFrame, lineage: dict
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map entity-level reports to the security code tradable when information became available."""

    lifecycle = _lifecycle_table(master)
    by_asset = lifecycle.set_index("asset")
    output = records.copy()
    output["source_asset"] = output["asset"]
    output["security_code_resolution"] = "direct"
    reassigned_predecessor = 0
    reassigned_successor = 0
    carried_forward = 0

    for item in lineage.get("code_migrations", []):
        predecessor = str(item["predecessor"]).zfill(6)
        successor = str(item["successor"]).zfill(6)
        if predecessor not in by_asset.index or successor not in by_asset.index:
            raise ValueError(f"fundamental lineage is absent from master: {predecessor}->{successor}")
        predecessor_end = pd.Timestamp(item["predecessor_end_date"]).normalize()
        successor_start = pd.Timestamp(item["successor_start_date"]).normalize()

        early_successor = output["asset"].eq(successor) & output["available_date"].lt(successor_start)
        reassigned_predecessor += int(early_successor.sum())
        output.loc[early_successor, "asset"] = predecessor
        output.loc[early_successor, "asset_name"] = str(by_asset.loc[predecessor, "asset_name"])
        output.loc[early_successor, "security_code_resolution"] = "reassigned_predecessor"

        late_predecessor = output["asset"].eq(predecessor) & output["available_date"].gt(predecessor_end)
        reassigned_successor += int(late_predecessor.sum())
        output.loc[late_predecessor, "asset"] = successor
        output.loc[late_predecessor, "asset_name"] = str(by_asset.loc[successor, "asset_name"])
        output.loc[late_predecessor, "security_code_resolution"] = "reassigned_successor"

        predecessor_history = output[
            output["asset"].eq(predecessor) & output["available_date"].le(predecessor_end)
        ].copy()
        if not predecessor_history.empty:
            carry = predecessor_history.copy()
            carry["asset"] = successor
            carry["asset_name"] = str(by_asset.loc[successor, "asset_name"])
            carry["available_date"] = successor_start
            carry["security_code_resolution"] = "carried_forward_successor"
            carried_forward += len(carry)
            output = pd.concat([output, carry], ignore_index=True)

    before_lifecycle = len(output)
    output = output.merge(
        lifecycle[["asset", "list_date", "delist_date"]], on="asset", how="inner", validate="many_to_one"
    )
    delayed_to_listing = output["available_date"].lt(output["list_date"])
    output.loc[delayed_to_listing, "available_date"] = output.loc[delayed_to_listing, "list_date"]
    within_lifecycle = output["delist_date"].isna() | output["available_date"].le(output["delist_date"])
    output = output.loc[within_lifecycle].drop(columns=["list_date", "delist_date"]).copy()

    resolution_rank = {
        "direct": 0,
        "reassigned_predecessor": 1,
        "reassigned_successor": 1,
        "carried_forward_successor": 2,
    }
    output["_resolution_rank"] = output["security_code_resolution"].map(resolution_rank).fillna(9)
    output = output.sort_values(["asset", "report_date", "available_date", "_resolution_rank"])
    output = output.drop_duplicates(["asset", "report_date", "available_date"], keep="first")
    output = output.drop(columns="_resolution_rank").reset_index(drop=True)
    stats = {
        "reassigned_predecessor_rows": reassigned_predecessor,
        "reassigned_successor_rows": reassigned_successor,
        "carried_forward_successor_rows": carried_forward,
        "delayed_to_listing_rows": int(delayed_to_listing.sum()),
        "dropped_outside_lifecycle_rows": int(before_lifecycle - len(output)),
    }
    return output, stats


def build_stock_fundamentals(
    sources: dict[str, pd.DataFrame],
    source_vintage: str,
    as_of: str | pd.Timestamp,
    master: pd.DataFrame | None = None,
    lineage: dict | None = None,
) -> pd.DataFrame:
    missing_sources = sorted(set(REPORT_SPECS).difference(sources))
    if missing_sources:
        raise ValueError(f"missing fundamental sources: {missing_sources}")

    performance = _latest(sources["performance"], "performance", "UPDATE_DATE")
    performance = performance.rename(
        columns={
            "REPORTDATE": "report_date",
            "NOTICE_DATE": "performance_ann_date",
            "UPDATE_DATE": "performance_update_date",
            "TOTAL_OPERATE_INCOME": "revenue",
            "PARENT_NETPROFIT": "net_profit",
            "WEIGHTAVG_ROE": "roe",
            "SECURITY_NAME_ABBR": "asset_name",
        }
    )[
        [
            "asset",
            "asset_name",
            "report_date",
            "performance_ann_date",
            "performance_update_date",
            "revenue",
            "net_profit",
            "roe",
        ]
    ]
    balance = _latest(sources["balance"], "balance", "NOTICE_DATE").rename(
        columns={
            "REPORT_DATE": "report_date",
            "NOTICE_DATE": "balance_ann_date",
            "TOTAL_ASSETS": "total_assets",
            "TOTAL_LIABILITIES": "total_liabilities",
        }
    )[["asset", "report_date", "balance_ann_date", "total_assets", "total_liabilities"]]
    cashflow = _latest(sources["cashflow"], "cashflow", "NOTICE_DATE").rename(
        columns={
            "REPORT_DATE": "report_date",
            "NOTICE_DATE": "cashflow_ann_date",
            "NETCASH_OPERATE": "operating_cash_flow",
        }
    )[["asset", "report_date", "cashflow_ann_date", "operating_cash_flow"]]

    out = performance.merge(balance, on=["asset", "report_date"], how="inner", validate="one_to_one")
    out = out.merge(cashflow, on=["asset", "report_date"], how="inner", validate="one_to_one")
    date_columns = [
        "performance_ann_date",
        "performance_update_date",
        "balance_ann_date",
        "cashflow_ann_date",
    ]
    for column in date_columns:
        out[column] = pd.to_datetime(out[column], errors="coerce")
        out.loc[out[column] < out["report_date"], column] = pd.NaT
    out["ann_date"] = out[["performance_ann_date", "balance_ann_date", "cashflow_ann_date"]].min(axis=1)
    out["update_date"] = out[date_columns].max(axis=1)
    # Final snapshots are withheld until their latest observable source date.
    out["available_date"] = out["update_date"]

    numeric_columns = [
        "revenue",
        "net_profit",
        "roe",
        "operating_cash_flow",
        "total_assets",
        "total_liabilities",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    as_of_date = pd.Timestamp(as_of).normalize()
    valid = (
        out[["report_date", "ann_date", "update_date", "available_date", *numeric_columns]].notna().all(axis=1)
        & out["report_date"].le(as_of_date)
        & out["available_date"].le(as_of_date)
        & out["ann_date"].ge(out["report_date"])
        & out["available_date"].ge(out["ann_date"])
        & out["total_assets"].gt(0)
        & out["total_liabilities"].ge(0)
    )
    out = out.loc[valid].copy()
    if master is not None:
        out, lifecycle_stats = reconcile_fundamental_lifecycles(
            out, master, lineage or {"code_migrations": []}
        )
    else:
        out["source_asset"] = out["asset"]
        out["security_code_resolution"] = "unreconciled_fixture"
        lifecycle_stats = {}
    out["data_source"] = "eastmoney.final_financial_snapshots.conservative_availability"
    out["source_vintage"] = source_vintage
    columns = [
        "asset",
        "asset_name",
        "report_date",
        "ann_date",
        "update_date",
        "available_date",
        "source_asset",
        "security_code_resolution",
        *numeric_columns,
        "data_source",
        "source_vintage",
    ]
    out = out[columns].sort_values(["asset", "report_date", "available_date"]).reset_index(drop=True)
    if out.empty or out.duplicated(["asset", "report_date", "available_date"]).any():
        raise ValueError("fundamental output contains no unique PIT records")
    out.attrs["lifecycle_reconciliation"] = lifecycle_stats
    return out


def run(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    as_of: str | pd.Timestamp,
    start_year: int = 2004,
    refresh: bool = False,
) -> dict[str, object]:
    periods = report_periods(start_year, as_of)
    sources, inputs = acquire(raw_dir, periods, refresh)
    if not MASTER_PATH.is_file() or not LINEAGE_PATH.is_file():
        raise ValueError("stock master and security lineage config are required")
    master = pd.read_csv(MASTER_PATH, dtype={"asset": str}, low_memory=False)
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    master_hash = _sha256(MASTER_PATH)
    lineage_hash = _sha256(LINEAGE_PATH)
    inputs.extend(
        [
            {"dataset": "governance", "period": "", "path": _relative(MASTER_PATH), "sha256": master_hash, "rows": len(master), "loaded_from": "local"},
            {"dataset": "governance", "period": "", "path": _relative(LINEAGE_PATH), "sha256": lineage_hash, "rows": len(lineage.get("code_migrations", [])), "loaded_from": "local"},
        ]
    )
    bundle = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in inputs], sort_keys=True).encode("utf-8")
    ).hexdigest()
    vintage = f"eastmoney_financial_bundle_sha256:{bundle}"
    output = build_stock_fundamentals(sources, vintage, as_of, master, lineage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(output_path)
    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "source_url": SOURCE_URL,
        "query_periods": periods,
        "inputs": inputs,
        "input_sha256": bundle,
        "source_vintage": vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "coverage_start": str(output["report_date"].min().date()),
        "coverage_end": str(output["report_date"].max().date()),
        "lifecycle_reconciliation": output.attrs.get("lifecycle_reconciliation", {}),
        "limitations": [
            "source supplies final snapshots rather than every filing revision",
            "final values are withheld until the maximum observed statement/update date",
            "balance-sheet and cash-flow endpoints lack row revision timestamps; their final snapshots are observation-only",
        ],
        "qualification_status": "OBSERVATION_ONLY_MISSING_VERSIONED_STATEMENTS",
        "historical_backtest_allowed": False,
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
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = run(args.raw_dir, args.output, args.manifest, args.as_of, args.start_year, args.refresh)
    summary_keys = (
        "qualification_status",
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
