"""Build a conservative observation-only SWS stock-industry history panel.

The current SWS workbook is a present-day snapshot with effective dates and
row update timestamps.  It is useful for schema and interval research, but it
is not, by itself, proof that every row was available historically.  This
builder therefore delays availability to the day after the recorded update
timestamp and always keeps historical backtest approval disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_sws_stock_industry"
    / "2026-07-12"
    / "StockClassifyUse_stock.xls"
)
DEFAULT_SECURITY_MASTER = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_industry_history.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_industry_builder_latest.json"
LINEAGE_PATH = ROOT / "configs" / "long_hold_v4_security_lineage.json"

SOURCE_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
SOURCE_SNAPSHOT_DATE = "2026-07-12"
KNOWN_LOCAL_CACHE_SHA256 = "8da3f757895a6d77a19d8f690cca3b3022fa0da56533cdb4769f5939b4bc49d2"
ORIGINAL_LOCAL_CACHE_PATH = (
    "E:/\u6295\u8d44/A\u80a1\u6295\u8d44\u6846\u67b6/\u4f4e\u4f30\u8d44\u4ea7\u53d1\u73b0/"
    "data_catalog/cache/etf_sw_industry_mapping/2026-07-12/StockClassifyUse_stock.xls"
)
SOURCE_COLUMNS = {
    "\u80a1\u7968\u4ee3\u7801": "asset",
    "\u8ba1\u5165\u65e5\u671f": "effective_from",
    "\u884c\u4e1a\u4ee3\u7801": "industry_code",
    "\u66f4\u65b0\u65e5\u671f": "information_time",
}
SOURCE_ASSET_COLUMN = next(column for column, canonical in SOURCE_COLUMNS.items() if canonical == "asset")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _asset_codes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


def _lifecycle_table(security_master: pd.DataFrame) -> pd.DataFrame:
    required = {"asset", "event_type", "list_date"}
    missing = sorted(required.difference(security_master.columns))
    if missing:
        raise ValueError(f"stock security master missing lifecycle columns: {missing}")
    listed = security_master[security_master["event_type"].eq("listing")][["asset", "list_date"]].copy()
    exits = security_master[security_master["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    if listed["asset"].duplicated().any() or exits["asset"].duplicated().any():
        raise ValueError("stock security master contains duplicate lifecycle events")
    lifecycle = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    lifecycle["asset"] = _asset_codes(lifecycle["asset"])
    lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"], errors="coerce").dt.normalize()
    lifecycle["delist_date"] = pd.to_datetime(lifecycle["delist_date"], errors="coerce").dt.normalize()
    if lifecycle[["asset", "list_date"]].isna().any(axis=None):
        raise ValueError("stock security master contains incomplete lifecycle keys")
    return lifecycle


def _expand_code_migrations(
    frame: pd.DataFrame, lifecycle: pd.DataFrame, lineage: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    output["source_effective_from"] = output["effective_from"]
    output["security_code_resolution"] = "direct"
    lifecycle_assets = set(lifecycle["asset"])
    reassigned = 0
    carried = 0
    for item in lineage.get("code_migrations", []):
        predecessor = str(item["predecessor"]).zfill(6)
        successor = str(item["successor"]).zfill(6)
        successor_start = pd.Timestamp(item["successor_start_date"]).normalize()
        if predecessor not in lifecycle_assets or successor not in lifecycle_assets:
            raise ValueError(f"industry lineage is absent from stock master: {predecessor}->{successor}")
        early_mask = output["asset"].eq(successor) & output["effective_from"].lt(successor_start)
        early = output.loc[early_mask].copy()
        if early.empty:
            continue
        predecessor_dates = set(output.loc[output["asset"].eq(predecessor), "source_effective_from"])
        overlap = predecessor_dates.intersection(set(early["source_effective_from"]))
        if overlap:
            raise ValueError(f"industry lineage conflicts with predecessor history: {predecessor}->{successor}")
        output = output.loc[~early_mask].copy()
        predecessor_rows = early.copy()
        predecessor_rows["asset"] = predecessor
        predecessor_rows["security_code_resolution"] = "reassigned_predecessor"
        reassigned += len(predecessor_rows)
        output = pd.concat([output, predecessor_rows], ignore_index=True)
        has_successor_start = bool(
            output["asset"].eq(successor).mul(output["effective_from"].eq(successor_start)).any()
        )
        if not has_successor_start:
            carry = early.sort_values("source_effective_from").tail(1).copy()
            carry["effective_from"] = successor_start
            carry["security_code_resolution"] = "carried_forward_successor"
            output = pd.concat([output, carry], ignore_index=True)
            carried += 1
    return output, {"reassigned_predecessor_rows": reassigned, "carried_forward_successor_rows": carried}


def build_stock_industry_history(
    source: pd.DataFrame,
    security_master: pd.DataFrame,
    source_vintage: str,
    as_of: str | pd.Timestamp,
    lineage: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Convert the SWS snapshot into non-overlapping, delayed-availability intervals."""

    missing = sorted(set(SOURCE_COLUMNS).difference(source.columns))
    if missing:
        raise ValueError(f"SWS stock classification source missing columns: {missing}")
    lifecycle = _lifecycle_table(security_master)

    as_of_date = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of_date):
        raise ValueError("as_of must be a valid date")

    frame = source[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    frame["asset"] = _asset_codes(frame["asset"])
    frame["industry_code"] = (
        frame["industry_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], errors="coerce").dt.normalize()
    frame["information_time"] = pd.to_datetime(frame["information_time"], errors="coerce")
    invalid_keys = (
        ~frame["asset"].str.fullmatch(r"\d{6}")
        | ~frame["industry_code"].str.fullmatch(r"\d{6}")
        | frame[["effective_from", "information_time"]].isna().any(axis=1)
    )
    if invalid_keys.any():
        raise ValueError(f"SWS stock classification contains invalid key rows: {int(invalid_keys.sum())}")

    eligible_assets = set(lifecycle["asset"])
    frame = frame[frame["asset"].isin(eligible_assets)].copy()
    if frame.empty:
        raise ValueError("SWS stock classification has no overlap with the stock security master")
    if frame.duplicated(["asset", "effective_from"]).any():
        duplicates = frame.loc[frame.duplicated(["asset", "effective_from"], keep=False), "asset"].unique()
        raise ValueError(f"SWS stock classification has ambiguous same-day rows: {sorted(duplicates)[:10]}")

    frame, migration_stats = _expand_code_migrations(
        frame, lifecycle, lineage or {"code_migrations": []}
    )
    frame = frame.merge(lifecycle, on="asset", how="inner", validate="many_to_one")
    within_end = frame["delist_date"].isna() | frame["effective_from"].le(frame["delist_date"])
    frame = frame.loc[within_end].copy()
    frame["lifecycle_start_clipped"] = frame["effective_from"].lt(frame["list_date"])
    frame["effective_from"] = pd.concat([frame["effective_from"], frame["list_date"]], axis=1).max(axis=1)
    frame = frame.sort_values(["asset", "effective_from", "source_effective_from"]).drop_duplicates(
        ["asset", "effective_from"], keep="last"
    )

    # A timestamp in a current workbook is not proof of pre-open availability.
    # Delaying it by one calendar day prevents same-session use.
    update_available = frame["information_time"].dt.normalize() + pd.Timedelta(days=1)
    frame["available_date"] = pd.concat(
        [frame["effective_from"], update_available, frame["list_date"]], axis=1
    ).max(axis=1)
    frame = frame[(frame["effective_from"] <= as_of_date) & (frame["available_date"] <= as_of_date)].copy()
    if frame.empty:
        raise ValueError("SWS stock classification contains no rows available by as_of")

    frame = frame.sort_values(["asset", "effective_from", "industry_code"]).reset_index(drop=True)
    next_effective = frame.groupby("asset", sort=False)["effective_from"].shift(-1)
    lifecycle_end = frame["delist_date"] + pd.Timedelta(days=1)
    frame["effective_to"] = pd.concat([next_effective, lifecycle_end], axis=1).min(axis=1)
    frame["pit_actionable"] = frame["effective_to"].isna() | (frame["available_date"] < frame["effective_to"])
    frame["classification_standard"] = "SWS_mixed_vintage_current_snapshot"
    frame["retrieval_provenance_verified"] = False
    frame["data_source"] = "SWS_StockClassifyUse_stock_local_cache_unverified_tls"
    frame["source_vintage"] = source_vintage
    frame["source_snapshot_date"] = pd.Timestamp(SOURCE_SNAPSHOT_DATE)

    columns = [
        "asset",
        "industry_code",
        "effective_from",
        "effective_to",
        "available_date",
        "information_time",
        "source_effective_from",
        "pit_actionable",
        "lifecycle_start_clipped",
        "security_code_resolution",
        "classification_standard",
        "retrieval_provenance_verified",
        "data_source",
        "source_vintage",
        "source_snapshot_date",
    ]
    output = frame[columns].copy()
    output.attrs["code_migration_reconciliation"] = migration_stats
    _validate_output(output, as_of_date)
    return output


def _validate_output(frame: pd.DataFrame, as_of: pd.Timestamp) -> None:
    if frame.empty or frame.duplicated(["asset", "industry_code", "effective_from"]).any():
        raise ValueError("stock industry history keys must be non-empty and unique")
    if frame[["asset", "industry_code", "effective_from", "available_date"]].isna().any(axis=None):
        raise ValueError("stock industry history contains missing required values")
    if (frame["available_date"] < frame["effective_from"]).any() or (frame["available_date"] > as_of).any():
        raise ValueError("stock industry history availability is invalid")
    bounded = frame["effective_to"].notna()
    if (frame.loc[bounded, "effective_to"] <= frame.loc[bounded, "effective_from"]).any():
        raise ValueError("stock industry history intervals must be positive")
    if frame["retrieval_provenance_verified"].any():
        raise ValueError("observation builder cannot claim verified retrieval provenance")


def run(
    input_path: Path,
    security_master_path: Path,
    output_path: Path,
    manifest_path: Path,
    as_of: str | pd.Timestamp,
) -> dict[str, object]:
    source = pd.read_excel(input_path, dtype=str)
    security_master = pd.read_csv(security_master_path, dtype={"asset": str}, low_memory=False)
    input_hash = _sha256(input_path)
    if input_hash != KNOWN_LOCAL_CACHE_SHA256:
        raise ValueError(
            "SWS observation input hash differs from the audited local cache; "
            "open a new source vintage instead of silently replacing it"
        )
    master_hash = _sha256(security_master_path)
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    lineage_hash = _sha256(LINEAGE_PATH)
    bundle_hash = hashlib.sha256(f"{input_hash}:{master_hash}:{lineage_hash}".encode()).hexdigest()
    source_vintage = f"sws_stock_industry_lifecycle_bundle_sha256:{bundle_hash}"
    output = build_stock_industry_history(source, security_master, source_vintage, as_of, lineage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(output_path)

    actionable = output["pit_actionable"].astype(bool)
    master_assets = set(_asset_codes(security_master["asset"]))
    covered_assets = set(output["asset"])
    missing_master_assets = sorted(master_assets.difference(covered_assets))
    source_assets = set(_asset_codes(source[SOURCE_ASSET_COLUMN]))
    delisted_assets = (
        set(_asset_codes(security_master.loc[security_master["list_status"].eq("delisted"), "asset"]))
        if "list_status" in security_master.columns
        else set()
    )
    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "source_url": SOURCE_URL,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "retrieval_provenance": {
            "status": "unverified_tls_local_cache",
            "original_local_path": ORIGINAL_LOCAL_CACHE_PATH,
            "tls_verification_passed": False,
        },
        "inputs": [
            {"path": _relative(input_path), "sha256": input_hash, "role": "sws_stock_classification_snapshot"},
            {"path": _relative(security_master_path), "sha256": master_hash, "role": "stock_security_master"},
            {"path": _relative(LINEAGE_PATH), "sha256": lineage_hash, "role": "security_code_lineage"},
        ],
        "source_vintage": source_vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "security_master_assets": len(master_assets),
        "security_master_asset_coverage_ratio": len(covered_assets) / len(master_assets),
        "missing_security_master_asset_count": len(missing_master_assets),
        "missing_security_master_asset_sample": missing_master_assets[:100],
        "source_only_asset_count": len(source_assets.difference(master_assets)),
        "delisted_master_assets": len(delisted_assets),
        "covered_delisted_assets": len(delisted_assets.intersection(covered_assets)),
        "code_migration_reconciliation": output.attrs.get("code_migration_reconciliation", {}),
        "industries": int(output["industry_code"].nunique()),
        "coverage_start": str(output["effective_from"].min().date()),
        "coverage_end": str(output["effective_from"].max().date()),
        "available_start": str(output["available_date"].min().date()),
        "available_end": str(output["available_date"].max().date()),
        "pit_actionable_rows": int(actionable.sum()),
        "retroactive_unactionable_rows": int((~actionable).sum()),
        "pit_actionable_ratio": float(actionable.mean()),
        "limitations": [
            "the workbook is a current snapshot, not a sequence of archived historical vintages",
            "the locally cached download was acquired without verified TLS provenance",
            "row update time is conservatively delayed by one calendar day and may not equal first publication time",
            "classification codes span multiple SWS standards and are intentionally not restated to a current taxonomy",
        ],
        "qualification_status": "OBSERVATION_ONLY",
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--security-master", type=Path, default=DEFAULT_SECURITY_MASTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.input, args.security_master, args.output, args.manifest, args.as_of),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
