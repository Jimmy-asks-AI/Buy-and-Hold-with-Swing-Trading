"""Build an event-sourced SSE/SZSE lifecycle master with JoinQuant reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_master"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_master_builder_latest.json"
LINEAGE_PATH = ROOT / "configs" / "long_hold_v4_security_lineage.json"

SOURCE_URLS = {
    "sse_main": "https://www.sse.com.cn/assortment/stock/list/share/",
    "sse_star": "https://www.sse.com.cn/assortment/stock/list/share/",
    "sse_delisted": "https://www.sse.com.cn/assortment/stock/list/delisting/",
    "szse_a": "https://www.szse.cn/market/product/stock/list/index.html",
    "szse_delisted": "https://www.szse.cn/market/stock/suspend/index.html",
    "bse_current": "https://www.bse.cn/nq/listedcompany.html",
    "joinquant_stock": "https://www.joinquant.com/help/api/doc?id=10029&name=JQDatadoc",
}

SOURCE_FIELDS = {
    "sse_main": ("SSE", "证券代码", "证券简称", "上市日期", None),
    "sse_star": ("SSE", "证券代码", "证券简称", "上市日期", None),
    "sse_delisted": ("SSE", "A_STOCK_CODE", "COMPANY_ABBR", "LIST_DATE", "DELIST_DATE"),
    "szse_a": ("SZSE", "A股代码", "A股简称", "A股上市日期", None),
    "szse_delisted": ("SZSE", "证券代码", "证券简称", "上市日期", "终止上市日期"),
    "bse_current": ("BSE", "证券代码", "证券简称", "上市日期", None),
}


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


def _is_a_share(asset: pd.Series, source_id: str) -> pd.Series:
    valid = asset.str.fullmatch(r"\d{6}")
    # Current-list endpoints and the raw SSE query are already A-share scoped. SZSE's
    # terminated-company sheet also contains 200xxx B shares. SSE's STAR list includes
    # 689xxx CDRs; both are outside this common-stock universe.
    return (
        valid
        & ~((source_id == "szse_delisted") & asset.str.startswith("200"))
        & ~asset.str.startswith("689")
    )


def _normalise_source(source_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    exchange, asset_col, name_col, list_col, delist_col = SOURCE_FIELDS[source_id]
    required = {asset_col, name_col, list_col, *([delist_col] if delist_col else [])}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_id} missing columns: {missing}")

    out = pd.DataFrame(
        {
            "asset": _asset_codes(frame[asset_col]),
            "asset_name": frame[name_col].astype(str).str.strip(),
            "list_date": pd.to_datetime(frame[list_col], errors="coerce"),
            "delist_date": pd.to_datetime(frame[delist_col], errors="coerce") if delist_col else pd.NaT,
            "exchange": exchange,
            "source_id": source_id,
        }
    )
    out = out[_is_a_share(out["asset"], source_id)].copy()
    if out.empty or out[["asset", "list_date"]].isna().any(axis=None):
        raise ValueError(f"{source_id} contains no usable records or missing lifecycle keys")
    if out.duplicated("asset").any():
        duplicates = sorted(out.loc[out.duplicated("asset", keep=False), "asset"].unique())[:10]
        raise ValueError(f"{source_id} contains duplicate assets: {duplicates}")
    bad_order = out["delist_date"].notna() & (out["delist_date"] < out["list_date"])
    if bad_order.any():
        raise ValueError(f"{source_id} contains delist dates before list dates")
    return out.reset_index(drop=True)


def _normalise_joinquant(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"code", "display_name", "start_date", "end_date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"joinquant_stock missing columns: {missing}")
    code = frame["code"].astype(str)
    out = pd.DataFrame(
        {
            "asset": code.str[:6],
            "asset_name": frame["display_name"].astype(str).str.strip(),
            "list_date": pd.to_datetime(frame["start_date"], errors="coerce"),
            "delist_date": pd.to_datetime(frame["end_date"], errors="coerce"),
            "exchange": code.str[-4:].map({"XSHG": "SSE", "XSHE": "SZSE"}),
            "source_id": "joinquant_stock",
        }
    )
    out.loc[out["delist_date"].dt.year.ge(2200), "delist_date"] = pd.NaT
    out = out[
        out["asset"].str.fullmatch(r"\d{6}")
        & out["exchange"].notna()
        & ~out["asset"].str.startswith("689")
    ].copy()
    if out.empty or out[["asset", "list_date"]].isna().any(axis=None) or out.duplicated("asset").any():
        raise ValueError("joinquant_stock contains invalid or duplicate lifecycle keys")
    return out.reset_index(drop=True)


def build_stock_security_master(
    sources: dict[str, pd.DataFrame],
    source_vintage: str,
    as_of: str | pd.Timestamp,
    lineage: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Reconcile exchange/JQ lifecycle tables into as-of-safe listing events."""

    required_sources = {"sse_main", "sse_star", "sse_delisted", "szse_a", "szse_delisted"}
    missing_sources = sorted(required_sources.difference(sources))
    if missing_sources:
        raise ValueError(f"missing stock master sources: {missing_sources}")
    as_of_date = pd.Timestamp(as_of).normalize()
    if pd.isna(as_of_date):
        raise ValueError("as_of must be a valid date")

    exchange_sources = {source_id: frame for source_id, frame in sources.items() if source_id != "joinquant_stock"}
    normalised = [_normalise_source(source_id, frame) for source_id, frame in exchange_sources.items()]
    entities: dict[str, dict[str, Any]] = {}
    for row in pd.concat(normalised, ignore_index=True).itertuples(index=False):
        candidate = row._asdict()
        existing = entities.get(row.asset)
        if existing is None:
            candidate["source_ids"] = {row.source_id}
            candidate["lifecycle_resolution"] = "exchange_only"
            entities[row.asset] = candidate
            continue
        if existing["exchange"] != row.exchange or pd.Timestamp(existing["list_date"]) != pd.Timestamp(row.list_date):
            raise ValueError(f"conflicting lifecycle identity for {row.asset}")
        existing["source_ids"].add(row.source_id)
        if pd.notna(row.delist_date):
            prior = existing["delist_date"]
            if pd.notna(prior) and pd.Timestamp(prior) != pd.Timestamp(row.delist_date):
                raise ValueError(f"conflicting delist dates for {row.asset}")
            existing["delist_date"] = row.delist_date
            existing["asset_name"] = row.asset_name

    if "joinquant_stock" in sources:
        for row in _normalise_joinquant(sources["joinquant_stock"]).itertuples(index=False):
            existing = entities.get(row.asset)
            if existing is None:
                candidate = row._asdict()
                candidate["source_ids"] = {row.source_id}
                candidate["lifecycle_resolution"] = "joinquant_only"
                entities[row.asset] = candidate
                continue
            if existing["exchange"] != row.exchange:
                raise ValueError(f"exchange conflict against JoinQuant for {row.asset}")
            existing["source_ids"].add(row.source_id)
            exchange_start = pd.Timestamp(existing["list_date"])
            jq_start = pd.Timestamp(row.list_date)
            if jq_start > exchange_start + pd.Timedelta(days=365):
                existing["list_date"] = jq_start
                existing["lifecycle_resolution"] = "joinquant_late_start_guard"
            elif jq_start == exchange_start:
                existing["lifecycle_resolution"] = "cross_source_match"
            else:
                existing["lifecycle_resolution"] = "exchange_a_share_date"
            if pd.isna(existing["delist_date"]) and pd.notna(row.delist_date):
                existing["delist_date"] = row.delist_date
                existing["asset_name"] = row.asset_name

    _apply_code_migrations(entities, lineage or {"code_migrations": []})

    events: list[dict[str, Any]] = []
    for asset, entity in entities.items():
        list_date = pd.Timestamp(entity["list_date"]).normalize()
        if list_date > as_of_date:
            continue
        source_ids = "+".join(sorted(entity["source_ids"]))
        common = {
            "asset": asset,
            "asset_name": entity["asset_name"],
            "list_date": list_date,
            "exchange": entity["exchange"],
            "lifecycle_resolution": entity["lifecycle_resolution"],
            "data_source": source_ids,
            "source_vintage": source_vintage,
            "predecessor_asset": entity.get("predecessor_asset", ""),
            "successor_asset": entity.get("successor_asset", ""),
        }
        events.append(
            {
                **common,
                "delist_date": pd.NaT,
                "list_status": "listed",
                "event_type": "listing",
                "available_date": list_date,
            }
        )
        delist_date = entity["delist_date"]
        if pd.notna(delist_date) and pd.Timestamp(delist_date).normalize() <= as_of_date:
            delist_date = pd.Timestamp(delist_date).normalize()
            events.append(
                {
                    **common,
                    "delist_date": delist_date,
                    "list_status": "delisted",
                    "event_type": "delisting",
                    "available_date": delist_date,
                }
            )

    out = pd.DataFrame(events)
    columns = [
        "asset",
        "asset_name",
        "list_date",
        "delist_date",
        "list_status",
        "event_type",
        "exchange",
        "lifecycle_resolution",
        "available_date",
        "data_source",
        "source_vintage",
        "predecessor_asset",
        "successor_asset",
    ]
    out = out[columns].sort_values(["asset", "available_date", "event_type"]).reset_index(drop=True)
    _validate_events(out, as_of_date)
    return out


def _apply_code_migrations(entities: dict[str, dict[str, Any]], lineage: dict[str, Any]) -> None:
    for item in lineage.get("code_migrations", []):
        predecessor = str(item["predecessor"]).zfill(6)
        successor = str(item["successor"]).zfill(6)
        if item.get("identity_continuity") is not True:
            raise ValueError(f"code migration lacks identity continuity approval: {predecessor}->{successor}")
        if predecessor not in entities or successor not in entities:
            raise ValueError(f"code migration is absent from security sources: {predecessor}->{successor}")
        predecessor_end = pd.Timestamp(item["predecessor_end_date"]).normalize()
        successor_start = pd.Timestamp(item["successor_start_date"]).normalize()
        effective_date = pd.Timestamp(item["effective_date"]).normalize()
        if predecessor_end >= successor_start or effective_date > successor_start:
            raise ValueError(f"invalid code migration interval: {predecessor}->{successor}")

        prior_end = entities[predecessor]["delist_date"]
        if pd.notna(prior_end) and pd.Timestamp(prior_end).normalize() != predecessor_end:
            raise ValueError(f"code migration conflicts with predecessor end: {predecessor}->{successor}")
        entities[predecessor]["delist_date"] = predecessor_end
        entities[predecessor]["successor_asset"] = successor
        entities[predecessor]["lifecycle_resolution"] = "official_code_migration"
        entities[successor]["list_date"] = successor_start
        entities[successor]["predecessor_asset"] = predecessor
        entities[successor]["lifecycle_resolution"] = "official_code_migration"


def _validate_events(frame: pd.DataFrame, as_of: pd.Timestamp) -> None:
    if frame.empty or frame.duplicated(["asset", "available_date", "list_status"]).any():
        raise ValueError("stock lifecycle event keys must be non-empty and unique")
    if (frame["available_date"] > as_of).any() or (frame["available_date"] < frame["list_date"]).any():
        raise ValueError("stock lifecycle event availability is invalid")
    listed = frame["list_status"].eq("listed")
    delisted = frame["list_status"].eq("delisted")
    if frame.loc[listed, "delist_date"].notna().any():
        raise ValueError("listing events must not expose future delist dates")
    if frame.loc[delisted, "delist_date"].isna().any():
        raise ValueError("delisting events require delist_date")
    if not (frame.loc[delisted, "available_date"] == frame.loc[delisted, "delist_date"]).all():
        raise ValueError("delisting facts must become available on their event date")
    first_status = frame.sort_values(["asset", "available_date"]).groupby("asset", sort=False)["list_status"].first()
    if not first_status.eq("listed").all():
        raise ValueError("every lifecycle must begin with a listing event")


def _fetch_sources(include_bse: bool) -> dict[str, pd.DataFrame]:
    import akshare as ak

    sources = {
        "sse_main": ak.stock_info_sh_name_code(symbol="主板A股"),
        "sse_star": ak.stock_info_sh_name_code(symbol="科创板"),
        "sse_delisted": _fetch_sse_delisted_a_shares(),
        "szse_a": ak.stock_info_sz_name_code(symbol="A股列表"),
        "szse_delisted": ak.stock_info_sz_delist(symbol="终止上市公司"),
        "joinquant_stock": _fetch_joinquant_stock_master(),
    }
    if include_bse:
        sources["bse_current"] = ak.stock_info_bj_name_code()
    return sources


def _fetch_joinquant_stock_master() -> pd.DataFrame:
    import jqdatasdk as jq

    credential_path = ROOT / "configs" / "data_credentials.json"
    credentials = json.loads(credential_path.read_text(encoding="utf-8-sig")).get("joinquant", {})
    username = str(credentials.get("username", "")).strip()
    password = str(credentials.get("password", "")).strip()
    if not username or not password:
        raise ValueError("JoinQuant credentials are required for stock lifecycle reconciliation")
    jq.auth(username, password)
    frame = jq.get_all_securities(types=["stock"])
    return frame.reset_index(names="code")


def _fetch_sse_delisted_a_shares() -> pd.DataFrame:
    """Fetch the raw A-share rows; AkShare's combined helper loses STOCK_TYPE and creates false duplicates."""

    import requests

    response = requests.get(
        "https://query.sse.com.cn/commonQuery.do",
        params={
            "sqlId": "COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L",
            "isPagination": "true",
            "STOCK_CODE": "",
            "CSRC_CODE": "",
            "REG_PROVINCE": "",
            "STOCK_TYPE": "1,8",
            "COMPANY_STATUS": "3",
            "type": "inParams",
            "pageHelp.cacheSize": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.pageSize": "500",
            "pageHelp.pageNo": "1",
            "pageHelp.endPage": "1",
        },
        headers={"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("result", [])
    total = int(payload.get("pageHelp", {}).get("total", len(rows)))
    if not rows or total != len(rows) or total > 500:
        raise ValueError(f"unexpected SSE delisted pagination: rows={len(rows)};total={total}")
    return pd.DataFrame(rows)


def _snapshot(frame: pd.DataFrame, source_id: str, raw_dir: Path) -> tuple[Path, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    temporary = raw_dir / f".{source_id}.csv.tmp"
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    digest = _sha256(temporary)
    destination = raw_dir / f"{source_id}_{digest[:16]}.csv"
    if destination.exists():
        temporary.unlink()
    else:
        temporary.replace(destination)
    return destination, digest


def _reuse_sources(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_inputs = [item for item in prior.get("inputs", []) if item.get("source_id") in SOURCE_URLS]
    sources: dict[str, pd.DataFrame] = {}
    inputs: list[dict[str, str]] = []
    for item in source_inputs:
        source_id = str(item["source_id"])
        path = ROOT / str(item["path"])
        digest = str(item["sha256"])
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"reusable stock-master source failed hash validation: {source_id}")
        # Exchange dates use compact YYYYMMDD values.  String loading is required;
        # integer inference would make pandas interpret them as nanoseconds in 1970.
        sources[source_id] = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        inputs.append(
            {
                "source_id": source_id,
                "source_url": str(item.get("source_url", SOURCE_URLS[source_id])),
                "path": _relative(path),
                "sha256": digest,
            }
        )
    required = {"sse_main", "sse_star", "sse_delisted", "szse_a", "szse_delisted"}
    if not required.issubset(sources) or "joinquant_stock" not in sources:
        raise ValueError("prior stock-master manifest lacks required reusable sources")
    return sources, inputs


def run(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    as_of: str | pd.Timestamp,
    include_bse: bool = False,
    reuse_latest: bool = False,
) -> dict[str, object]:
    if reuse_latest:
        sources, inputs = _reuse_sources(manifest_path)
    else:
        sources = _fetch_sources(include_bse)
        inputs = []
        for source_id, frame in sources.items():
            path, digest = _snapshot(frame, source_id, raw_dir)
            inputs.append(
                {
                    "source_id": source_id,
                    "source_url": SOURCE_URLS[source_id],
                    "path": _relative(path),
                    "sha256": digest,
                }
            )
    lineage = json.loads(LINEAGE_PATH.read_text(encoding="utf-8"))
    lineage_hash = _sha256(LINEAGE_PATH)
    inputs.append(
        {
            "source_id": "security_code_lineage",
            "source_url": "governed_official_evidence_registry",
            "path": _relative(LINEAGE_PATH),
            "sha256": lineage_hash,
        }
    )
    bundle = hashlib.sha256(
        json.dumps([(item["source_id"], item["sha256"]) for item in inputs], sort_keys=True).encode("utf-8")
    ).hexdigest()
    vintage = f"reconciled_stock_master_bundle_sha256:{bundle}"
    output = build_stock_security_master(sources, vintage, as_of, lineage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(output_path)

    result: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "universe_scope": ["SSE", "SZSE"] if not include_bse else ["SSE", "SZSE", "BSE"],
        "limitations": [
            "JoinQuant is a lifecycle cross-check; exchange A-share dates remain authoritative except for guarded code migrations or reuse",
            *(
                ["BSE excluded until an all-status lifecycle source is available"]
                if not include_bse
                else ["BSE current list has no independently verified delisted-security history"]
            ),
            "689xxx CDRs are excluded from the common-stock universe",
        ],
        "inputs": inputs,
        "input_sha256": bundle,
        "source_vintage": vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "listed_events": int(output["list_status"].eq("listed").sum()),
        "delisted_events": int(output["list_status"].eq("delisted").sum()),
        "lifecycle_resolution_counts": {
            str(key): int(value)
            for key, value in output.loc[output["list_status"].eq("listed"), "lifecycle_resolution"].value_counts().items()
        },
        "code_migrations": lineage.get("code_migrations", []),
        "coverage_start": str(output["list_date"].min().date()),
        "coverage_end": str(output["list_date"].max().date()),
        "historical_backtest_allowed": not include_bse,
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
    parser.add_argument("--include-bse", action="store_true")
    parser.add_argument("--reuse-latest", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.raw_dir, args.output, args.manifest, args.as_of, args.include_bse, args.reuse_latest),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
