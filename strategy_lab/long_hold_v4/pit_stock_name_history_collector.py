"""Collect governed historical A-share security-name changes.

SZSE publishes a market-wide official workbook. SSE has no equivalent public
machine-readable history in this project, so per-security Sina pages are kept
as secondary evidence. The combined result remains observation-only until its
historical status intervals pass independent cross-provider validation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "stock_security_master.csv"
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_stock_name_history"
SZSE_DIR = RAW_DIR / "szse"
SSE_DIR = RAW_DIR / "sse"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_OUTPUT = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "stock_name_history_progress.csv"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "stock_name_history_probe_latest.json"
SZSE_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=SSGSGMXX&TABKEY=tab2"
SINA_URL = "https://finance.sina.com.cn/stock/company/sh/{asset}/34.shtml"
SINA_FALLBACK_URL = "https://money.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/{asset}.phtml"
OUTPUT_COLUMNS = [
    "asset",
    "exchange",
    "effective_date",
    "old_name",
    "new_name",
    "old_status",
    "new_status",
    "available_date",
    "source_tier",
    "data_source",
    "source_vintage",
    "source_url",
    "source_artifact_sha256",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_bytes(encoded, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
        }
    )
    return session


def load_lifecycles(path: Path = MASTER_PATH, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"asset": str})
    listed_columns = ["asset", "exchange", "list_date", "asset_name", "predecessor_asset"]
    listed = frame[frame["event_type"].eq("listing")][listed_columns].copy()
    exits = frame[frame["event_type"].eq("delisting")][["asset", "delist_date"]].copy()
    output = listed.merge(exits, on="asset", how="left", validate="one_to_one")
    output["asset"] = output["asset"].astype(str).str.zfill(6)
    output["list_date"] = pd.to_datetime(output["list_date"], errors="coerce")
    output["delist_date"] = pd.to_datetime(output["delist_date"], errors="coerce")
    if output[["asset", "exchange", "list_date"]].isna().any(axis=None):
        raise ValueError("stock master contains incomplete name-history lifecycle keys")
    if as_of is not None:
        output = output[output["list_date"].le(pd.Timestamp(as_of).normalize())].copy()
    return output.sort_values("asset").reset_index(drop=True)


def classify_security_name(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "unknown"
    name = re.sub(r"\s+", "", str(value)).upper()
    if name.startswith("退市") or name.endswith("退"):
        return "delisting"
    if name.startswith("PT"):
        return "special_transfer"
    if re.match(r"^(?:N|S|G)?\*?ST", name):
        return "risk_warning"
    if name.endswith("暂停"):
        return "listing_suspended"
    if name.startswith(("N", "C")):
        return "listing_marker"
    return "normal"


def parse_szse_workbook(payload: bytes) -> pd.DataFrame:
    if not payload.startswith(b"PK"):
        raise ValueError("SZSE name-history response is not an xlsx workbook")
    frame = pd.read_excel(io.BytesIO(payload), dtype={"证券代码": str})
    required = {"变更日期", "证券代码", "变更前简称", "变更后简称"}
    if not required.issubset(frame.columns):
        raise ValueError(f"SZSE name-history workbook missing columns: {sorted(required.difference(frame.columns))}")
    return frame


def parse_sina_name_history(payload: bytes) -> pd.DataFrame:
    header = payload[:4096].decode("ascii", errors="ignore")
    declared = re.search(r"charset\s*=\s*['\"]?([A-Za-z0-9_-]+)", header, flags=re.IGNORECASE)
    encodings = [declared.group(1) if declared else None, "utf-8", "gb18030"]
    text = ""
    for encoding in dict.fromkeys(item for item in encodings if item):
        try:
            text = payload.decode(encoding)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if not text:
        raise ValueError("Sina name-history page has an unsupported character encoding")
    try:
        tables = pd.read_html(io.StringIO(text))
    except (ImportError, ValueError) as exc:
        raise ValueError("Sina name-history page contains no readable tables") from exc
    for table in tables:
        if table.shape[1] < 3:
            continue
        candidate = table.iloc[:, :3].copy()
        candidate.columns = ["change_date", "old_name", "new_name"]
        strings = candidate.fillna("").astype(str)
        header = (
            strings["change_date"].str.contains("更名日期", regex=False)
            & strings["old_name"].str.contains("更名前", regex=False)
            & strings["new_name"].str.contains("更名后", regex=False)
        )
        if header.any():
            candidate = candidate.loc[candidate.index > header[header].index[0]].copy()
        candidate["change_date"] = pd.to_datetime(candidate["change_date"], format="%Y-%m-%d", errors="coerce")
        candidate = candidate[candidate["change_date"].notna()].copy()
        if candidate.empty:
            continue
        candidate["old_name"] = candidate["old_name"].where(candidate["old_name"].notna(), "").astype(str).str.strip()
        candidate["new_name"] = candidate["new_name"].where(candidate["new_name"].notna(), "").astype(str).str.strip()
        if candidate["new_name"].eq("").any() or candidate["change_date"].duplicated().any():
            raise ValueError("Sina name-history table has blank new names or duplicate dates")
        return candidate.sort_values("change_date").reset_index(drop=True)
    # A syntactically valid company page may have no historical name changes.
    if "公司资料" in text or "证券资料" in text:
        return pd.DataFrame(columns=["change_date", "old_name", "new_name"])
    raise ValueError("Sina response is not a recognizable company page")


def parse_sina_undated_name_summary(payload: bytes) -> list[str]:
    text = ""
    for encoding in ("utf-8", "gb18030"):
        try:
            text = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise ValueError("Sina company profile has an unsupported character encoding")
    match = re.search(
        r"证券简称更名历史：</td>\s*<td[^>]*>(.*?)</td>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Sina company profile is missing the name-history field")
    value = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).replace("\xa0", " ")
    return [item for item in re.split(r"\s+", value.strip()) if item]


def normalise_name_events(
    raw: pd.DataFrame,
    lifecycle: Any,
    as_of: str | pd.Timestamp,
    *,
    source_tier: str,
    data_source: str,
    source_url: str,
    source_hash: str,
) -> pd.DataFrame:
    rename = {
        "变更日期": "change_date",
        "证券代码": "asset",
        "变更前简称": "old_name",
        "变更后简称": "new_name",
    }
    frame = raw.rename(columns=rename).copy()
    required = {"change_date", "old_name", "new_name"}
    if not required.issubset(frame.columns):
        raise ValueError(f"name events missing columns: {sorted(required.difference(frame.columns))}")
    asset = str(lifecycle.asset).zfill(6)
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        frame = frame[frame["asset"].eq(asset)].copy()
    else:
        frame["asset"] = asset
    start = pd.Timestamp(lifecycle.list_date).normalize()
    as_of_date = pd.Timestamp(as_of).normalize()
    end = min(pd.Timestamp(lifecycle.delist_date).normalize(), as_of_date) if pd.notna(lifecycle.delist_date) else as_of_date
    frame["effective_date"] = pd.to_datetime(frame["change_date"], errors="coerce").dt.normalize()
    frame = frame[frame["effective_date"].between(start, end)].copy()
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame["old_name"] = frame["old_name"].where(frame["old_name"].notna(), "").astype(str).str.strip()
    frame["new_name"] = frame["new_name"].where(frame["new_name"].notna(), "").astype(str).str.strip()
    if frame["new_name"].eq("").any() or frame["effective_date"].duplicated().any():
        raise ValueError(f"{asset} name events contain blank new names or duplicate dates")
    frame["exchange"] = str(lifecycle.exchange)
    frame["old_status"] = frame["old_name"].map(classify_security_name)
    frame["new_status"] = frame["new_name"].map(classify_security_name)
    frame["available_date"] = frame["effective_date"]
    frame["source_tier"] = source_tier
    frame["data_source"] = data_source
    frame["source_vintage"] = f"{data_source}_artifact_sha256:{source_hash}"
    frame["source_url"] = source_url
    frame["source_artifact_sha256"] = source_hash
    return frame[OUTPUT_COLUMNS].sort_values("effective_date").reset_index(drop=True)


def _sse_paths(asset: str) -> tuple[Path, Path, Path]:
    return SSE_DIR / f"{asset}.html.gz", SSE_DIR / f"{asset}.csv", SSE_DIR / f"{asset}.json"


def _valid_sse_cache(asset: str) -> bool:
    html_path, csv_path, meta_path = _sse_paths(asset)
    if not html_path.is_file() or not csv_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return bool(
            meta.get("status") == "success"
            and meta.get("coverage_mode")
            in {"dated_history", "no_name_changes", "undated_non_status_names"}
            and _sha256(html_path) == meta.get("html_sha256")
            and _sha256(csv_path) == meta.get("parsed_sha256")
            and {"change_date", "old_name", "new_name"}.issubset(pd.read_csv(csv_path, nrows=1).columns)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def fetch_sse_asset(session: requests.Session, asset: str) -> dict[str, Any]:
    url = SINA_URL.format(asset=asset)
    response = session.get(url, timeout=20)
    undated_names: list[str] = []
    if response.status_code == 404:
        url = SINA_FALLBACK_URL.format(asset=asset)
        response = session.get(url, timeout=20)
        response.raise_for_status()
        raw_payload = response.content
        undated_names = parse_sina_undated_name_summary(raw_payload)
        unsafe_statuses = {
            "risk_warning",
            "delisting",
            "special_transfer",
            "listing_suspended",
        }
        observed_statuses = {classify_security_name(name) for name in undated_names}
        if observed_statuses.intersection(unsafe_statuses):
            raise ValueError(
                f"Sina fallback has undated execution-relevant names: {sorted(observed_statuses.intersection(unsafe_statuses))}"
            )
        coverage_mode = "no_name_changes" if not undated_names else "undated_non_status_names"
        parsed = pd.DataFrame(columns=["change_date", "old_name", "new_name"])
    else:
        response.raise_for_status()
        raw_payload = response.content
        parsed = parse_sina_name_history(raw_payload)
        coverage_mode = "dated_history"
    html_path, csv_path, meta_path = _sse_paths(asset)
    compressed = gzip.compress(raw_payload, mtime=0)
    _atomic_bytes(compressed, html_path)
    _atomic_csv(parsed, csv_path)
    metadata = {
        "status": "success",
        "asset": asset,
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_url": url,
        "html_path": _relative(html_path),
        "html_sha256": _sha256(html_path),
        "parsed_path": _relative(csv_path),
        "parsed_sha256": _sha256(csv_path),
        "rows": int(len(parsed)),
        "coverage_mode": coverage_mode,
        "undated_names": undated_names,
    }
    _atomic_json(metadata, meta_path)
    return metadata


def fetch_szse_workbook(session: requests.Session, force: bool = False) -> dict[str, Any]:
    latest_path = SZSE_DIR / "latest.json"
    if not force and latest_path.is_file():
        try:
            metadata = json.loads(latest_path.read_text(encoding="utf-8"))
            archive = ROOT / metadata["path"]
            if archive.is_file() and _sha256(archive) == metadata.get("sha256"):
                parse_szse_workbook(archive.read_bytes())
                return metadata
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    response = session.get(
        SZSE_URL,
        timeout=30,
        headers={
            "Referer": "https://www.szse.cn/www/market/stock/changename/index.html",
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/octet-stream;q=0.9,*/*;q=0.8",
        },
    )
    response.raise_for_status()
    payload = response.content
    frame = parse_szse_workbook(payload)
    digest = hashlib.sha256(payload).hexdigest()
    archive = SZSE_DIR / f"name_changes_{digest[:16]}.xlsx"
    if not archive.is_file():
        _atomic_bytes(payload, archive)
    metadata = {
        "status": "success",
        "attempted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_url": SZSE_URL,
        "path": _relative(archive),
        "sha256": digest,
        "rows": int(len(frame)),
    }
    _atomic_json(metadata, latest_path)
    return metadata


def run_collection(
    as_of: str,
    collect_limit: int | None = None,
    sleep_seconds: float = 0.15,
    force_szse: bool = False,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    lifecycles = load_lifecycles(as_of=as_of_date)
    sse_lifecycles = lifecycles[lifecycles["exchange"].eq("SSE")].copy()
    session = _session()
    source_errors: dict[str, str] = {}
    try:
        szse_meta = fetch_szse_workbook(session, force=force_szse)
    except (OSError, ValueError, requests.RequestException) as exc:
        szse_meta = {"status": "failed"}
        source_errors["SZSE"] = f"{type(exc).__name__}: {str(exc)[:400]}"

    pending = [str(asset) for asset in sse_lifecycles["asset"] if not _valid_sse_cache(str(asset))]
    selected = pending[: max(0, collect_limit)] if collect_limit is not None else pending
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "sse_target_assets": int(len(sse_lifecycles)),
        "sse_cached_before": int(len(sse_lifecycles) - len(pending)),
        "sse_pending_before_limit": int(len(pending)),
        "sse_selected_assets": int(len(selected)),
        "assets": {},
    }
    for position, asset in enumerate(selected):
        try:
            status["assets"][asset] = fetch_sse_asset(session, asset)
        except (OSError, ValueError, requests.RequestException) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
            status["assets"][asset] = {"status": "failed", "error": error}
            source_errors[asset] = error
        _atomic_json(status, STATUS_PATH)
        if sleep_seconds > 0 and position + 1 < len(selected):
            time.sleep(sleep_seconds)

    event_frames: list[pd.DataFrame] = []
    inputs: list[dict[str, str]] = [
        {"source_id": "stock_security_master", "path": _relative(MASTER_PATH), "sha256": _sha256(MASTER_PATH)}
    ]
    normalization_errors: dict[str, str] = {}
    if szse_meta.get("status") == "success":
        szse_path = ROOT / str(szse_meta["path"])
        szse_raw = parse_szse_workbook(szse_path.read_bytes())
        for lifecycle in lifecycles[lifecycles["exchange"].eq("SZSE")].itertuples(index=False):
            try:
                event_frames.append(
                    normalise_name_events(
                        szse_raw,
                        lifecycle,
                        as_of_date,
                        source_tier="official_exchange",
                        data_source="szse_name_change_workbook",
                        source_url=SZSE_URL,
                        source_hash=str(szse_meta["sha256"]),
                    )
                )
            except ValueError as exc:
                normalization_errors[str(lifecycle.asset)] = str(exc)
        inputs.append({"source_id": "szse_name_changes", "path": str(szse_meta["path"]), "sha256": str(szse_meta["sha256"])})

    sse_valid_assets: list[str] = []
    sse_coverage_modes: dict[str, int] = {}
    for lifecycle in sse_lifecycles.itertuples(index=False):
        asset = str(lifecycle.asset)
        if not _valid_sse_cache(asset):
            continue
        _, csv_path, meta_path = _sse_paths(asset)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            raw = pd.read_csv(csv_path, dtype=str)
            event_frames.append(
                normalise_name_events(
                    raw,
                    lifecycle,
                    as_of_date,
                    source_tier="secondary_public_page",
                    data_source="sina_company_name_history",
                    source_url=str(meta["source_url"]),
                    source_hash=str(meta["html_sha256"]),
                )
            )
            sse_valid_assets.append(asset)
            coverage_mode = str(meta.get("coverage_mode", "unknown"))
            sse_coverage_modes[coverage_mode] = sse_coverage_modes.get(coverage_mode, 0) + 1
            inputs.extend(
                [
                    {"source_id": f"sse_name_html:{asset}", "path": str(meta["html_path"]), "sha256": str(meta["html_sha256"])},
                    {"source_id": f"sse_name_parsed:{asset}", "path": str(meta["parsed_path"]), "sha256": str(meta["parsed_sha256"])},
                ]
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            normalization_errors[asset] = f"{type(exc).__name__}: {str(exc)[:400]}"

    events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not events.empty:
        events = events.sort_values(["asset", "effective_date"]).reset_index(drop=True)
        if events.duplicated(["asset", "effective_date"]).any():
            conflicts = events[events.duplicated(["asset", "effective_date"], keep=False)]
            for asset in conflicts["asset"].unique():
                normalization_errors[str(asset)] = "duplicate name-change dates across source rows"
            events = events[~events["asset"].isin(conflicts["asset"])].copy()
    bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    events["source_vintage"] = events["source_vintage"].astype(str) + f"|bundle:{bundle_hash}"
    _atomic_csv(events[OUTPUT_COLUMNS], OBSERVATION_OUTPUT)

    sse_complete = len(sse_valid_assets) == len(sse_lifecycles)
    szse_complete = szse_meta.get("status") == "success"
    risk_status_source_complete = bool(sse_complete and szse_complete and not normalization_errors)
    dated_name_event_complete = bool(
        risk_status_source_complete and not sse_coverage_modes.get("undated_non_status_names", 0)
    )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "inputs": inputs,
        "source_vintage": f"stock_name_history_bundle_sha256:{bundle_hash}",
        "output_path": _relative(OBSERVATION_OUTPUT),
        "output_sha256": _sha256(OBSERVATION_OUTPUT),
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "rows": int(len(events)),
        "assets_with_events": int(events["asset"].nunique()) if not events.empty else 0,
        "target_assets": int(len(lifecycles)),
        "sse_target_assets": int(len(sse_lifecycles)),
        "sse_valid_assets": int(len(sse_valid_assets)),
        "sse_asset_coverage": round(len(sse_valid_assets) / len(sse_lifecycles), 6) if len(sse_lifecycles) else 1.0,
        "szse_official_workbook_ready": szse_complete,
        "sse_coverage_modes": sse_coverage_modes,
        "risk_status_source_complete": risk_status_source_complete,
        "dated_name_event_complete": dated_name_event_complete,
        "status_counts": events["new_status"].value_counts().sort_index().to_dict() if not events.empty else {},
        "source_errors": source_errors,
        "normalization_errors": normalization_errors,
        "qualification_status": (
            "OBSERVATION_COMPLETE_REQUIRES_CROSS_SOURCE_VALIDATION"
            if risk_status_source_complete
            else "COLLECTION_IN_PROGRESS"
        ),
        "qualification_blockers": [
            "SSE history is a secondary public page rather than an exchange dataset",
            "effective-dated status intervals have not passed BaoStock and JoinQuant cross-validation",
            "collection-time snapshots are not equivalent to licensed point-in-time name history",
            "ordinary undated renames in fallback pages are excluded from the event output",
        ],
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(
        {
            "sse_valid_assets_after": int(len(sse_valid_assets)),
            "sse_remaining_assets": int(len(sse_lifecycles) - len(sse_valid_assets)),
            "source_errors": source_errors,
        }
    )
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--collect-limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--force-szse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_collection(args.as_of, args.collect_limit, args.sleep_seconds, args.force_szse)
    keys = (
        "qualification_status",
        "rows",
        "assets_with_events",
        "target_assets",
        "sse_valid_assets",
        "sse_target_assets",
        "sse_asset_coverage",
        "szse_official_workbook_ready",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
