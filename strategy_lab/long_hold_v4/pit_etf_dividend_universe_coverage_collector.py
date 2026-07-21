"""Collect and audit official cash-distribution announcements for the full ETF universe.

This collector is independent of the price-derived dividend discovery queue. Its
outputs remain observation-only until a separate validator reconciles every
additional official event and promotes a new formal PIT event table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pdfminer.pdfparser import PDFSyntaxError

from . import pit_etf_dividend_announcement_collector as official


ROOT = Path(__file__).resolve().parents[2]
ETF_MASTER_PATH = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
ETF_MASTER_MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_master_builder_latest.json"
VALIDATED_CANDIDATE_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "etf_dividend_registry_candidates.csv"
)
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_dividend_universe_coverage"
STATUS_PATH = RAW_DIR / "collection_status.json"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
ANNOUNCEMENT_PATH = OBSERVATION_DIR / "etf_dividend_universe_official_announcements.csv"
DOCUMENT_INDEX_PATH = OBSERVATION_DIR / "etf_dividend_universe_document_index.csv"
EVENT_INVENTORY_PATH = OBSERVATION_DIR / "etf_dividend_universe_event_inventory.csv"
COMPLETE_CANDIDATE_PATH = OBSERVATION_DIR / "etf_dividend_universe_complete_candidates.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_dividend_universe_review_queue.csv"
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_dividend_universe_coverage_latest.json"
REPORT_PATH = ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_dividend_universe_coverage" / "report.json"

DOCUMENT_COLUMNS = [
    "asset",
    "announcement_date",
    "announcement_title",
    "source_url",
    "source_type",
    "pdf_status",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "cash_candidates_json",
    "record_date_candidates_json",
    "ex_date_candidates_json",
    "pay_date_candidates_json",
    "error",
]
EVENT_COLUMNS = [
    "asset",
    "announcement_date",
    "record_date",
    "ex_date",
    "pay_date",
    "cash_per_share",
    "parse_status",
    "candidate_combination_count",
    "best_score_tie_count",
    "announcement_title",
    "source_url",
    "source_type",
    "pdf_path",
    "pdf_sha256",
    "text_path",
    "text_sha256",
    "historical_backtest_allowed",
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
    _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    temporary.replace(path)


def load_targets(
    *,
    assets: list[str] | None = None,
    exchange: str = "all",
) -> pd.DataFrame:
    master_manifest = json.loads(ETF_MASTER_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        master_manifest.get("historical_backtest_allowed") is not True
        or str(master_manifest.get("output_path", "")).replace("\\", "/") != _relative(ETF_MASTER_PATH)
        or str(master_manifest.get("output_sha256", "")) != _sha256(ETF_MASTER_PATH)
    ):
        raise ValueError("ETF master lineage does not authenticate the full-universe target list")
    master = pd.read_csv(ETF_MASTER_PATH, dtype={"asset": str})
    required = {"asset", "asset_name", "exchange", "list_status", "event_type", "list_date", "delist_date"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"ETF master misses universe columns: {missing}")
    targets = master[
        master["list_status"].eq("listed") & master["event_type"].eq("listing")
    ][["asset", "asset_name", "exchange", "list_date"]].copy()
    delist = master[
        master["list_status"].eq("delisted") & master["event_type"].eq("delisting")
    ][["asset", "delist_date"]].copy()
    targets = targets.merge(delist, on="asset", how="left", validate="one_to_one")
    targets["asset"] = targets["asset"].astype(str).str.zfill(6)
    targets["list_date"] = pd.to_datetime(targets["list_date"], errors="coerce").dt.normalize()
    targets["delist_date"] = pd.to_datetime(targets["delist_date"], errors="coerce").dt.normalize()
    if targets["list_date"].isna().any():
        raise ValueError("ETF master universe has invalid listing dates")
    if targets["asset"].duplicated().any() or not set(targets["exchange"]).issubset({"SSE", "SZSE"}):
        raise ValueError("ETF master universe has duplicate assets or unsupported exchanges")
    if exchange != "all":
        targets = targets[targets["exchange"].eq(exchange)]
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        targets = targets[targets["asset"].isin(requested)]
    if targets.empty:
        raise ValueError("ETF universe coverage selected no targets")
    return targets.sort_values(["exchange", "asset"]).reset_index(drop=True)


def _select_date_record(
    candidates: list[dict[str, Any]],
    target: pd.Timestamp,
    *,
    on_or_after: bool,
) -> dict[str, Any] | None:
    eligible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for candidate in candidates:
        value = pd.to_datetime(candidate.get("date"), errors="coerce")
        if pd.isna(value):
            continue
        date = pd.Timestamp(value).normalize()
        if on_or_after and date < target:
            continue
        if not on_or_after and date > target:
            continue
        score = (int(candidate.get("rank", 99)), abs((date - target).days), date)
        eligible.append((score, {**candidate, "parsed_date": date}))
    return None if not eligible else min(eligible, key=lambda item: item[0])[1]


def select_direct_official_event(
    parsed: dict[str, list[dict[str, Any]]],
    announcement_date: str | pd.Timestamp,
) -> dict[str, Any]:
    announcement = pd.to_datetime(announcement_date, errors="coerce")
    if pd.isna(announcement):
        return {"parse_status": "invalid_announcement_date", "candidate_combination_count": 0, "best_score_tie_count": 0}
    announcement = pd.Timestamp(announcement).normalize()
    combinations: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for ex_candidate in parsed.get("ex_date_candidates", []):
        ex_value = pd.to_datetime(ex_candidate.get("date"), errors="coerce")
        if pd.isna(ex_value):
            continue
        ex_date = pd.Timestamp(ex_value).normalize()
        record = _select_date_record(parsed.get("record_date_candidates", []), ex_date, on_or_after=False)
        pay = _select_date_record(parsed.get("pay_date_candidates", []), ex_date, on_or_after=True)
        if record is None or pay is None:
            continue
        record_date = pd.Timestamp(record["parsed_date"])
        pay_date = pd.Timestamp(pay["parsed_date"])
        if not announcement <= record_date <= ex_date <= pay_date:
            continue
        for cash in parsed.get("cash_candidates", []):
            cash_value = pd.to_numeric(cash.get("cash_per_share"), errors="coerce")
            if pd.isna(cash_value) or float(cash_value) <= 0:
                continue
            score = (
                int(cash.get("rank", 99)),
                int(ex_candidate.get("rank", 99)),
                int(record.get("rank", 99)),
                int(pay.get("rank", 99)),
                abs((ex_date - announcement).days),
                abs((ex_date - record_date).days),
                abs((pay_date - ex_date).days),
            )
            combinations.append(
                (
                    score,
                    {
                        "announcement_date": announcement,
                        "record_date": record_date,
                        "ex_date": ex_date,
                        "pay_date": pay_date,
                        "cash_per_share": float(cash_value),
                    },
                )
            )
    if not combinations:
        missing = []
        if not parsed.get("cash_candidates"):
            missing.append("cash")
        if not parsed.get("record_date_candidates"):
            missing.append("record_date")
        if not parsed.get("ex_date_candidates"):
            missing.append("ex_date")
        if not parsed.get("pay_date_candidates"):
            missing.append("pay_date")
        suffix = "_".join(missing) if missing else "chronology"
        return {
            "parse_status": f"incomplete_{suffix}",
            "candidate_combination_count": 0,
            "best_score_tie_count": 0,
        }
    combinations.sort(key=lambda item: (*item[0], item[1]["cash_per_share"]))
    best_score = combinations[0][0]
    best = [item[1] for item in combinations if item[0] == best_score]
    unique = {
        (
            item["record_date"],
            item["ex_date"],
            item["pay_date"],
            round(float(item["cash_per_share"]), 12),
        )
        for item in best
    }
    selected = combinations[0][1]
    return {
        **selected,
        "parse_status": "complete_unique_official_event" if len(unique) == 1 else "ambiguous_best_official_event",
        "candidate_combination_count": len(combinations),
        "best_score_tie_count": len(unique),
    }


def build_document_index(
    announcements: pd.DataFrame,
    sessions: dict[str, requests.Session],
    *,
    status: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    failures: dict[str, str] = {}
    for position, announcement in enumerate(announcements.itertuples(index=False), start=1):
        source_url = str(announcement.source_url)
        base = {
            "asset": str(announcement.asset).zfill(6),
            "announcement_date": pd.Timestamp(announcement.announcement_date).normalize(),
            "announcement_title": str(announcement.announcement_title),
            "source_url": source_url,
            "source_type": str(announcement.source_type),
            "pdf_status": "failed",
            "pdf_path": "",
            "pdf_sha256": "",
            "text_path": "",
            "text_sha256": "",
            "cash_candidates_json": "[]",
            "record_date_candidates_json": "[]",
            "ex_date_candidates_json": "[]",
            "pay_date_candidates_json": "[]",
            "error": "",
        }
        try:
            session = sessions["SSE"] if str(announcement.exchange) == "SSE" else sessions["SZSE"]
            document = official.download_document(session, source_url)
            text_meta = official.extract_document_text(document)
            text = (ROOT / str(text_meta["path"])).read_text(encoding="utf-8")
            parsed = official.parse_document_text(text)
            base.update(
                {
                    "pdf_status": "success",
                    "pdf_path": str(document["path"]),
                    "pdf_sha256": str(document["sha256"]),
                    "text_path": str(text_meta["path"]),
                    "text_sha256": str(text_meta["sha256"]),
                    "cash_candidates_json": json.dumps(parsed["cash_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "record_date_candidates_json": json.dumps(parsed["record_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "ex_date_candidates_json": json.dumps(parsed["ex_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                    "pay_date_candidates_json": json.dumps(parsed["pay_date_candidates"], ensure_ascii=False, separators=(",", ":")),
                }
            )
            token = hashlib.sha256(source_url.encode()).hexdigest()[:16]
            inputs.extend(
                [
                    {"source_id": f"universe_etf_dividend_pdf:{token}", "path": str(document["path"]), "sha256": str(document["sha256"])},
                    {"source_id": f"universe_etf_dividend_text:{token}", "path": str(text_meta["path"]), "sha256": str(text_meta["sha256"])},
                ]
            )
        except (OSError, ValueError, requests.RequestException, PDFSyntaxError, json.JSONDecodeError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:400]}"
            base["error"] = message
            failures[source_url] = message
        rows.append(base)
        if position % 25 == 0 or position == len(announcements):
            status["documents_processed"] = position
            status["document_failures"] = failures
            _atomic_json(status, STATUS_PATH)
    return pd.DataFrame(rows, columns=DOCUMENT_COLUMNS), inputs


def build_event_inventory(document_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in document_index.itertuples(index=False):
        base = {
            "asset": str(document.asset).zfill(6),
            "announcement_date": pd.Timestamp(document.announcement_date).normalize(),
            "record_date": pd.NaT,
            "ex_date": pd.NaT,
            "pay_date": pd.NaT,
            "cash_per_share": pd.NA,
            "parse_status": "document_unavailable",
            "candidate_combination_count": 0,
            "best_score_tie_count": 0,
            "announcement_title": str(document.announcement_title),
            "source_url": str(document.source_url),
            "source_type": str(document.source_type),
            "pdf_path": str(document.pdf_path),
            "pdf_sha256": str(document.pdf_sha256),
            "text_path": str(document.text_path),
            "text_sha256": str(document.text_sha256),
            "historical_backtest_allowed": False,
        }
        if str(document.pdf_status) == "success":
            parsed = {
                "cash_candidates": json.loads(str(document.cash_candidates_json)),
                "record_date_candidates": json.loads(str(document.record_date_candidates_json)),
                "ex_date_candidates": json.loads(str(document.ex_date_candidates_json)),
                "pay_date_candidates": json.loads(str(document.pay_date_candidates_json)),
            }
            base.update(select_direct_official_event(parsed, document.announcement_date))
        rows.append(base)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)


def _event_keys(frame: pd.DataFrame) -> set[tuple[str, str, float]]:
    if frame.empty:
        return set()
    return {
        (str(row.asset).zfill(6), str(pd.Timestamp(row.ex_date).date()), round(float(row.cash_per_share), 10))
        for row in frame.itertuples(index=False)
        if pd.notna(row.ex_date) and pd.notna(row.cash_per_share)
    }


def recover_delisted_cninfo_identity(
    session: requests.Session,
    *,
    asset: str,
    asset_name: str,
    list_date: pd.Timestamp,
    delist_date: pd.Timestamp,
    as_of_date: pd.Timestamp,
    maximum_pages: int = 100,
) -> dict[str, Any]:
    if pd.isna(delist_date):
        raise ValueError(f"CNInfo fallback is restricted to delisted ETFs: {asset}")
    end_date = min(pd.Timestamp(delist_date).normalize(), as_of_date)
    variants = [str(asset_name).strip()]
    if variants[0].endswith("基金"):
        variants.append(variants[0][:-2])
    variants.extend(["终止上市", "基金合同终止"])
    variants = list(dict.fromkeys(value for value in variants if value))
    evidence: list[dict[str, Any]] = []
    pages_scanned = 0
    for searchkey in variants:
        page_number = 1
        page_count = 1
        while page_number <= min(page_count, maximum_pages):
            params = official.cninfo_api._query_params(asset, "", searchkey, as_of_date, page_number)
            params.update(
                {
                    "stock": "",
                    "searchkey": searchkey,
                    "seDate": f"{pd.Timestamp(list_date).date().isoformat()}~{end_date.date().isoformat()}",
                    "sortName": "time",
                    "sortType": "desc",
                }
            )
            response = session.post(official.cninfo_api.QUERY_URL, data=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo fallback identity search returned invalid rows for {asset}")
            pages_scanned += 1
            for row in announcements:
                if str(row.get("secCode", "")).zfill(6) != asset or not str(row.get("orgId", "")).strip():
                    continue
                evidence.append(
                    {
                        "searchkey": searchkey,
                        "page_number": page_number,
                        "sec_code": asset,
                        "org_id": str(row["orgId"]),
                        "announcement_title": official.cninfo_api.clean_announcement_title(
                            row.get("announcementTitle", "")
                        ),
                        "announcement_url": str(row.get("adjunctUrl", "")),
                    }
                )
            if evidence:
                break
            page_count = max(1, int(payload.get("totalpages") or 1))
            page_number += 1
        if evidence:
            break
    org_ids = sorted({str(item["org_id"]) for item in evidence})
    if len(org_ids) != 1:
        raise ValueError(
            f"CNInfo delisted-fund identity fallback is ambiguous or missing for {asset}: {len(org_ids)} org ids"
        )
    return {
        "code": asset,
        "org_id": org_ids[0],
        "fund_name": str(asset_name),
        "category": "fund",
        "delisted": True,
        "resolution": "fulltext_name_search_exact_sec_code",
        "pages_scanned": pages_scanned,
        "evidence": evidence[:5],
    }


def fetch_cninfo_query_with_delisted_fallback(
    session: requests.Session,
    target: Any,
    as_of_date: pd.Timestamp,
) -> dict[str, Any]:
    asset = str(target.asset).zfill(6)
    try:
        return official._fetch_cninfo_query(session, asset, as_of_date)
    except ValueError as exc:
        if "identity is ambiguous or missing" not in str(exc):
            raise
    identity = recover_delisted_cninfo_identity(
        session,
        asset=asset,
        asset_name=str(target.asset_name),
        list_date=pd.Timestamp(target.list_date),
        delist_date=pd.Timestamp(target.delist_date),
        as_of_date=as_of_date,
    )
    responses: list[dict[str, Any]] = []
    for keyword in official.KEYWORDS:
        rows: list[dict[str, Any]] = []
        page_number = 1
        page_count = 1
        while page_number <= page_count:
            response = session.post(
                official.cninfo_api.QUERY_URL,
                data=official.cninfo_api._query_params(
                    asset,
                    str(identity["org_id"]),
                    keyword,
                    as_of_date,
                    page_number,
                ),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            announcements = payload.get("announcements") or []
            if not isinstance(announcements, list):
                raise ValueError(f"CNInfo dividend query returned invalid rows for {asset}/{keyword}")
            rows.extend(announcements)
            page_count = max(1, int(payload.get("totalpages") or 1))
            if page_count > 100:
                raise ValueError(f"CNInfo dividend query has implausible page count for {asset}/{keyword}")
            page_number += 1
        responses.append({"keyword": keyword, "page_count": page_count, "rows": rows})
    return {
        "asset": asset,
        "exchange": "SZSE",
        "as_of_date": as_of_date.date().isoformat(),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "query_keywords": list(official.KEYWORDS),
        "identity": identity,
        "responses": responses,
    }


def run_collection(
    as_of: str,
    *,
    assets: list[str] | None = None,
    exchange: str = "all",
    max_assets: int | None = None,
    sleep_seconds: float = 0.1,
    max_consecutive_failures: int = 5,
) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    targets = load_targets(assets=assets, exchange=exchange)
    pending = [
        row
        for row in targets.itertuples(index=False)
        if not official._valid_query_cache(str(row.exchange), str(row.asset), as_of_date)
    ]
    selected = pending[:max_assets] if max_assets is not None else pending
    status: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "target_assets": int(len(targets)),
        "pending_assets_at_start": len(pending),
        "selected_uncached_assets": len(selected),
        "assets": {},
    }
    sessions = {
        "SSE": official._session("https://www.sse.com.cn/disclosure/fund/announcement/index.shtml"),
        "SZSE": official._session("https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"),
    }
    failures = 0
    for position, target in enumerate(selected, start=1):
        target_exchange = str(target.exchange)
        asset = str(target.asset)
        try:
            artifact = (
                official._fetch_sse_query(sessions["SSE"], asset, as_of_date)
                if target_exchange == "SSE"
                else fetch_cninfo_query_with_delisted_fallback(sessions["SZSE"], target, as_of_date)
            )
            status["assets"][asset] = official._save_query_artifact(artifact)
            failures = 0
        except (OSError, ValueError, requests.RequestException) as exc:
            status["assets"][asset] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            failures += 1
        status["queries_processed"] = position
        _atomic_json(status, STATUS_PATH)
        if failures >= max_consecutive_failures:
            status["circuit_breaker"] = f"stopped_after_{failures}_consecutive_failures"
            break
        if sleep_seconds > 0 and position < len(selected):
            time.sleep(sleep_seconds)

    artifacts: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = [
        {"source_id": "etf_security_master", "path": _relative(ETF_MASTER_PATH), "sha256": _sha256(ETF_MASTER_PATH)},
        {"source_id": "etf_security_master_manifest", "path": _relative(ETF_MASTER_MANIFEST_PATH), "sha256": _sha256(ETF_MASTER_MANIFEST_PATH)},
    ]
    complete_queries = 0
    for row in targets.itertuples(index=False):
        target_exchange, asset = str(row.exchange), str(row.asset)
        if not official._valid_query_cache(target_exchange, asset, as_of_date):
            continue
        complete_queries += 1
        data_path, meta_path = official._query_paths(target_exchange, asset)
        artifacts.append(json.loads(data_path.read_text(encoding="utf-8")))
        inputs.extend(
            [
                {"source_id": f"universe_etf_dividend_query:{target_exchange}:{asset}", "path": _relative(data_path), "sha256": _sha256(data_path)},
                {"source_id": f"universe_etf_dividend_query_meta:{target_exchange}:{asset}", "path": _relative(meta_path), "sha256": _sha256(meta_path)},
            ]
        )
    asset_names = targets.set_index("asset")["asset_name"].astype(str).to_dict()
    announcements = official.parse_query_artifacts(artifacts, asset_names)
    query_bundle_hash = hashlib.sha256(
        "|".join(sorted(f"{item['source_id']}:{item['sha256']}" for item in inputs)).encode()
    ).hexdigest()
    if not announcements.empty:
        announcements["data_source"] = "SSE and CNInfo official full-universe ETF distribution announcement queries"
        announcements["source_vintage"] = f"official_etf_dividend_universe_query_bundle_sha256:{query_bundle_hash}"
    announcements = announcements.reindex(columns=official.ANNOUNCEMENT_COLUMNS)
    document_index, document_inputs = build_document_index(announcements, sessions, status=status)
    inputs.extend(document_inputs)
    inventory = build_event_inventory(document_index)
    complete = inventory[inventory["parse_status"].eq("complete_unique_official_event")].copy()

    validated = pd.read_csv(VALIDATED_CANDIDATE_PATH, dtype={"asset": str}) if VALIDATED_CANDIDATE_PATH.is_file() else pd.DataFrame()
    direct_keys = _event_keys(complete)
    validated_keys = _event_keys(validated)
    extra_keys = direct_keys.difference(validated_keys)
    missing_keys = validated_keys.difference(direct_keys)
    validated_source_urls = set(validated.get("source_url", pd.Series(dtype=str)).dropna().astype(str))
    successful_source_urls = set(
        document_index.loc[document_index["pdf_status"].eq("success"), "source_url"].dropna().astype(str)
    )
    missing_validated_documents = validated_source_urls.difference(successful_source_urls)
    ambiguity_resolved_by_validated_crosscheck = int(
        inventory[
            inventory["parse_status"].eq("ambiguous_best_official_event")
            & inventory["source_url"].isin(validated_source_urls)
        ]["source_url"].nunique()
    )
    review_mask = ~inventory["parse_status"].eq("complete_unique_official_event")
    review = inventory[review_mask].copy()
    if extra_keys:
        complete_keys = [
            (str(row.asset).zfill(6), str(pd.Timestamp(row.ex_date).date()), round(float(row.cash_per_share), 10))
            for row in complete.itertuples(index=False)
        ]
        extra_positions = [position for position, key in enumerate(complete_keys) if key in extra_keys]
        review = pd.concat([review, complete.iloc[extra_positions]], ignore_index=True)

    _atomic_csv(announcements, ANNOUNCEMENT_PATH)
    _atomic_csv(document_index, DOCUMENT_INDEX_PATH)
    _atomic_csv(inventory, EVENT_INVENTORY_PATH)
    _atomic_csv(complete, COMPLETE_CANDIDATE_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)
    document_failures = int(document_index["pdf_status"].ne("success").sum())
    ambiguous = int(inventory["parse_status"].eq("ambiguous_best_official_event").sum())
    qualification = (
        "FULL_UNIVERSE_QUERIED_OFFICIAL_EVENTS_REQUIRE_VALIDATION"
        if complete_queries == len(targets) and document_failures == 0 and not missing_validated_documents
        else "PARTIAL_OR_REVIEW_REQUIRED"
    )
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date.date().isoformat(),
        "qualification_status": qualification,
        "target_assets": int(len(targets)),
        "query_complete_assets": int(complete_queries),
        "announcement_candidates": int(len(announcements)),
        "documents": int(len(document_index)),
        "document_failures": document_failures,
        "complete_unique_official_events": int(len(complete)),
        "complete_unique_assets": int(complete["asset"].nunique()) if not complete.empty else 0,
        "ambiguous_event_documents": ambiguous,
        "validated_event_keys": len(validated_keys),
        "validated_source_documents": len(validated_source_urls),
        "validated_source_documents_missing": len(missing_validated_documents),
        "validated_keys_missing_from_direct_parse": len(missing_keys),
        "direct_ambiguities_resolved_by_validated_crosscheck": ambiguity_resolved_by_validated_crosscheck,
        "direct_event_keys_not_in_validated_table": len(extra_keys),
        "earliest_complete_announcement_date": (
            str(pd.Timestamp(complete["announcement_date"].min()).date()) if not complete.empty else None
        ),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "Full-universe official discovery and parser audit only; additional events require independent validation before formal PIT use.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [ANNOUNCEMENT_PATH, DOCUMENT_INDEX_PATH, EVENT_INVENTORY_PATH, COMPLETE_CANDIDATE_PATH, REVIEW_QUEUE_PATH, REPORT_PATH]
    manifest = {
        "schema_version": 1,
        **report,
        "inputs": inputs,
        "outputs": [{"path": _relative(path), "sha256": _sha256(path), "rows": int(len(pd.read_csv(path))) if path.suffix == ".csv" else None} for path in outputs],
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"role": "official_dividend_parser", "path": _relative(Path(official.__file__).resolve()), "sha256": _sha256(Path(official.__file__).resolve())}
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    status.update(report)
    _atomic_json(status, STATUS_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--asset", action="append")
    parser.add_argument("--exchange", choices=("all", "SSE", "SZSE"), default="all")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_collection(
        args.as_of,
        assets=args.asset,
        exchange=args.exchange,
        max_assets=args.max_assets,
        sleep_seconds=args.sleep_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    keys = (
        "qualification_status",
        "target_assets",
        "query_complete_assets",
        "announcement_candidates",
        "complete_unique_official_events",
        "validated_keys_missing_from_direct_parse",
        "direct_event_keys_not_in_validated_table",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: manifest[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
