"""Safely probe and pre-cache reviewed CSIndex ETF benchmark mappings.

The probe never edits the activation registry. A successful result is evidence
for a later manual activation review; provider blocking fails closed and defers
the remaining requests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from .etf_index_registry import INDEX_REGISTRY_PATH, load_index_registry


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data_raw" / "long_hold_v4"
DEFAULT_OUTPUT = RAW_ROOT / "pit_history" / "observations" / "etf_index_registry_probe.csv"
DEFAULT_MANIFEST = RAW_ROOT / "manifests" / "etf_index_registry_probe_latest.json"
IMMUTABLE_RUN_DIR = RAW_ROOT / "manifests" / "etf_index_registry_probe_runs"
PERF_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
INDICATOR_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/indicator/"
    "{code}indicator.xls"
)
PENDING_STATUSES = {"verified_pending_history_cache", "verified_pending_history_probe"}


class ProviderBlockedError(RuntimeError):
    """Raised when the upstream provider explicitly rate-limits or blocks."""


class HistoryInsufficientError(ValueError):
    """Raised with audit metrics when an identity is valid but history is too short."""

    def __init__(self, message: str, metrics: dict[str, Any]):
        super().__init__(message)
        self.metrics = metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalise_name(value: Any) -> str:
    return "".join(str(value).split())


def _raise_for_provider(response: requests.Response, label: str) -> None:
    if response.status_code in {403, 405, 429}:
        raise ProviderBlockedError(f"CSIndex provider blocked {label}: HTTP {response.status_code}")
    response.raise_for_status()


def _fetch_history(session: requests.Session, code: str, as_of: pd.Timestamp) -> pd.DataFrame:
    response = session.get(
        PERF_URL,
        params={"indexCode": code, "startDate": "20000101", "endDate": as_of.strftime("%Y%m%d")},
        timeout=60,
    )
    _raise_for_provider(response, f"history {code}")
    try:
        rows = response.json().get("data") or []
    except ValueError as exc:
        raise ValueError(f"CSIndex history returned non-JSON content for {code}") from exc
    if not rows:
        raise ValueError(f"CSIndex history is empty for {code}")
    out = pd.DataFrame(rows).rename(
        columns={
            "tradeDate": "date",
            "indexCode": "index_code",
            "indexNameCnAll": "index_name",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tradingVol": "volume",
            "tradingValue": "amount",
            "peg": "pe_ttm",
        }
    )
    required = {"date", "index_code", "index_name", "close", "pe_ttm"}
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"CSIndex history is missing columns for {code}: {missing}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount", "pe_ttm"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[out["date"].le(as_of)].sort_values("date").reset_index(drop=True)
    out["data_source"] = "csindex.index-perf"
    out["fetched_at"] = _now()
    return out


def _fetch_static_bytes(url: str, label: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310 - reviewed HTTPS source
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 405, 429}:
                raise ProviderBlockedError(f"CSIndex provider blocked {label}: HTTP {exc.code}") from exc
            raise
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == 1:
                raise RuntimeError(f"CSIndex static source failed for {label}: {exc}") from exc
            time.sleep(2.0)
    raise AssertionError("unreachable static-source retry state")


def _fetch_valuation(code: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, bytes]:
    payload = _fetch_static_bytes(INDICATOR_URL.format(code=code), f"valuation {code}")
    raw = pd.read_excel(io.BytesIO(payload))
    if raw.shape[1] != 10:
        raise ValueError(f"CSIndex valuation has an unexpected schema for {code}")
    raw.columns = [
        "date",
        "index_code",
        "index_name",
        "index_short_name",
        "index_name_en",
        "index_short_name_en",
        "pe_total_shares",
        "pe_calculation_shares",
        "dividend_yield_total_shares_pct",
        "dividend_yield_calculation_shares_pct",
    ]
    raw["date"] = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in [
        "pe_total_shares",
        "pe_calculation_shares",
        "dividend_yield_total_shares_pct",
        "dividend_yield_calculation_shares_pct",
    ]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    out = raw[raw["date"].le(as_of)].sort_values("date").reset_index(drop=True)
    out["data_source"] = "csindex.indicator.xls"
    out["fetched_at"] = _now()
    return out, payload


def validate_history_identity(
    frame: pd.DataFrame,
    expected_code: str,
    expected_name: str,
    as_of: str | pd.Timestamp,
    require_pe_history: bool,
) -> dict[str, Any]:
    required = {"date", "index_code", "index_name", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"index history validation is missing columns: {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    cutoff = pd.Timestamp(as_of).normalize()
    data = data[data["date"].le(cutoff)].sort_values("date")
    if data.empty or data["date"].isna().any() or data["date"].duplicated().any() or (data["close"] <= 0).any():
        raise ValueError(f"index history is invalid for {expected_code}")
    codes = set(data["index_code"].astype(str).str.strip())
    names = {_normalise_name(value) for value in data["index_name"]}
    if codes != {expected_code} or names != {_normalise_name(expected_name)}:
        raise ValueError(f"index history identity mismatch for {expected_code}: codes={codes};names={names}")
    span_years = (data["date"].max() - data["date"].min()).days / 365.25
    metrics: dict[str, Any] = {
        "rows": int(len(data)),
        "coverage_start": str(data["date"].min().date()),
        "coverage_end": str(data["date"].max().date()),
        "valid_pe_rows": 0,
        "valid_pe_start": "",
        "valid_pe_end": "",
    }
    if len(data) < 1000 or span_years < 4.5:
        raise HistoryInsufficientError(f"index history is shorter than five years for {expected_code}", metrics)
    if require_pe_history:
        if "pe_ttm" not in data.columns:
            raise ValueError(f"price index history lacks PE for {expected_code}")
        valid_pe = data[pd.to_numeric(data["pe_ttm"], errors="coerce").gt(0)]
        metrics["valid_pe_rows"] = int(len(valid_pe))
        if not valid_pe.empty:
            metrics["valid_pe_start"] = str(valid_pe["date"].min().date())
            metrics["valid_pe_end"] = str(valid_pe["date"].max().date())
        if metrics["valid_pe_rows"] < 1000:
            raise HistoryInsufficientError(f"price index PE history is insufficient for {expected_code}", metrics)
    return metrics


def validate_valuation_identity(
    frame: pd.DataFrame,
    expected_code: str,
    expected_name: str,
    as_of: str | pd.Timestamp,
) -> dict[str, Any]:
    required = {"date", "index_code", "index_name", "dividend_yield_calculation_shares_pct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"index valuation validation is missing columns: {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[data["date"].le(pd.Timestamp(as_of).normalize())].sort_values("date")
    codes = data["index_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    if expected_code.isdigit() and len(expected_code) == 6:
        codes = codes.str.zfill(6)
    names = {_normalise_name(value) for value in data["index_name"]}
    yields = pd.to_numeric(data["dividend_yield_calculation_shares_pct"], errors="coerce")
    if data.empty or set(codes) != {expected_code} or names != {_normalise_name(expected_name)} or not yields.notna().any():
        raise ValueError(f"index valuation identity mismatch for {expected_code}")
    return {"rows": int(len(data)), "coverage_end": str(data["date"].max().date())}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_probe_run_id(
    created_at: str,
    as_of: str | pd.Timestamp,
    price_codes: list[str],
    output_sha256: str,
) -> str:
    timestamp = pd.Timestamp(created_at).strftime("%Y%m%dT%H%M%S%z")
    code_slug = "-".join("".join(char for char in str(code) if char.isalnum()) for code in price_codes)
    if not code_slug:
        code_slug = "none"
    return f"{pd.Timestamp(as_of).strftime('%Y%m%d')}_{timestamp}_{code_slug}_{output_sha256[:12]}"


def archive_probe_artifacts(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    latest_manifest_path: Path,
    results: list[dict[str, Any]] | None = None,
    run_dir: Path = IMMUTABLE_RUN_DIR,
) -> dict[str, Any]:
    """Write an immutable result/manifest pair and refresh the latest pointer."""

    selected_codes = [str(code) for code in manifest.get("selected_price_codes", [])]
    if not selected_codes and "price_code" in frame.columns:
        selected_codes = frame["price_code"].dropna().astype(str).drop_duplicates().tolist()
    output_sha = str(manifest["output_sha256"])
    run_id = build_probe_run_id(
        str(manifest["created_at"]), str(manifest["as_of_date"]), selected_codes, output_sha
    )
    immutable_result = run_dir / f"{run_id}.csv"
    immutable_manifest = run_dir / f"{run_id}.json"

    if immutable_result.exists():
        if _sha256(immutable_result) != output_sha:
            raise ValueError(f"immutable probe result collision: {immutable_result}")
    else:
        _atomic_csv(frame, immutable_result)
        if _sha256(immutable_result) != output_sha:
            raise ValueError("immutable probe result differs from latest output")

    payload = dict(manifest)
    payload["selected_price_codes"] = selected_codes
    if results is None and isinstance(manifest.get("results"), list):
        results = list(manifest["results"])
    if results is None:
        clean = frame.astype(object).where(pd.notna(frame), None)
        results = clean.to_dict(orient="records")
    payload["results"] = results
    payload["immutable_result_path"] = _relative(immutable_result)
    payload["immutable_result_sha256"] = _sha256(immutable_result)
    payload["immutable_run_manifest_path"] = _relative(immutable_manifest)

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if immutable_manifest.exists():
        if immutable_manifest.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"immutable probe manifest collision: {immutable_manifest}")
    else:
        _atomic_json(payload, immutable_manifest)
    _atomic_json(payload, latest_manifest_path)
    return payload


def archive_latest_probe(
    output_path: Path,
    manifest_path: Path,
    run_dir: Path = IMMUTABLE_RUN_DIR,
) -> dict[str, Any]:
    if not output_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("latest ETF index probe output/manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(output_path) != manifest.get("output_sha256"):
        raise ValueError("latest ETF index probe output hash mismatch")
    frame = pd.read_csv(output_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return archive_probe_artifacts(frame, manifest, manifest_path, run_dir=run_dir)


def _cache_paths(raw_dir: Path, item: dict[str, Any]) -> dict[str, Path]:
    return {
        "price": raw_dir / "index_price" / f"{item['price_code']}.csv",
        "total_return": raw_dir / "index_total_return" / f"{item['total_return_code']}.csv",
        "valuation": raw_dir / "index_valuation" / f"{item['price_code']}.csv",
    }


def probe_mapping(
    item: dict[str, Any],
    as_of: pd.Timestamp,
    raw_dir: Path,
    evidence_dir: Path,
    performance_session: requests.Session,
) -> dict[str, Any]:
    valuation, valuation_payload = _fetch_valuation(str(item["price_code"]), as_of)
    valuation_metrics = validate_valuation_identity(
        valuation, str(item["price_code"]), str(item["tracking_index_name"]), as_of
    )

    factsheet_urls = [url for url in item.get("evidence_urls", []) if str(url).endswith("factsheet.pdf")]
    if not factsheet_urls:
        raise ValueError(f"mapping has no official factsheet: {item['tracking_index_name']}")
    factsheet_payload = _fetch_static_bytes(factsheet_urls[0], f"factsheet {item['price_code']}")
    if not factsheet_payload.startswith(b"%PDF"):
        raise ValueError(f"factsheet is not a PDF for {item['tracking_index_name']}")
    price = _fetch_history(performance_session, str(item["price_code"]), as_of)
    total_return = _fetch_history(performance_session, str(item["total_return_code"]), as_of)
    total_metrics = validate_history_identity(
        total_return, str(item["total_return_code"]), str(item["total_return_name"]), as_of, False
    )
    price_metrics = validate_history_identity(price, str(item["price_code"]), str(item["tracking_index_name"]), as_of, True)

    paths = _cache_paths(raw_dir, item)
    for role, frame in (("price", price), ("total_return", total_return), ("valuation", valuation)):
        _atomic_csv(frame, paths[role])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    factsheet_hash = hashlib.sha256(factsheet_payload).hexdigest()
    factsheet_path = evidence_dir / f"{item['price_code']}_factsheet_{factsheet_hash[:16]}.pdf"
    if not factsheet_path.exists():
        factsheet_path.write_bytes(factsheet_payload)
    valuation_hash = hashlib.sha256(valuation_payload).hexdigest()
    valuation_raw_path = evidence_dir / f"{item['price_code']}_indicator_{valuation_hash[:16]}.xls"
    if not valuation_raw_path.exists():
        valuation_raw_path.write_bytes(valuation_payload)
    return {
        "probe_status": "passed_cache_ready_for_manual_activation_review",
        "price_rows": price_metrics["rows"],
        "total_return_rows": total_metrics["rows"],
        "valuation_rows": valuation_metrics["rows"],
        "coverage_start": price_metrics["coverage_start"],
        "coverage_end": price_metrics["coverage_end"],
        "cache_paths": {role: _relative(path) for role, path in paths.items()},
        "evidence_paths": [_relative(factsheet_path), _relative(valuation_raw_path)],
    }


ProbeCallable = Callable[[dict[str, Any]], dict[str, Any]]


def process_entries(entries: list[dict[str, Any]], probe: ProbeCallable) -> list[dict[str, Any]]:
    """Apply a provider circuit breaker while retaining every requested row."""

    results: list[dict[str, Any]] = []
    provider_blocked = False
    for item in entries:
        common = {
            "tracking_index_name": item["tracking_index_name"],
            "price_code": item["price_code"],
            "total_return_code": item["total_return_code"],
            "registry_status": item["status"],
            "probe_status": "deferred_provider_blocked" if provider_blocked else "failed",
            "error": "upstream provider circuit is open" if provider_blocked else "",
        }
        if not provider_blocked:
            try:
                common.update(probe(item))
            except ProviderBlockedError as exc:
                provider_blocked = True
                common["probe_status"] = "provider_blocked"
                common["error"] = str(exc)
            except HistoryInsufficientError as exc:
                common["probe_status"] = "observation_only_insufficient_history"
                common["error"] = str(exc)
                common.update(exc.metrics)
            except Exception as exc:  # noqa: BLE001 - evidence row must survive a failed mapping
                common["error"] = str(exc)
        common["fetched_at"] = _now()
        results.append(common)
    return results


def select_pending_entries(
    registry: dict[str, Any],
    limit: int,
    price_codes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select pending CSIndex mappings, optionally by explicit price code."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    pending = [
        item
        for item in registry["mappings"]
        if item["provider"] == "csindex" and item["status"] in PENDING_STATUSES
    ]
    if price_codes:
        requested = list(dict.fromkeys(str(code).strip() for code in price_codes if str(code).strip()))
        pending_by_code = {str(item["price_code"]): item for item in pending}
        unavailable = [code for code in requested if code not in pending_by_code]
        if unavailable:
            raise ValueError(
                "requested price codes are not pending CSIndex mappings: " + ", ".join(unavailable)
            )
        pending = [pending_by_code[code] for code in requested]
    return pending[:limit]


def run(
    as_of: str | pd.Timestamp,
    raw_dir: Path,
    evidence_dir: Path,
    output_path: Path,
    manifest_path: Path,
    limit: int,
    sleep_seconds: float,
    price_codes: list[str] | None = None,
) -> dict[str, Any]:
    registry = load_index_registry()
    entries = select_pending_entries(registry, limit, price_codes)
    if not entries:
        raise ValueError("ETF index registry has no pending CSIndex mappings")
    cutoff = pd.Timestamp(as_of).normalize()
    performance_session = requests.Session()
    performance_session.trust_env = False
    performance_session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"})

    def probe(item: dict[str, Any]) -> dict[str, Any]:
        result = probe_mapping(item, cutoff, raw_dir, evidence_dir, performance_session)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return result

    results = process_entries(entries, probe)
    frame = pd.DataFrame(results)
    _atomic_csv(frame, output_path)
    generated_paths = sorted({path for result in results for path in result.get("cache_paths", {}).values()})
    evidence_paths = sorted({path for result in results for path in result.get("evidence_paths", [])})
    manifest: dict[str, Any] = {
        "created_at": _now(),
        "as_of_date": str(cutoff.date()),
        "registry_path": _relative(INDEX_REGISTRY_PATH),
        "registry_sha256": _sha256(INDEX_REGISTRY_PATH),
        "code_path": _relative(Path(__file__)),
        "code_sha256": _sha256(Path(__file__)),
        "requested_price_codes": list(price_codes or []),
        "selected_price_codes": [str(item["price_code"]) for item in entries],
        "requested_mappings": len(entries),
        "status_counts": frame["probe_status"].value_counts().to_dict(),
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "generated_cache_files": [
            {"path": path, "sha256": _sha256(ROOT / path)} for path in generated_paths
        ],
        "evidence_files": [{"path": path, "sha256": _sha256(ROOT / path)} for path in evidence_paths],
        "registry_mutated": False,
        "automatic_activation_allowed": False,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    return archive_probe_artifacts(frame, manifest, manifest_path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--price-code",
        action="append",
        help="Probe a specific pending CSIndex price code; repeat for multiple codes and raise --limit accordingly.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--archive-latest-only",
        action="store_true",
        help="Archive the existing latest probe evidence without making network requests.",
    )
    args = parser.parse_args()
    if args.archive_latest_only:
        result = archive_latest_probe(args.output, args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    cutoff = pd.Timestamp(args.as_of).normalize()
    raw_dir = args.raw_dir or RAW_ROOT / "etf_raw" / cutoff.strftime("%Y%m%d")
    evidence_dir = args.evidence_dir or RAW_ROOT / "pit_history" / "raw_etf_index_registry"
    result = run(
        cutoff,
        raw_dir,
        evidence_dir,
        args.output,
        args.manifest,
        args.limit,
        args.sleep_seconds,
        args.price_code,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
