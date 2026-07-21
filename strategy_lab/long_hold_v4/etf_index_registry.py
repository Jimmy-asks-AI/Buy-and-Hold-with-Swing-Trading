"""Govern reviewed ETF benchmark mappings and their activation status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INDEX_REGISTRY_PATH = ROOT / "configs" / "long_hold_v4_etf_index_registry.json"
ALLOWED_STATUSES = {
    "active",
    "verified_pending_history_cache",
    "verified_pending_history_probe",
    "observation_only_insufficient_pe_history",
    "observation_only_missing_historical_valuation",
    "unresolved_external_provider",
}
ALLOWED_PROVIDERS = {"csindex", "cni", "spglobal"}


def load_index_registry(path: Path = INDEX_REGISTRY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("ETF index registry schema_version must be 1")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("ETF index registry mappings must be a non-empty list")
    names: set[str] = set()
    active_codes: set[tuple[str, str]] = set()
    for position, item in enumerate(mappings):
        if not isinstance(item, dict):
            raise ValueError(f"ETF index registry row {position} is not an object")
        name = str(item.get("tracking_index_name", "")).strip()
        provider = str(item.get("provider", "")).strip()
        status = str(item.get("status", "")).strip()
        if not name or name in names:
            raise ValueError(f"ETF index registry contains missing/duplicate name: {name!r}")
        names.add(name)
        if provider not in ALLOWED_PROVIDERS:
            raise ValueError(f"ETF index registry has unsupported provider for {name}: {provider}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"ETF index registry has unsupported status for {name}: {status}")
        observed_count = item.get("observed_etf_count")
        if not isinstance(observed_count, int) or observed_count <= 0:
            raise ValueError(f"ETF index registry has invalid observed_etf_count for {name}")
        if not isinstance(item.get("local_cache_ready"), bool):
            raise ValueError(f"ETF index registry has invalid local_cache_ready flag for {name}")
        evidence_urls = item.get("evidence_urls")
        if not isinstance(evidence_urls, list) or any(not isinstance(url, str) or not url.startswith("https://") for url in evidence_urls):
            raise ValueError(f"ETF index registry has invalid evidence URLs for {name}")
        if status != "unresolved_external_provider":
            price_code = str(item.get("price_code", "")).strip()
            total_code = str(item.get("total_return_code", "")).strip()
            total_name = str(item.get("total_return_name", "")).strip()
            if not price_code or not total_code or not total_name or price_code == total_code or not evidence_urls:
                raise ValueError(f"ETF index registry has incomplete reviewed codes for {name}")
        if status == "active":
            if item.get("history_identity_status") != "passed" or item.get("valuation_identity_status") != "passed":
                raise ValueError(f"active ETF index mapping lacks identity evidence for {name}")
            if item.get("local_cache_ready") is not True:
                raise ValueError(f"active ETF index mapping lacks a ready local cache for {name}")
            code_key = (str(item["price_code"]), str(item["total_return_code"]))
            if code_key in active_codes:
                raise ValueError(f"active ETF index mapping reuses a code pair: {code_key}")
            active_codes.add(code_key)
    return payload


def active_index_map(path: Path = INDEX_REGISTRY_PATH) -> dict[str, dict[str, str]]:
    registry = load_index_registry(path)
    return {
        str(item["tracking_index_name"]): {
            "price_code": str(item["price_code"]),
            "total_return_code": str(item["total_return_code"]),
        }
        for item in registry["mappings"]
        if item["status"] == "active"
    }
