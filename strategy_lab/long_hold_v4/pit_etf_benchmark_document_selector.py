"""Select a bounded, auditable official-document set for ETF benchmark history."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_official_announcement_catalog_latest.json"
)
RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark_documents"
LINEAGE_DIR = RAW_DIR / "lineage"
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
SELECTION_PATH = OBSERVATION_DIR / "etf_benchmark_document_selection.csv"
COVERAGE_PATH = OBSERVATION_DIR / "etf_benchmark_document_selection_coverage_registry.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_benchmark_document_collection_queue.csv"
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_benchmark_document_selection" / "selection_report.json"
)
MANIFEST_PATH = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_document_selector_latest.json"

SCHEMA_VERSION = 1
SELECTION_POLICY_VERSION = "benchmark_document_routing_v7"
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
NON_SUBSTANTIVE_INITIAL_PATTERN = re.compile(r"公告|终止|清算|提示")
NON_FULL_PROSPECTUS_PATTERN = re.compile(r"摘要|产品资料概要|公告")
POST_EVENT_ANCHOR_PATTERN = re.compile(
    r"(?:变更|更换|调整|更新).{0,80}(?:标的指数|业绩比较基准)|"
    r"(?:标的指数|业绩比较基准).{0,80}(?:名称|代码)?.{0,40}(?:变更|更换|调整|更新)|"
    r"基金合同生效暨基金更名"
)
FUND_RENAME_POST_EVENT_ANCHOR_PATTERN = re.compile(
    r"(?:变更|更改).{0,120}基金名称.{0,80}(?:修订|修改).{0,40}基金合同|"
    r"基金更名.{0,80}(?:修订|修改).{0,40}(?:基金合同|法律文件)|"
    r"更名并(?:修订|修改).{0,40}(?:基金合同|法律文件)"
)
POST_EVENT_ANCHOR_EXCLUSION_PATTERN = re.compile(
    r"许可使用费|指数使用费|使用许可费|许可使用基点费|纳入北京证券交易所股票|"
    r"调整样本|修订指数编制方案|基金托管人(?:名称)?(?:更名|变更)|"
    r"基金管理人(?:名称)?(?:更名|变更)"
)
NON_FULL_POST_EVENT_LEGAL_PATTERN = re.compile(r"公告|提示|摘要|产品资料概要|法律意见|托管协议")

SOURCE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "announcement_date",
    "published_at",
    "announcement_title",
    "source_url",
    "source_type",
    "source_category",
    "source_observed_at",
    "available_at",
    "available_trade_date",
    "available_date",
    "data_source",
    "source_vintage",
    "query_path",
    "query_sha256",
    "title_tags_json",
    "candidate_roles_json",
]
SELECTION_COLUMNS = [
    *SOURCE_COLUMNS,
    "document_key",
    "selection_policy_version",
    "selection_priority",
    "selection_priority_rank",
    "selection_reasons_json",
    "baseline_selection_state",
    "document_validation_status",
    "benchmark_evidence_state",
    "no_change_claim_allowed",
    "historical_backtest_allowed",
    "model_promotion_allowed",
]
COVERAGE_COLUMNS = [
    "asset",
    "asset_name",
    "exchange",
    "list_date",
    "delist_date",
    "candidate_document_count",
    "selected_document_count",
    "p0_document_count",
    "p1_document_count",
    "baseline_document_count",
    "preferred_baseline_document_count",
    "legal_baseline_document_count",
    "listing_only_baseline_document_count",
    "fallback_baseline_document_count",
    "listing_context_document_count",
    "initial_prospectus_supplement_document_count",
    "updated_prospectus_fallback_document_count",
    "benchmark_change_document_count",
    "contract_amendment_document_count",
    "holder_resolution_document_count",
    "post_event_fund_contract_document_count",
    "post_event_prospectus_document_count",
    "selection_state",
    "benchmark_evidence_state",
    "no_change_claim_allowed",
    "review_priority",
    "review_reason",
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
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _atomic_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 >= ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(ATOMIC_REPLACE_SLEEP_SECONDS)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        lineterminator="\n",
    ).encode("utf-8-sig")
    _atomic_bytes(payload, path)


def _content_snapshot(path: Path) -> Path:
    digest = _sha256(path)
    snapshot = LINEAGE_DIR / f"{digest}{path.suffix.lower()}"
    if not snapshot.is_file():
        _atomic_bytes(path.read_bytes(), snapshot)
    if _sha256(snapshot) != digest:
        raise ValueError(f"ETF benchmark lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _json_values(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return set()
    parsed = value if isinstance(value, list) else json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON list, received {type(parsed).__name__}")
    return {str(item) for item in parsed}


def _json_list(values: set[str] | list[str]) -> str:
    return json.dumps(sorted({str(value) for value in values if str(value)}), ensure_ascii=False)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_substantive_initial(row: Any) -> bool:
    tags = row._tags
    roles = row._roles
    title = str(row.announcement_title)
    if "initial_benchmark_candidate" not in roles:
        return False
    if "listing_document" in tags:
        return True
    if NON_SUBSTANTIVE_INITIAL_PATTERN.search(title):
        return False
    if "fund_contract" in tags:
        return "contract_amendment" not in tags
    return "prospectus" in tags and "prospectus_update" not in tags


def _is_post_event_anchor(row: Any) -> bool:
    if "holder_resolution" in row._tags:
        return True
    title = str(row.announcement_title)
    benchmark_anchor = bool(
        "benchmark_change_candidate" in row._roles
        and POST_EVENT_ANCHOR_PATTERN.search(title)
    )
    fund_rename_anchor = bool(
        "contract_amendment" in row._tags
        and FUND_RENAME_POST_EVENT_ANCHOR_PATTERN.search(title)
    )
    return bool(
        (benchmark_anchor or fund_rename_anchor)
        and not POST_EVENT_ANCHOR_EXCLUSION_PATTERN.search(title)
    )


def _post_event_legal_kind(row: Any) -> str:
    title = str(row.announcement_title)
    if NON_FULL_POST_EVENT_LEGAL_PATTERN.search(title):
        return ""
    if "fund_contract" in row._tags and "基金合同" in title and "招募说明书" not in title:
        return "fund_contract"
    if "prospectus" in row._tags and "招募说明书" in title:
        return "prospectus"
    return ""


def _preferred_baseline_rank(row: Any) -> tuple[int, pd.Timestamp, str, str]:
    tags = row._tags
    title = str(row.announcement_title)
    if "fund_contract" in tags and "摘要" not in title:
        rank = 0
    elif "prospectus" in tags:
        rank = 1
    elif "fund_contract" in tags:
        rank = 2
    elif "listing_document" in tags:
        rank = 3
    else:
        rank = 9
    return rank, row._date, title, str(row.source_url)


def _fallback_baseline_rank(row: Any) -> tuple[pd.Timestamp, int, str, str]:
    tags = row._tags
    if "fund_contract" in tags:
        authority_rank = 0
    elif "prospectus" in tags:
        authority_rank = 1
    elif "product_summary" in tags:
        authority_rank = 2
    else:
        authority_rank = 9
    return row._date, authority_rank, str(row.announcement_title), str(row.source_url)


def _document_key(asset: str, source_url: str) -> str:
    return hashlib.sha256(f"{asset}|{source_url}".encode("utf-8")).hexdigest()


def select_documents(candidates: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(SOURCE_COLUMNS + ["historical_backtest_allowed", "model_promotion_allowed"]).difference(candidates.columns))
    if missing:
        raise ValueError(f"ETF benchmark candidates miss columns: {missing}")
    required_coverage = {"asset", "asset_name", "exchange", "list_date", "delist_date", "query_complete"}
    missing = sorted(required_coverage.difference(coverage.columns))
    if missing:
        raise ValueError(f"ETF benchmark discovery coverage misses columns: {missing}")

    frame = candidates.copy().reset_index(drop=True)
    registry = coverage.copy().reset_index(drop=True)
    frame["asset"] = frame["asset"].astype(str).str.zfill(6)
    registry["asset"] = registry["asset"].astype(str).str.zfill(6)
    frame["_date"] = pd.to_datetime(frame["announcement_date"], errors="coerce").dt.normalize()
    if frame["_date"].isna().any() or frame["source_url"].astype(str).str.strip().eq("").any():
        raise ValueError("ETF benchmark candidates contain invalid dates or source URLs")
    if frame.duplicated(["asset", "source_url"]).any():
        raise ValueError("ETF benchmark candidates contain duplicate asset/source_url keys")
    if frame["historical_backtest_allowed"].map(_is_true).any() or frame["model_promotion_allowed"].map(_is_true).any():
        raise ValueError("discovery candidates unexpectedly authorize historical use")
    if not registry["query_complete"].map(_is_true).all():
        raise ValueError("document selection requires complete official announcement queries")
    candidate_assets = set(frame["asset"])
    coverage_assets = set(registry["asset"])
    if candidate_assets != coverage_assets:
        raise ValueError(
            "ETF benchmark candidate/coverage asset mismatch: "
            f"missing_candidates={sorted(coverage_assets - candidate_assets)[:10]};"
            f"outside_coverage={sorted(candidate_assets - coverage_assets)[:10]}"
        )

    frame["_tags"] = frame["title_tags_json"].map(_json_values)
    frame["_roles"] = frame["candidate_roles_json"].map(_json_values)
    reasons: dict[int, set[str]] = {int(index): set() for index in frame.index}
    baseline_states: dict[int, str] = {}

    for asset, asset_rows in frame.groupby("asset", sort=True):
        preferred_indices = [int(index) for index, row in asset_rows.iterrows() if _is_substantive_initial(row)]
        if preferred_indices:
            baseline_index = min(preferred_indices, key=lambda index: _preferred_baseline_rank(frame.loc[index]))
            baseline_tags = frame.loc[baseline_index, "_tags"]
            if "listing_document" in baseline_tags:
                reason = "one_listing_only_initial_document"
                state = "listing_only_initial_candidate"
            else:
                reason = "one_preferred_initial_legal_document"
                state = "preferred_initial_legal_candidate"
        else:
            fallback_indices = [
                int(index)
                for index, row in asset_rows.iterrows()
                if row._tags.intersection({"fund_contract", "prospectus", "product_summary"})
                and not NON_SUBSTANTIVE_INITIAL_PATTERN.search(str(row.announcement_title))
            ]
            if not fallback_indices:
                raise ValueError(f"no substantive initial or fallback benchmark document for {asset}")
            baseline_index = min(fallback_indices, key=lambda index: _fallback_baseline_rank(frame.loc[index]))
            reason = "one_fallback_updated_benchmark_document_requires_validation"
            state = "fallback_update_candidate"
        reasons[baseline_index].add(reason)
        baseline_states[baseline_index] = state

        listing_indices = [
            int(index) for index, row in asset_rows.iterrows() if "listing_document" in row._tags
        ]
        if listing_indices:
            listing_index = min(
                listing_indices,
                key=lambda index: (
                    frame.loc[index, "_date"],
                    str(frame.loc[index, "announcement_title"]),
                    str(frame.loc[index, "source_url"]),
                ),
            )
            reasons[listing_index].add("one_canonical_listing_context_document")

        prospectus_indices = [
            int(index)
            for index, row in asset_rows.iterrows()
            if "prospectus" in row._tags
            and "prospectus_update" not in row._tags
            and not NON_SUBSTANTIVE_INITIAL_PATTERN.search(str(row.announcement_title))
        ]
        if prospectus_indices:
            prospectus_index = min(
                prospectus_indices,
                key=lambda index: (
                    frame.loc[index, "_date"],
                    str(frame.loc[index, "announcement_title"]),
                    str(frame.loc[index, "source_url"]),
                ),
            )
            reasons[prospectus_index].add("one_initial_prospectus_supplement_document")
        else:
            updated_prospectus_indices = [
                int(index)
                for index, row in asset_rows.iterrows()
                if "prospectus" in row._tags
                and "prospectus_update" in row._tags
                and not NON_FULL_PROSPECTUS_PATTERN.search(str(row.announcement_title))
            ]
            if updated_prospectus_indices:
                updated_prospectus_index = min(
                    updated_prospectus_indices,
                    key=lambda index: (
                        frame.loc[index, "_date"],
                        str(frame.loc[index, "announcement_title"]),
                        str(frame.loc[index, "source_url"]),
                    ),
                )
                reasons[updated_prospectus_index].add(
                    "one_earliest_updated_prospectus_fallback_document"
                )

        anchor_indices = [
            int(index) for index, row in asset_rows.iterrows() if _is_post_event_anchor(row)
        ]
        for anchor_index in anchor_indices:
            anchor = frame.loc[anchor_index]
            anchor_date = anchor["_date"]
            strict_after = "holder_resolution" in anchor["_tags"]
            maximum_date = anchor_date + pd.Timedelta(days=730)
            legal_indices = [
                int(index)
                for index, row in asset_rows.iterrows()
                if _post_event_legal_kind(row)
                and (row._date > anchor_date if strict_after else row._date >= anchor_date)
                and row._date <= maximum_date
            ]
            for kind, reason_name in (
                ("fund_contract", "one_first_post_event_fund_contract_document"),
                ("prospectus", "one_first_post_event_prospectus_document"),
            ):
                kind_indices = [
                    index for index in legal_indices if _post_event_legal_kind(frame.loc[index]) == kind
                ]
                if kind_indices:
                    selected_index = min(
                        kind_indices,
                        key=lambda index: (
                            frame.loc[index, "_date"],
                            str(frame.loc[index, "announcement_title"]),
                            str(frame.loc[index, "source_url"]),
                        ),
                    )
                    reasons[selected_index].add(reason_name)

    for index, row in frame.iterrows():
        if "benchmark_change_candidate" in row._roles:
            reasons[int(index)].add("all_title_routed_benchmark_change_documents")
        if "contract_amendment" in row._tags:
            reasons[int(index)].add("all_contract_amendments")
        if "holder_resolution" in row._tags:
            reasons[int(index)].add("all_holder_resolutions")

    selected_indices = [index for index, values in reasons.items() if values]
    selected = frame.loc[selected_indices].copy()
    selected["document_key"] = [
        _document_key(str(row.asset), str(row.source_url)) for row in selected.itertuples(index=False)
    ]
    selected["selection_policy_version"] = SELECTION_POLICY_VERSION
    selected["selection_reasons_json"] = [_json_list(reasons[int(index)]) for index in selected.index]
    selected["baseline_selection_state"] = [
        baseline_states.get(int(index), "not_baseline") for index in selected.index
    ]
    selected["selection_priority"] = [
        "P0"
        if baseline_state != "not_baseline"
        or bool(
            _json_values(selection_reasons).intersection(
                {
                    "all_title_routed_benchmark_change_documents",
                    "one_first_post_event_fund_contract_document",
                    "one_first_post_event_prospectus_document",
                }
            )
        )
        else "P1"
        for baseline_state, selection_reasons in zip(
            selected["baseline_selection_state"], selected["selection_reasons_json"], strict=True
        )
    ]
    selected["selection_priority_rank"] = selected["selection_priority"].map({"P0": 0, "P1": 1})
    selected["document_validation_status"] = "not_started"
    selected["benchmark_evidence_state"] = "evidence_insufficient"
    selected["no_change_claim_allowed"] = False
    selected["historical_backtest_allowed"] = False
    selected["model_promotion_allowed"] = False
    selected = selected.reindex(columns=SELECTION_COLUMNS).sort_values(
        ["selection_priority_rank", "asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)

    if selected["document_key"].duplicated().any():
        raise ValueError("ETF benchmark selection produced duplicate document keys")
    baseline_counts = selected["baseline_selection_state"].ne("not_baseline").groupby(selected["asset"]).sum()
    if not baseline_counts.reindex(sorted(coverage_assets), fill_value=0).eq(1).all():
        raise ValueError("ETF benchmark selection must contain exactly one baseline document per asset")

    selection_rows = {asset: rows for asset, rows in selected.groupby("asset", sort=True)}
    coverage_records: list[dict[str, Any]] = []
    for row in registry.itertuples(index=False):
        asset = str(row.asset)
        asset_selection = selection_rows[asset]
        reason_sets = asset_selection["selection_reasons_json"].map(_json_values)
        legal = int(asset_selection["baseline_selection_state"].eq("preferred_initial_legal_candidate").sum())
        listing_only = int(asset_selection["baseline_selection_state"].eq("listing_only_initial_candidate").sum())
        fallback = int(asset_selection["baseline_selection_state"].eq("fallback_update_candidate").sum())
        coverage_records.append(
            {
                "asset": asset,
                "asset_name": str(row.asset_name),
                "exchange": str(row.exchange),
                "list_date": row.list_date,
                "delist_date": row.delist_date,
                "candidate_document_count": int(frame["asset"].eq(asset).sum()),
                "selected_document_count": int(len(asset_selection)),
                "p0_document_count": int(asset_selection["selection_priority"].eq("P0").sum()),
                "p1_document_count": int(asset_selection["selection_priority"].eq("P1").sum()),
                "baseline_document_count": legal + listing_only + fallback,
                "preferred_baseline_document_count": legal + listing_only,
                "legal_baseline_document_count": legal,
                "listing_only_baseline_document_count": listing_only,
                "fallback_baseline_document_count": fallback,
                "listing_context_document_count": int(reason_sets.map(lambda values: "one_canonical_listing_context_document" in values).sum()),
                "initial_prospectus_supplement_document_count": int(
                    reason_sets.map(lambda values: "one_initial_prospectus_supplement_document" in values).sum()
                ),
                "updated_prospectus_fallback_document_count": int(
                    reason_sets.map(
                        lambda values: "one_earliest_updated_prospectus_fallback_document" in values
                    ).sum()
                ),
                "benchmark_change_document_count": int(reason_sets.map(lambda values: "all_title_routed_benchmark_change_documents" in values).sum()),
                "contract_amendment_document_count": int(reason_sets.map(lambda values: "all_contract_amendments" in values).sum()),
                "holder_resolution_document_count": int(reason_sets.map(lambda values: "all_holder_resolutions" in values).sum()),
                "post_event_fund_contract_document_count": int(
                    reason_sets.map(
                        lambda values: "one_first_post_event_fund_contract_document" in values
                    ).sum()
                ),
                "post_event_prospectus_document_count": int(
                    reason_sets.map(
                        lambda values: "one_first_post_event_prospectus_document" in values
                    ).sum()
                ),
                "selection_state": "selected_for_document_collection",
                "benchmark_evidence_state": "evidence_insufficient",
                "no_change_claim_allowed": False,
                "review_priority": "P0" if fallback else "P1",
                "review_reason": (
                    "fallback_initial_document_requires_content_validation"
                    if fallback
                    else "collect_and_parse_selected_official_documents"
                ),
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
    selection_coverage = pd.DataFrame(coverage_records).reindex(columns=COVERAGE_COLUMNS).sort_values("asset").reset_index(drop=True)
    return selected, selection_coverage


def _authenticate_discovery() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(DISCOVERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        != "FULL_AUTHENTICATED_MASTER_TITLE_CATALOG_DOCUMENT_VALIDATION_REQUIRED"
        or int(manifest.get("target_assets", 0)) != 1701
        or int(manifest.get("query_complete_assets", 0)) != 1701
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF benchmark discovery does not authorize full-universe document routing")
    code_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        code_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    paths: dict[str, Path] = {}
    inputs: list[dict[str, Any]] = []
    manifest_snapshot = _content_snapshot(DISCOVERY_MANIFEST_PATH)
    inputs.append({"role": "discovery_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)})
    inputs.append({"role": "authenticated_discovery_code", "path": _relative(authenticated_code), "sha256": _sha256(authenticated_code)})
    for role in ("benchmark_document_candidates", "benchmark_discovery_coverage"):
        item = outputs.get(role, {})
        path = ROOT / str(item.get("path", ""))
        if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
            raise ValueError(f"ETF benchmark discovery output hash mismatch: {role}")
        snapshot = _content_snapshot(path)
        paths[role] = snapshot
        inputs.append({"role": f"{role}_snapshot", "path": _relative(snapshot), "sha256": _sha256(snapshot)})
    candidates = pd.read_csv(paths["benchmark_document_candidates"], dtype={"asset": str}, low_memory=False)
    coverage = pd.read_csv(paths["benchmark_discovery_coverage"], dtype={"asset": str}, low_memory=False)
    return manifest, candidates, coverage, inputs


def run_selection() -> dict[str, Any]:
    discovery_manifest, candidates, coverage, inputs = _authenticate_discovery()
    selection, selection_coverage = select_documents(candidates, coverage)
    review_queue = selection.copy()
    _atomic_csv(selection, SELECTION_PATH)
    _atomic_csv(selection_coverage, COVERAGE_PATH)
    _atomic_csv(review_queue, REVIEW_QUEUE_PATH)

    reason_sets = selection["selection_reasons_json"].map(_json_values)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(discovery_manifest["as_of_date"]),
        "qualification_status": "FULL_AUTHENTICATED_DOCUMENT_ROUTING_COLLECTION_REQUIRED",
        "selection_policy_version": SELECTION_POLICY_VERSION,
        "target_assets": int(coverage["asset"].nunique()),
        "candidate_documents": int(len(candidates)),
        "selected_documents": int(len(selection)),
        "p0_documents": int(selection["selection_priority"].eq("P0").sum()),
        "p1_documents": int(selection["selection_priority"].eq("P1").sum()),
        "preferred_baseline_assets": int(
            selection["baseline_selection_state"].isin(
                {"preferred_initial_legal_candidate", "listing_only_initial_candidate"}
            ).sum()
        ),
        "legal_baseline_assets": int(selection["baseline_selection_state"].eq("preferred_initial_legal_candidate").sum()),
        "listing_only_baseline_assets": int(selection["baseline_selection_state"].eq("listing_only_initial_candidate").sum()),
        "fallback_baseline_assets": int(selection["baseline_selection_state"].eq("fallback_update_candidate").sum()),
        "listing_context_documents": int(reason_sets.map(lambda values: "one_canonical_listing_context_document" in values).sum()),
        "initial_prospectus_supplement_documents": int(
            reason_sets.map(lambda values: "one_initial_prospectus_supplement_document" in values).sum()
        ),
        "updated_prospectus_fallback_documents": int(
            reason_sets.map(
                lambda values: "one_earliest_updated_prospectus_fallback_document" in values
            ).sum()
        ),
        "benchmark_change_documents": int(reason_sets.map(lambda values: "all_title_routed_benchmark_change_documents" in values).sum()),
        "contract_amendment_documents": int(reason_sets.map(lambda values: "all_contract_amendments" in values).sum()),
        "holder_resolution_documents": int(reason_sets.map(lambda values: "all_holder_resolutions" in values).sum()),
        "post_event_fund_contract_documents": int(
            reason_sets.map(
                lambda values: "one_first_post_event_fund_contract_document" in values
            ).sum()
        ),
        "post_event_prospectus_documents": int(
            reason_sets.map(
                lambda values: "one_first_post_event_prospectus_document" in values
            ).sum()
        ),
        "unselected_observation_documents": int(len(candidates) - len(selection)),
        "document_collection_complete": False,
        "formal_history_rows": 0,
        "benchmark_history_complete_assets": 0,
        "official_no_benchmark_change_assets": 0,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "Deterministic document routing only. A baseline title, a zero-result title search, or a selected document "
            "does not establish benchmark history or absence of benchmark changes."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "benchmark_document_selection", "path": _relative(SELECTION_PATH), "sha256": _sha256(SELECTION_PATH), "rows": int(len(selection))},
        {"role": "benchmark_document_selection_coverage", "path": _relative(COVERAGE_PATH), "sha256": _sha256(COVERAGE_PATH), "rows": int(len(selection_coverage))},
        {"role": "benchmark_document_collection_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review_queue))},
        {"role": "selection_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(source_code_archive.__file__).resolve()), "sha256": _sha256(Path(source_code_archive.__file__).resolve())}
        ],
        "current_final_snapshot": True,
        "contains_benchmark_facts": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def main() -> None:
    result = run_selection()
    keys = (
        "qualification_status",
        "target_assets",
        "candidate_documents",
        "selected_documents",
        "p0_documents",
        "p1_documents",
        "preferred_baseline_assets",
        "fallback_baseline_assets",
        "historical_backtest_allowed",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
