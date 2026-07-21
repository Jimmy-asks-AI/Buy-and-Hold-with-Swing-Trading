"""Immutable raw-sample registry for HIRSSM V3.80."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RawSampleRegistryConfig:
    v3_78_manifest_path: Path
    v3_79_manifest_path: Path
    incoming_sample_dir: Path
    target_source_path: Path
    previous_registry_path: Path
    license_status_path: Path
    output_dir: Path
    catalog_path: Path
    approved_source_tokens: tuple[str, ...]
    allowed_extensions: tuple[str, ...]
    license_approved_values: tuple[str, ...]


RESERVED_NON_SAMPLE_CSVS = {
    "license_review_status.csv",
}


def _workspace_suffix(path: Path) -> str:
    anchors = ("data_raw", "outputs", "configs", "strategy_lab", "reports", "data_catalog")
    parts = path.parts
    for anchor in anchors:
        if anchor in parts:
            return Path(*parts[parts.index(anchor) :]).as_posix()
    return path.as_posix()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns)
    except Exception:
        return []


def _read_source_values(path: Path) -> list[str]:
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", nrows=10, low_memory=False)
    except Exception:
        return []
    source_cols = [col for col in frame.columns if str(col).strip().lower() in {"data_source", "source", "provider", "endpoint"}]
    values: list[str] = []
    for col in source_cols:
        values.extend(frame[col].dropna().astype(str).str.strip().unique().tolist())
    return sorted(set(values))


def _token_match(path: Path, source_values: list[str], tokens: tuple[str, ...]) -> str:
    text = (path.name + " " + " ".join(source_values)).lower()
    for token in tokens:
        if token.lower() in text:
            return token
    return ""


def _previous_first_seen(previous: pd.DataFrame, sha256: str, fallback: str) -> str:
    if previous.empty or "sha256" not in previous.columns or "first_seen_utc" not in previous.columns:
        return fallback
    rows = previous.loc[previous["sha256"].astype(str).eq(sha256)]
    if rows.empty:
        return fallback
    return str(rows["first_seen_utc"].iloc[0])


def read_previous_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()


def read_license_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.DataFrame()
    for column in ["sha256", "sample_file", "license_status", "license_evidence_path"]:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _license_for_sample(path: Path, digest: str, license_status: pd.DataFrame) -> tuple[str, str]:
    if license_status.empty:
        return "unknown", ""
    rows = license_status.loc[license_status["sha256"].astype(str).eq(digest)]
    if rows.empty:
        suffix = _workspace_suffix(path)
        rows = license_status.loc[license_status["sample_file"].astype(str).eq(suffix)]
    if rows.empty:
        return "unknown", ""
    row = rows.iloc[0]
    return str(row.get("license_status", "unknown")).strip() or "unknown", str(row.get("license_evidence_path", "")).strip()


def build_raw_sample_registry(config: RawSampleRegistryConfig) -> pd.DataFrame:
    previous = read_previous_registry(config.previous_registry_path)
    license_status = read_license_status(config.license_status_path)
    allowed_exts = {ext.lower() for ext in config.allowed_extensions}
    if not config.incoming_sample_dir.exists():
        return pd.DataFrame(
            [
                {
                    "registry_status": "waiting_for_sample_directory",
                    "sample_file": _workspace_suffix(config.incoming_sample_dir),
                    "file_exists": False,
                    "size_bytes": 0,
                    "sha256": "",
                    "first_seen_utc": "",
                    "last_seen_utc": _now_utc(),
                    "header_columns": "",
                    "declared_source_values": "",
                    "matched_source_token": "",
                    "source_token_approved": False,
                    "license_status": "missing",
                    "license_evidence_path": "",
                    "immutable_status": "no_file",
                    "v3_78_review_allowed": False,
                    "controlled_handoff_allowed": False,
                    "next_action": "create incoming sample directory and place licensed provider sample",
                }
            ]
        )
    files = sorted(
        path
        for path in config.incoming_sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_exts and path.name.lower() not in RESERVED_NON_SAMPLE_CSVS
    )
    if not files:
        return pd.DataFrame(
            [
                {
                    "registry_status": "waiting_for_sample_file",
                    "sample_file": _workspace_suffix(config.incoming_sample_dir),
                    "file_exists": True,
                    "size_bytes": 0,
                    "sha256": "",
                    "first_seen_utc": "",
                    "last_seen_utc": _now_utc(),
                    "header_columns": "",
                    "declared_source_values": "",
                    "matched_source_token": "",
                    "source_token_approved": False,
                    "license_status": "missing",
                    "license_evidence_path": "",
                    "immutable_status": "directory_empty",
                    "v3_78_review_allowed": False,
                    "controlled_handoff_allowed": False,
                    "next_action": "place licensed provider sample CSV in incoming directory",
                }
            ]
        )
    rows = []
    seen_at = _now_utc()
    for path in files:
        digest = _sha256(path)
        source_values = _read_source_values(path)
        matched = _token_match(path, source_values, config.approved_source_tokens)
        license_value, license_evidence_path = _license_for_sample(path, digest, license_status)
        license_ok = license_value in set(config.license_approved_values) and bool(license_evidence_path)
        token_ok = bool(matched)
        rows.append(
            {
                "registry_status": "registered_waiting_license_review",
                "sample_file": _workspace_suffix(path),
                "file_exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "first_seen_utc": _previous_first_seen(previous, digest, seen_at),
                "last_seen_utc": seen_at,
                "header_columns": "|".join(_read_header(path)),
                "declared_source_values": "|".join(source_values),
                "matched_source_token": matched,
                "source_token_approved": token_ok,
                "license_status": license_value,
                "license_evidence_path": license_evidence_path,
                "immutable_status": "hash_recorded",
                "v3_78_review_allowed": bool(token_ok and license_ok),
                "controlled_handoff_allowed": False,
                "next_action": "ready for V3.78 sample validation" if token_ok and license_ok else "attach license evidence and rerun registry before V3.78" if token_ok else "repair source metadata before V3.78",
            }
        )
    return pd.DataFrame(rows)


def build_registry_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "sample_file",
                "required": True,
                "description": "Workspace-relative immutable raw sample path.",
                "example": "data_raw/market_labels/incoming_samples/vendor_sample.csv",
            },
            {
                "column": "sha256",
                "required": True,
                "description": "Content hash used as the immutable identity.",
                "example": "64 hexadecimal characters",
            },
            {
                "column": "license_status",
                "required": True,
                "description": "Manual status before V3.78 review is allowed.",
                "example": "unknown|approved_internal_research|rejected",
            },
            {
                "column": "license_evidence_path",
                "required": True,
                "description": "Path to license, email, contract excerpt, or manual approval record.",
                "example": "outputs/agent_runs/v3_80/license_evidence/provider_approval.md",
            },
            {
                "column": "v3_78_review_allowed",
                "required": True,
                "description": "True only when source token and license status pass.",
                "example": "False",
            },
        ]
    )


def build_license_review_queue(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.itertuples(index=False):
        rows.append(
            {
                "sample_file": getattr(item, "sample_file", ""),
                "license_status": getattr(item, "license_status", ""),
                "source_token_approved": getattr(item, "source_token_approved", False),
                "review_status": "blocked" if not bool(getattr(item, "v3_78_review_allowed", False)) else "pass",
                "required_evidence": "license permits local storage, internal research, model validation, and derived label files",
                "next_action": getattr(item, "next_action", ""),
            }
        )
    return pd.DataFrame(rows)


def build_controlled_handoff(registry: pd.DataFrame, config: RawSampleRegistryConfig) -> pd.DataFrame:
    allowed = registry["v3_78_review_allowed"].astype(bool).any() if "v3_78_review_allowed" in registry.columns else False
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "handoff_step": "raw sample registry",
                "status": "done" if not registry.empty else "blocked",
                "may_execute_now": False,
                "reason": "hash/provenance registry produced",
            },
            {
                "step_order": 2,
                "handoff_step": "license review",
                "status": "active" if not allowed else "done",
                "may_execute_now": False,
                "reason": "manual evidence required before V3.78",
            },
            {
                "step_order": 3,
                "handoff_step": "run V3.78 on registered sample",
                "status": "active" if allowed else "blocked",
                "may_execute_now": False,
                "reason": "manual rerun after registry approval",
            },
            {
                "step_order": 4,
                "handoff_step": "copy to controlled V3.75 review path",
                "status": "blocked",
                "may_execute_now": False,
                "reason": "only after V3.78 pass",
            },
            {
                "step_order": 5,
                "handoff_step": "write final target source",
                "status": "blocked",
                "may_execute_now": False,
                "reason": f"target remains protected at {_workspace_suffix(config.target_source_path)}",
            },
        ]
    )


def build_immutability_policy(config: RawSampleRegistryConfig) -> str:
    return "\n".join(
        [
            "# V3.80 Raw Sample Immutability Policy",
            "",
            "## Rules",
            "",
            f"- Incoming directory: `{_workspace_suffix(config.incoming_sample_dir)}`",
            f"- Protected target source: `{_workspace_suffix(config.target_source_path)}`",
            "- Raw provider samples must be treated as immutable once registered.",
            "- Identity is the SHA256 hash, not the file name.",
            "- License evidence is required before a registered sample can be sent to V3.78.",
            "- A V3.78 pass still does not authorize final target write; V3.75 and V3.76 remain required.",
            "- Never overwrite an incoming raw sample in place. Add a new file and let the registry assign a new hash.",
            "",
            "## Allowed License Status Values",
            "",
            "- `unknown`",
            "- `approved_internal_research`",
            "- `approved_research_and_derived_labels`",
            "- `rejected`",
            "",
        ]
    )


def build_no_execution_guard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "raw_sample_registry",
                "produced": True,
                "blocked": False,
                "reason": "V3.80 creates provenance and license-gate evidence.",
            },
            {
                "result_type": "target_csv_write",
                "produced": False,
                "blocked": True,
                "reason": "Registry must not write the official target CSV.",
            },
            {
                "result_type": "v3_78_sample_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.80 records files but does not run sample validation.",
            },
            {
                "result_type": "v3_53_label_generation",
                "produced": False,
                "blocked": True,
                "reason": "No validated official source exists.",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "No labels are generated here.",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "Registry evidence is not model evidence.",
            },
        ]
    )


def build_acceptance_checks(registry: pd.DataFrame, queue: pd.DataFrame, handoff: pd.DataFrame, guard: pd.DataFrame, config: RawSampleRegistryConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "registry_written",
                "status": "pass" if not registry.empty else "fail",
                "detail": f"rows={len(registry)}",
            },
            {
                "check": "license_queue_written",
                "status": "pass" if not queue.empty else "fail",
                "detail": f"rows={len(queue)}",
            },
            {
                "check": "controlled_handoff_blocks_target_write",
                "status": "pass" if not handoff.loc[handoff["handoff_step"].astype(str).eq("write final target source"), "may_execute_now"].astype(bool).any() else "fail",
                "detail": "target write is blocked",
            },
            {
                "check": "target_source_not_written",
                "status": "pass" if not config.target_source_path.exists() else "warn",
                "detail": _workspace_suffix(config.target_source_path),
            },
            {
                "check": "downstream_not_executed",
                "status": "pass" if not guard.loc[guard["result_type"].isin(["target_csv_write", "v3_78_sample_validation", "v3_53_label_generation", "portfolio_backtest", "model_promotion"]), "produced"].astype(bool).any() else "fail",
                "detail": "registry only",
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    actual = [col for col in columns if col in frame.columns]
    for _, row in frame.loc[:, actual].head(max_rows).iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "/").replace("\n", " ") for col in columns) + " |")
    return lines


def build_report(registry: pd.DataFrame, queue: pd.DataFrame, handoff: pd.DataFrame, acceptance: pd.DataFrame, config: RawSampleRegistryConfig) -> str:
    registered = int(registry["sha256"].astype(str).ne("").sum()) if "sha256" in registry.columns else 0
    allowed = int(registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in registry.columns else 0
    lines = [
        "# V3.80 MARKET Raw Sample Registry",
        "",
        "## Decision",
        "",
        "- V3.80 records raw sample provenance and license-gate status before V3.78.",
        "- It does not validate sample contents, write the target source, generate labels, run portfolios, or promote a model.",
        "- Samples without license evidence remain blocked even if their source token looks valid.",
        "",
        "## Key Metrics",
        "",
        f"- Incoming directory: `{_workspace_suffix(config.incoming_sample_dir)}`",
        f"- Registered files with hash: `{registered}`",
        f"- Files allowed for V3.78 review: `{allowed}`",
        f"- Target source exists: `{config.target_source_path.exists()}`",
        "",
        "## Registry",
        "",
    ]
    lines.extend(markdown_table(registry, ["registry_status", "sample_file", "size_bytes", "sha256", "matched_source_token", "license_status", "v3_78_review_allowed", "next_action"], 20))
    lines.extend(["", "## License Review Queue", ""])
    lines.extend(markdown_table(queue, ["sample_file", "license_status", "source_token_approved", "review_status", "required_evidence", "next_action"], 20))
    lines.extend(["", "## Controlled Handoff", ""])
    lines.extend(markdown_table(handoff, ["step_order", "handoff_step", "status", "may_execute_now", "reason"], 20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], 20))
    lines.extend(["", "## Next Step", "", "- Attach license evidence for any real sample, then rerun V3.80 before V3.78.", ""])
    return "\n".join(lines)


def build_catalog(registry: pd.DataFrame, config: RawSampleRegistryConfig) -> str:
    registered = int(registry["sha256"].astype(str).ne("").sum()) if "sha256" in registry.columns else 0
    allowed = int(registry["v3_78_review_allowed"].astype(bool).sum()) if "v3_78_review_allowed" in registry.columns else 0
    return "\n".join(
        [
            "# A-share MARKET Raw Sample Registry V3.80",
            "",
            "## Dataset Decision",
            "",
            f"- Incoming sample directory: `{_workspace_suffix(config.incoming_sample_dir)}`",
            f"- Registered files with hash: `{registered}`",
            f"- Files allowed for V3.78 review: `{allowed}`",
            f"- Target source path: `{_workspace_suffix(config.target_source_path)}`",
            "- No target CSV, labels, portfolio validation, or model promotion are produced.",
            "",
        ]
    )
