"""Build an authenticated read view from native document text and OCR sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import pit_etf_terminal_event_document_collector as documents
from . import pit_etf_terminal_event_windows_ocr as ocr


ROOT = Path(__file__).resolve().parents[2]
OCR_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_windows_ocr_latest.json"
)
OBSERVATION_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations"
MERGED_INDEX_PATH = OBSERVATION_DIR / "etf_terminal_event_document_merged_index.csv"
REVIEW_QUEUE_PATH = ROOT / "data_catalog" / "long_hold_v4_etf_terminal_event_ocr_field_review_queue.csv"
REPORT_PATH = (
    ROOT
    / "outputs"
    / "long_hold_v4"
    / "pit_validation"
    / "etf_terminal_event_documents"
    / "document_merge_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_document_merge_latest.json"
)

SCHEMA_VERSION = 1
TEXT_AVAILABLE_STATUSES = {"success", "ocr_derived_unvalidated"}


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


def _authenticated_path(item: dict[str, Any]) -> Path:
    path = ROOT / str(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError(f"terminal document merge input hash mismatch: {item.get('role')}")
    return path


def _authenticate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    manifest = json.loads(OCR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "OCR_SIDECAR_COMPLETE_REQUIRES_FIELD_VALIDATION"
        or manifest.get("historical_backtest_allowed") is not False
        or manifest.get("model_promotion_allowed") is not False
    ):
        raise ValueError("OCR sidecar manifest does not authorize a merged observation view")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("OCR producer code hash mismatch")
    input_items = {str(item.get("role")): item for item in manifest.get("inputs", [])}
    output_items = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    base_item = input_items.get("document_index_snapshot")
    ocr_item = output_items.get("ocr_index")
    if base_item is None or ocr_item is None:
        raise ValueError("OCR manifest misses base document or OCR indexes")
    base_path = _authenticated_path(base_item)
    ocr_path = _authenticated_path(ocr_item)
    base = pd.read_csv(base_path, dtype={"asset": str})
    sidecar = pd.read_csv(ocr_path, dtype={"asset": str})
    lineage = [
        {"role": "ocr_manifest", "path": _relative(OCR_MANIFEST_PATH), "sha256": _sha256(OCR_MANIFEST_PATH)},
        {"role": "document_index_snapshot", "path": _relative(base_path), "sha256": _sha256(base_path)},
        {"role": "ocr_index", "path": _relative(ocr_path), "sha256": _sha256(ocr_path)},
    ]
    return base, sidecar, lineage


def build_merged_index(base: pd.DataFrame, sidecar: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required_base = {"asset", "source_url", "pdf_path", "pdf_sha256", "text_status", "text_path", "text_sha256"}
    required_ocr = {
        "asset",
        "source_url",
        "pdf_sha256",
        "ocr_status",
        "ocr_text_path",
        "ocr_text_sha256",
        "field_validation_status",
    }
    if required_base.difference(base.columns) or required_ocr.difference(sidecar.columns):
        raise ValueError("terminal document merge indexes miss required columns")
    if base.duplicated(["asset", "source_url"]).any() or sidecar.duplicated(["asset", "source_url"]).any():
        raise ValueError("terminal document merge indexes contain duplicate document keys")
    sidecar_by_key = {
        (str(row.asset).zfill(6), str(row.source_url)): row for row in sidecar.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    text_inputs: list[dict[str, Any]] = []
    used_ocr_keys: set[tuple[str, str]] = set()
    for row in base.itertuples(index=False):
        item = row._asdict()
        asset = str(row.asset).zfill(6)
        key = (asset, str(row.source_url))
        pdf_path = ROOT / str(row.pdf_path)
        if not pdf_path.is_file() or _sha256(pdf_path) != str(row.pdf_sha256):
            raise ValueError(f"terminal document PDF hash mismatch: {asset}/{row.source_url}")
        if str(row.text_status) == "success":
            text_path = ROOT / str(row.text_path)
            if not text_path.is_file() or _sha256(text_path) != str(row.text_sha256):
                raise ValueError(f"terminal native text hash mismatch: {asset}/{row.source_url}")
            validation_status = str(item.get("document_validation_status", "not_started"))
        elif str(row.text_status) == "no_extractable_text":
            ocr_row = sidecar_by_key.get(key)
            if ocr_row is None or str(ocr_row.ocr_status) != "ocr_derived_unvalidated":
                raise ValueError(f"terminal scanned PDF lacks an authenticated OCR sidecar: {asset}/{row.source_url}")
            if str(ocr_row.pdf_sha256) != str(row.pdf_sha256):
                raise ValueError(f"terminal OCR sidecar PDF hash mismatch: {asset}/{row.source_url}")
            text_path = ROOT / str(ocr_row.ocr_text_path)
            if not text_path.is_file() or _sha256(text_path) != str(ocr_row.ocr_text_sha256):
                raise ValueError(f"terminal OCR text hash mismatch: {asset}/{row.source_url}")
            item.update(
                {
                    "text_status": "ocr_derived_unvalidated",
                    "text_path": str(ocr_row.ocr_text_path),
                    "text_sha256": str(ocr_row.ocr_text_sha256),
                    "text_characters": int(ocr_row.ocr_text_characters),
                }
            )
            validation_status = "ocr_field_validation_not_started"
            used_ocr_keys.add(key)
        else:
            raise ValueError(f"terminal document text is unavailable: {asset}/{row.source_url}")
        item["asset"] = asset
        item["document_validation_status"] = validation_status
        item["historical_backtest_allowed"] = False
        item["model_promotion_allowed"] = False
        rows.append(item)
        role_suffix = hashlib.sha256(str(row.source_url).encode("utf-8")).hexdigest()[:16]
        text_inputs.append(
            {
                "role": f"merged_text:{asset}:{role_suffix}",
                "path": _relative(text_path),
                "sha256": _sha256(text_path),
            }
        )
    if used_ocr_keys != set(sidecar_by_key):
        raise ValueError("OCR sidecar contains rows that do not map to native text gaps")
    merged = pd.DataFrame(rows).reindex(columns=documents.DOCUMENT_INDEX_COLUMNS).sort_values(
        ["asset", "announcement_date", "source_url"]
    ).reset_index(drop=True)
    if not merged["text_status"].isin(TEXT_AVAILABLE_STATUSES).all():
        raise ValueError("merged terminal document view contains unavailable text")
    return merged, text_inputs


def run() -> dict[str, Any]:
    base, sidecar, inputs = _authenticate_inputs()
    merged, text_inputs = build_merged_index(base, sidecar)
    review = merged[merged["text_status"].eq("ocr_derived_unvalidated")].copy()
    _atomic_csv(merged, MERGED_INDEX_PATH)
    _atomic_csv(review, REVIEW_QUEUE_PATH)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "FULL_IMMUTABLE_DOCUMENT_TEXT_VIEW_REQUIRES_EVENT_VALIDATION",
        "document_rows": int(len(merged)),
        "covered_assets": int(merged["asset"].nunique()),
        "native_text_rows": int(merged["text_status"].eq("success").sum()),
        "ocr_derived_unvalidated_rows": int(merged["text_status"].eq("ocr_derived_unvalidated").sum()),
        "ocr_field_review_queue_rows": int(len(review)),
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "The merged view is readable evidence only. OCR-derived fields cannot be promoted without PDF or independent official-text reconciliation.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "merged_document_index", "path": _relative(MERGED_INDEX_PATH), "sha256": _sha256(MERGED_INDEX_PATH), "rows": int(len(merged))},
        {"role": "ocr_field_review_queue", "path": _relative(REVIEW_QUEUE_PATH), "sha256": _sha256(REVIEW_QUEUE_PATH), "rows": int(len(review))},
        {"role": "document_merge_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": [*inputs, *text_inputs],
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {"path": _relative(Path(documents.__file__).resolve()), "sha256": _sha256(Path(documents.__file__).resolve())},
            {"path": _relative(Path(ocr.__file__).resolve()), "sha256": _sha256(Path(ocr.__file__).resolve())},
        ],
        "current_final_snapshot": True,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parse_args()
    result = run()
    keys = (
        "qualification_status",
        "document_rows",
        "native_text_rows",
        "ocr_derived_unvalidated_rows",
    )
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
