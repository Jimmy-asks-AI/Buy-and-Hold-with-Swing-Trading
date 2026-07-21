"""Add Windows Media OCR text to terminal-event PDFs without native text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from . import pit_etf_terminal_event_document_collector as documents


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_document_collector_latest.json"
)
OCR_SCRIPT_PATH = ROOT / "scripts" / "windows_ocr_images.ps1"
OCR_INDEX_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "etf_terminal_event_ocr_index.csv"
)
LINEAGE_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_terminal_event_universe"
    / "ocr_lineage"
)
REPORT_PATH = (
    ROOT / "outputs" / "long_hold_v4" / "pit_validation" / "etf_terminal_event_documents" / "ocr_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_terminal_event_windows_ocr_latest.json"
)

SCHEMA_VERSION = 2
RENDER_SCALE = 2.0
LANGUAGE_TAG = "zh-Hans-CN"
OCR_COLUMNS = [
    "asset",
    "announcement_date",
    "announcement_title",
    "source_url",
    "pdf_path",
    "pdf_sha256",
    "page_count",
    "ocr_status",
    "ocr_text_path",
    "ocr_text_sha256",
    "ocr_text_characters",
    "ocr_engine",
    "ocr_language",
    "render_scale",
    "ocr_script_sha256",
    "field_validation_status",
    "error",
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


def _content_snapshot(path: Path) -> Path:
    digest = _sha256(path)
    snapshot = LINEAGE_DIR / f"{digest}{path.suffix.lower()}"
    if not snapshot.is_file():
        _atomic_bytes(path.read_bytes(), snapshot)
    if _sha256(snapshot) != digest:
        raise ValueError(f"OCR lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _authenticate_document_index() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    manifest = json.loads(DOCUMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status")
        not in {
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_COMPLETE",
            "FULL_SELECTED_OFFICIAL_DOCUMENT_SET_COLLECTED_NATIVE_TEXT_GAPS",
        }
        or int(manifest.get("covered_assets", 0)) != 123
        or int(manifest.get("collected_documents", 0)) != int(manifest.get("selected_documents", -1))
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("terminal document collection does not authorize OCR supplementation")
    code_path = ROOT / str(manifest.get("code_path", ""))
    if not code_path.is_file() or _sha256(code_path) != str(manifest.get("code_sha256", "")):
        raise ValueError("terminal document collector code hash mismatch")
    output = next(
        (item for item in manifest.get("outputs", []) if item.get("role") == "document_index"),
        None,
    )
    if output is None:
        raise ValueError("terminal document manifest misses document index")
    path = ROOT / str(output.get("path", ""))
    if not path.is_file() or _sha256(path) != str(output.get("sha256", "")):
        raise ValueError("terminal document index hash mismatch")
    manifest_snapshot = _content_snapshot(DOCUMENT_MANIFEST_PATH)
    index_snapshot = _content_snapshot(path)
    frame = pd.read_csv(index_snapshot, dtype={"asset": str})
    return frame, [
        {"role": "document_manifest_snapshot", "path": _relative(manifest_snapshot), "sha256": _sha256(manifest_snapshot)},
        {"role": "document_index_snapshot", "path": _relative(index_snapshot), "sha256": _sha256(index_snapshot)},
        {"role": "windows_ocr_script", "path": _relative(OCR_SCRIPT_PATH), "sha256": _sha256(OCR_SCRIPT_PATH)},
    ]


def _ocr_pdf(pdf_path: Path, output_path: Path) -> str:
    work_root = documents.RAW_DIR / "ocr_work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terminal_ocr_", dir=work_root) as temporary:
        temporary_path = Path(temporary)
        with fitz.open(pdf_path) as pdf:
            for number, page in enumerate(pdf, start=1):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE),
                    alpha=False,
                )
                pixmap.save(temporary_path / f"page_{number:04d}.png")
        temporary_output = temporary_path / "ocr.txt"
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(OCR_SCRIPT_PATH),
                "-InputDirectory",
                str(temporary_path),
                "-OutputPath",
                str(temporary_output),
                "-LanguageTag",
                LANGUAGE_TAG,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if completed.returncode != 0 or not temporary_output.is_file():
            detail = (completed.stderr or completed.stdout or "OCR produced no output")[:1000]
            raise ValueError(f"Windows OCR failed with exit {completed.returncode}: {detail}")
        payload = temporary_output.read_bytes()
        text = payload.decode("utf-8-sig")
        if len(re.sub(r"\s+", "", text)) < 100:
            raise ValueError("Windows OCR output is implausibly short")
        _atomic_bytes(text.encode("utf-8"), output_path)
        return text


def run(*, assets: list[str] | None = None) -> dict[str, Any]:
    index, inputs = _authenticate_document_index()
    targets = index[index["text_status"].eq("no_extractable_text")].copy()
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        targets = targets[targets["asset"].astype(str).str.zfill(6).isin(requested)]
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        pdf_path = ROOT / str(row.pdf_path)
        pdf_hash = str(row.pdf_sha256)
        base = {
            "asset": str(row.asset),
            "announcement_date": str(row.announcement_date),
            "announcement_title": str(row.announcement_title),
            "source_url": str(row.source_url),
            "pdf_path": str(row.pdf_path),
            "pdf_sha256": pdf_hash,
            "page_count": int(row.page_count),
            "ocr_engine": "Windows.Media.Ocr.OcrEngine",
            "ocr_language": LANGUAGE_TAG,
            "render_scale": RENDER_SCALE,
            "ocr_script_sha256": _sha256(OCR_SCRIPT_PATH),
            "historical_backtest_allowed": False,
        }
        try:
            if not pdf_path.is_file() or _sha256(pdf_path) != pdf_hash:
                raise ValueError("source PDF path or hash mismatch")
            output_path = documents.TEXT_DIR / f"{pdf_hash}.ocr.txt"
            text = _ocr_pdf(pdf_path, output_path)
            text_hash = _sha256(output_path)
            rows.append(
                {
                    **base,
                    "ocr_status": "ocr_derived_unvalidated",
                    "ocr_text_path": _relative(output_path),
                    "ocr_text_sha256": text_hash,
                    "ocr_text_characters": len(text),
                    "field_validation_status": "not_started",
                    "error": "",
                }
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            rows.append(
                {
                    **base,
                    "ocr_status": "failed",
                    "ocr_text_path": "",
                    "ocr_text_sha256": "",
                    "ocr_text_characters": 0,
                    "field_validation_status": "blocked_ocr_failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:800]}",
                }
            )
    ocr_index = pd.DataFrame(rows, columns=OCR_COLUMNS)
    _atomic_csv(ocr_index, OCR_INDEX_PATH)
    success = int(ocr_index["ocr_status"].eq("ocr_derived_unvalidated").sum()) if not ocr_index.empty else 0
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": "OCR_SIDECAR_COMPLETE_REQUIRES_FIELD_VALIDATION" if success == len(targets) else "OCR_PARTIAL_OR_FAILED",
        "target_documents": int(len(targets)),
        "ocr_success_documents": success,
        "ocr_failed_documents": int(len(targets) - success),
        "ocr_engine": "Windows.Media.Ocr.OcrEngine",
        "ocr_language": LANGUAGE_TAG,
        "render_scale": RENDER_SCALE,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": "OCR text is derived evidence and must be reconciled against the original PDF and independent official documents.",
    }
    _atomic_json(report, REPORT_PATH)
    outputs = [
        {"role": "ocr_index", "path": _relative(OCR_INDEX_PATH), "sha256": _sha256(OCR_INDEX_PATH), "rows": int(len(ocr_index))},
        {"role": "ocr_report", "path": _relative(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    ]
    for row in ocr_index.itertuples(index=False):
        if row.ocr_status == "ocr_derived_unvalidated":
            path = ROOT / str(row.ocr_text_path)
            role_suffix = hashlib.sha256(str(row.source_url).encode("utf-8")).hexdigest()[:16]
            outputs.append(
                {"role": f"ocr_text:{row.asset}:{role_suffix}", "path": _relative(path), "sha256": _sha256(path)}
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(assets=args.asset)
    keys = ("qualification_status", "target_documents", "ocr_success_documents", "ocr_failed_documents")
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
