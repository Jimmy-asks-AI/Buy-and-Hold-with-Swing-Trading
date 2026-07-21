"""Create review-only Windows OCR sidecars for scanned ETF benchmark documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
import pandas as pd

from . import pit_etf_benchmark_document_collector as collector
from . import pit_source_code_archive as source_code_archive


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_MANIFEST_PATH = collector.MANIFEST_PATH
OCR_SCRIPT_PATH = ROOT / "scripts" / "windows_ocr_images.ps1"
OCR_INDEX_PATH = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "observations"
    / "etf_benchmark_document_ocr_index.csv"
)
LINEAGE_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_benchmark_documents"
    / "ocr_lineage"
)
OCR_TEXT_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_benchmark_documents"
    / "text"
)
OCR_WORK_DIR = (
    ROOT
    / "data_raw"
    / "long_hold_v4"
    / "pit_history"
    / "raw_etf_benchmark_documents"
    / "ocr_work"
)
REPORT_PATH = (
    ROOT
    / "outputs"
    / "long_hold_v4"
    / "pit_validation"
    / "etf_benchmark_documents"
    / "ocr_report.json"
)
MANIFEST_PATH = (
    ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_document_windows_ocr_latest.json"
)

SCHEMA_VERSION = 1
RENDER_SCALE = 2.0
LANGUAGE_TAG = "zh-Hans-CN"
ATOMIC_REPLACE_ATTEMPTS = 20
ATOMIC_REPLACE_SLEEP_SECONDS = 0.05
OCR_COLUMNS = [
    "document_key",
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
    "model_promotion_allowed",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
        raise ValueError(f"ETF benchmark OCR lineage snapshot hash mismatch: {snapshot}")
    return snapshot


def _authenticate_document_index() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    manifest = json.loads(COLLECTOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("qualification_status") != "PARTIAL_SELECTED_OFFICIAL_DOCUMENT_SET"
        or int(manifest.get("selected_documents", 0)) < 1701
        or int(manifest.get("collected_documents", 0)) <= 0
        or manifest.get("historical_backtest_allowed") is not False
    ):
        raise ValueError("ETF benchmark document collection does not authorize OCR supplementation")
    producer_path = ROOT / str(manifest.get("code_path", ""))
    authenticated_code = source_code_archive.authenticate_current_or_archive(
        producer_path, str(manifest.get("code_sha256", ""))
    )
    outputs = {str(item.get("role")): item for item in manifest.get("outputs", [])}
    item = outputs.get("benchmark_document_index", {})
    path = ROOT / str(item.get("path", ""))
    if not path.is_file() or _sha256(path) != str(item.get("sha256", "")):
        raise ValueError("ETF benchmark document index hash mismatch")
    manifest_snapshot = _content_snapshot(COLLECTOR_MANIFEST_PATH)
    index_snapshot = _content_snapshot(path)
    frame = pd.read_csv(index_snapshot, dtype={"asset": str}, low_memory=False)
    return frame, [
        {
            "role": "collector_manifest_snapshot",
            "path": _relative(manifest_snapshot),
            "sha256": _sha256(manifest_snapshot),
        },
        {
            "role": "benchmark_document_index_snapshot",
            "path": _relative(index_snapshot),
            "sha256": _sha256(index_snapshot),
        },
        {
            "role": "authenticated_collector_code",
            "path": _relative(authenticated_code),
            "sha256": _sha256(authenticated_code),
        },
        {
            "role": "windows_ocr_script",
            "path": _relative(OCR_SCRIPT_PATH),
            "sha256": _sha256(OCR_SCRIPT_PATH),
        },
    ]


def _ocr_pdf(pdf_path: Path, output_path: Path) -> str:
    if output_path.is_file():
        cached = output_path.read_text(encoding="utf-8")
        if len(re.sub(r"\s+", "", cached)) >= 100:
            return cached
    OCR_WORK_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark_ocr_", dir=OCR_WORK_DIR) as temporary:
        temporary_path = Path(temporary)
        with fitz.open(pdf_path) as pdf:
            for number, page in enumerate(pdf, start=1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE), alpha=False)
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
            timeout=900,
            check=False,
        )
        if completed.returncode != 0 or not temporary_output.is_file():
            detail = (completed.stderr or completed.stdout or "OCR produced no output")[:1000]
            raise ValueError(f"Windows OCR failed with exit {completed.returncode}: {detail}")
        text = temporary_output.read_bytes().decode("utf-8-sig")
        if len(re.sub(r"\s+", "", text)) < 100:
            raise ValueError("Windows OCR output is implausibly short")
        _atomic_bytes(text.encode("utf-8"), output_path)
        return text


def run(*, assets: list[str] | None = None) -> dict[str, Any]:
    index, inputs = _authenticate_document_index()
    targets = index[
        index["collection_status"].eq("success")
        & index["text_status"].eq("no_extractable_text")
    ].copy()
    if assets:
        requested = {str(asset).zfill(6) for asset in assets}
        targets = targets[targets["asset"].astype(str).str.zfill(6).isin(requested)]
    targets = targets.sort_values(["asset", "announcement_date", "document_key"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        pdf_path = ROOT / str(row.pdf_path)
        pdf_hash = str(row.pdf_sha256)
        base = {
            "document_key": str(row.document_key),
            "asset": str(row.asset).zfill(6),
            "announcement_date": str(row.announcement_date),
            "announcement_title": str(row.announcement_title),
            "source_url": str(row.source_url),
            "pdf_path": str(row.pdf_path),
            "pdf_sha256": pdf_hash,
            "page_count": int(float(row.page_count)),
            "ocr_engine": "Windows.Media.Ocr.OcrEngine",
            "ocr_language": LANGUAGE_TAG,
            "render_scale": RENDER_SCALE,
            "ocr_script_sha256": _sha256(OCR_SCRIPT_PATH),
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
        try:
            if not pdf_path.is_file() or _sha256(pdf_path) != pdf_hash:
                raise ValueError("source PDF path or hash mismatch")
            output_path = OCR_TEXT_DIR / f"{pdf_hash}.ocr.txt"
            text = _ocr_pdf(pdf_path, output_path)
            rows.append(
                {
                    **base,
                    "ocr_status": "ocr_derived_unvalidated",
                    "ocr_text_path": _relative(output_path),
                    "ocr_text_sha256": _sha256(output_path),
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
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "qualification_status": (
            "OCR_SIDECAR_COMPLETE_REQUIRES_FIELD_VALIDATION"
            if success == len(targets)
            else "OCR_PARTIAL_OR_FAILED"
        ),
        "target_documents": int(len(targets)),
        "ocr_success_documents": success,
        "ocr_failed_documents": int(len(targets) - success),
        "ocr_engine": "Windows.Media.Ocr.OcrEngine",
        "ocr_language": LANGUAGE_TAG,
        "render_scale": RENDER_SCALE,
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
        "boundary": (
            "OCR is a derived sidecar. It can route field review but cannot promote a benchmark fact without "
            "original-page or independent official-text confirmation."
        ),
    }
    _atomic_json(report, REPORT_PATH)
    outputs: list[dict[str, Any]] = [
        {
            "role": "benchmark_document_ocr_index",
            "path": _relative(OCR_INDEX_PATH),
            "sha256": _sha256(OCR_INDEX_PATH),
            "rows": int(len(ocr_index)),
        },
        {
            "role": "benchmark_document_ocr_report",
            "path": _relative(REPORT_PATH),
            "sha256": _sha256(REPORT_PATH),
        },
    ]
    for row in ocr_index.itertuples(index=False):
        if row.ocr_status == "ocr_derived_unvalidated":
            path = ROOT / str(row.ocr_text_path)
            outputs.append(
                {
                    "role": f"ocr_text:{row.document_key}",
                    "path": _relative(path),
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        **report,
        "inputs": inputs,
        "outputs": outputs,
        "code_path": _relative(Path(__file__).resolve()),
        "code_sha256": _sha256(Path(__file__).resolve()),
        "code_dependencies": [
            {
                "path": _relative(Path(collector.__file__).resolve()),
                "sha256": _sha256(Path(collector.__file__).resolve()),
            },
            {
                "path": _relative(Path(source_code_archive.__file__).resolve()),
                "sha256": _sha256(Path(source_code_archive.__file__).resolve()),
            },
            {"path": _relative(OCR_SCRIPT_PATH), "sha256": _sha256(OCR_SCRIPT_PATH)},
        ],
        "current_final_snapshot": True,
        "contains_validated_benchmark_facts": False,
    }
    _atomic_json(manifest, MANIFEST_PATH)
    return manifest


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--asset", action="append")
    return argument_parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(assets=args.asset)
    keys = ("qualification_status", "target_documents", "ocr_success_documents", "ocr_failed_documents")
    print(json.dumps({key: result[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
