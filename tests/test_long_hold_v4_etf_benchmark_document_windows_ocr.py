import hashlib

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_benchmark_document_windows_ocr as ocr


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_creates_review_only_sidecar_for_scanned_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"mock-pdf")
    script = tmp_path / "ocr.ps1"
    script.write_text("# mock", encoding="utf-8")
    index = pd.DataFrame(
        [
            {
                "document_key": "a" * 64,
                "asset": "510220",
                "announcement_date": "2022-11-19",
                "announcement_title": "持有人大会决议",
                "source_url": "https://example.test/scan.pdf",
                "collection_status": "success",
                "text_status": "no_extractable_text",
                "pdf_path": str(pdf),
                "pdf_sha256": _sha256(pdf),
                "page_count": 5,
            },
            {
                "document_key": "b" * 64,
                "asset": "510050",
                "announcement_date": "2005-02-01",
                "announcement_title": "基金合同",
                "source_url": "https://example.test/native.pdf",
                "collection_status": "success",
                "text_status": "success",
                "pdf_path": str(pdf),
                "pdf_sha256": _sha256(pdf),
                "page_count": 1,
            },
        ]
    )

    monkeypatch.setattr(ocr, "OCR_SCRIPT_PATH", script)
    monkeypatch.setattr(ocr, "OCR_INDEX_PATH", tmp_path / "ocr_index.csv")
    monkeypatch.setattr(ocr, "OCR_TEXT_DIR", tmp_path / "text")
    monkeypatch.setattr(ocr, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(ocr, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(ocr, "_authenticate_document_index", lambda: (index, []))

    def fake_ocr(_pdf_path, output_path):
        text = "扫描文本" * 100
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        return text

    monkeypatch.setattr(ocr, "_ocr_pdf", fake_ocr)
    result = ocr.run()

    output = pd.read_csv(ocr.OCR_INDEX_PATH, dtype={"asset": str})
    assert result["qualification_status"] == "OCR_SIDECAR_COMPLETE_REQUIRES_FIELD_VALIDATION"
    assert result["target_documents"] == 1
    assert len(output) == 1
    assert output.iloc[0]["ocr_status"] == "ocr_derived_unvalidated"
    assert not bool(output.iloc[0]["historical_backtest_allowed"])
    assert not bool(output.iloc[0]["model_promotion_allowed"])
