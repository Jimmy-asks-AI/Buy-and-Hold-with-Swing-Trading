import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pymupdf

from strategy_lab.long_hold_v4 import pit_etf_benchmark_document_collector as collector
from strategy_lab.long_hold_v4 import pit_etf_benchmark_document_selector as selector


class _Response:
    def __init__(self, content, content_type="application/pdf"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self, content, content_type="application/pdf"):
        self.content = content
        self.content_type = content_type

    def get(self, *_args, **_kwargs):
        return _Response(self.content, self.content_type)


def _pdf_bytes():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Benchmark index CSI 300")
    payload = document.tobytes()
    document.close()
    return payload


def _html_bytes():
    return (
        "<html><head><meta charset='gb2312'></head><body>"
        "<script>title();</script><h1>关于修改基金合同条款的公告</h1>"
        "<p>本次修订不涉及标的指数变更，公告内容用于独立核验。</p>"
        "</body></html>"
    ).encode("gb18030")


def _row(source_url="https://static.cninfo.com.cn/finalpage/test.PDF"):
    key = hashlib.sha256(f"510050|{source_url}".encode("utf-8")).hexdigest()
    return SimpleNamespace(asset="510050", exchange="SSE", source_url=source_url, document_key=key)


class BenchmarkDocumentCollectorTests(unittest.TestCase):
    def test_collects_content_addressed_pdf_and_native_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(collector, "DOCUMENT_DIR", root / "documents"),
                patch.object(collector, "TEXT_DIR", root / "text"),
                patch.object(collector, "DOCUMENT_META_DIR", root / "metadata"),
            ):
                row = _row()
                metadata = collector.collect_document(_Session(_pdf_bytes()), row)
                self.assertEqual(metadata["collection_status"], "success")
                self.assertEqual(metadata["text_status"], "success")
                self.assertGreater(metadata["page_count"], 0)
                text_path = Path(metadata["text_path"])
                self.assertTrue(text_path.is_file())
                self.assertIn("Benchmark index CSI 300", text_path.read_text(encoding="utf-8"))
                self.assertIsNotNone(collector._valid_document_cache(row))

    def test_rejects_unsupported_payload(self):
        with self.assertRaisesRegex(ValueError, "format is unsupported"):
            collector.collect_document(_Session(b"not-a-document"), _row())

    def test_collects_official_legacy_html_with_detected_chinese_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(collector, "DOCUMENT_DIR", root / "documents"),
                patch.object(collector, "TEXT_DIR", root / "text"),
                patch.object(collector, "DOCUMENT_META_DIR", root / "metadata"),
            ):
                row = _row("https://static.cninfo.com.cn/finalpage/test.html")
                metadata = collector.collect_document(
                    _Session(_html_bytes(), "text/html; charset=UTF-8"),
                    row,
                )
                self.assertEqual(metadata["document_format"], "html")
                self.assertEqual(metadata["text_status"], "success")
                self.assertEqual(metadata["pdf_path"], "")
                self.assertTrue(metadata["raw_document_path"].endswith(".html"))
                text_path = Path(metadata["text_path"])
                self.assertIn("不涉及标的指数变更", text_path.read_text(encoding="utf-8"))
                self.assertIsNotNone(collector._valid_document_cache(row))

    def test_rejects_unofficial_domain_before_request(self):
        with self.assertRaisesRegex(ValueError, "unsupported official"):
            collector.collect_document(_Session(_pdf_bytes()), _row("https://example.com/a.pdf"))

    def test_selection_validation_keeps_evidence_gate_closed(self):
        row = {column: "" for column in selector.SELECTION_COLUMNS}
        row.update(
            {
                "asset": "510050",
                "source_url": "https://www.sse.com.cn/a.pdf",
                "document_key": "a" * 64,
                "selection_policy_version": selector.SELECTION_POLICY_VERSION,
                "selection_priority_rank": 0,
                "selection_priority": "P0",
                "announcement_date": "2005-02-01",
                "baseline_selection_state": "preferred_initial_candidate",
                "historical_backtest_allowed": False,
                "model_promotion_allowed": False,
            }
        )
        manifest = {"selected_documents": 1}
        validated = collector._validate_selection(pd.DataFrame([row]), manifest)
        self.assertEqual(len(validated), 1)
        row["historical_backtest_allowed"] = True
        with self.assertRaisesRegex(ValueError, "historical use"):
            collector._validate_selection(pd.DataFrame([row]), manifest)

    def test_collection_scope_can_select_one_document_reason(self):
        selection = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "selection_priority": "P1",
                    "selection_reasons_json": '["one_initial_prospectus_supplement_document"]',
                },
                {
                    "asset": "510050",
                    "selection_priority": "P1",
                    "selection_reasons_json": '["one_canonical_listing_context_document"]',
                },
                {
                    "asset": "159001",
                    "selection_priority": "P0",
                    "selection_reasons_json": '["one_preferred_initial_legal_document"]',
                },
            ]
        )
        scoped = collector._filter_collection_scope(
            selection,
            priorities=["P1"],
            selection_reasons=["one_initial_prospectus_supplement_document"],
        )
        self.assertEqual(len(scoped), 1)
        self.assertIn("prospectus", scoped.iloc[0]["selection_reasons_json"])

    def test_collection_scope_rejects_unknown_document_reason(self):
        selection = pd.DataFrame(
            [
                {
                    "asset": "510050",
                    "selection_priority": "P1",
                    "selection_reasons_json": '["one_initial_prospectus_supplement_document"]',
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "outside ETF benchmark selection"):
            collector._filter_collection_scope(selection, selection_reasons=["missing_reason"])


if __name__ == "__main__":
    unittest.main()
