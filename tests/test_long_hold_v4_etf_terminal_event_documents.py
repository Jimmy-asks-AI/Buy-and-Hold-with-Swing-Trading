import hashlib
import json

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_terminal_event_document_collector as documents
from strategy_lab.long_hold_v4 import pit_etf_terminal_event_document_merge as document_merge


def _announcement(
    asset: str,
    date: str,
    title: str,
    tags: list[str],
    event_types: list[str],
    suffix: str,
) -> dict[str, object]:
    return {
        "asset": asset,
        "asset_name": f"ETF {asset}",
        "exchange": "SZSE" if asset.startswith("1") else "SSE",
        "announcement_date": date,
        "announcement_title": title,
        "source_url": f"https://static.cninfo.com.cn/{suffix}.PDF" if asset.startswith("1") else f"https://www.sse.com.cn/{suffix}.pdf",
        "source_type": "official",
        "matched_keywords_json": "[]",
        "title_tags_json": json.dumps(tags),
        "candidate_event_types_json": json.dumps(event_types),
        "source_observed_at": "2026-07-19T12:00:00+08:00",
        "available_date": date,
        "data_source": "official",
        "source_vintage": "fixture",
        "document_validation_status": "not_started",
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }


def test_document_selection_keeps_cash_chain_and_excludes_redundant_prompts():
    announcements = pd.DataFrame(
        [
            _announcement(
                "510700",
                "2013-05-17",
                "上市初期基金份额折算结果公告",
                ["successor_share_conversion"],
                ["conversion_to_successor"],
                "old-fold",
            ),
            _announcement(
                "510700",
                "2015-10-20",
                "基金合同终止及基金财产清算的公告",
                ["fund_contract_termination", "liquidation"],
                ["cash_liquidation_or_extinguishment"],
                "mechanism",
            ),
            _announcement(
                "510700",
                "2015-10-21",
                "基金合同终止及基金财产清算的第一次提示性公告",
                ["fund_contract_termination", "liquidation"],
                ["cash_liquidation_or_extinguishment"],
                "prompt-1",
            ),
            _announcement(
                "510700",
                "2015-12-04",
                "基金清算报告",
                ["liquidation"],
                ["cash_liquidation_or_extinguishment"],
                "report",
            ),
            _announcement(
                "510700",
                "2015-12-12",
                "清算资金发放公告",
                ["liquidation"],
                ["cash_liquidation_or_extinguishment"],
                "cash",
            ),
            _announcement(
                "510700",
                "2016-05-10",
                "终止上市公告",
                ["delisting"],
                ["exchange_delisting"],
                "delist",
            ),
            _announcement(
                "510700",
                "2016-05-12",
                "终止上市的提示性公告",
                ["delisting"],
                ["exchange_delisting"],
                "delist-prompt",
            ),
        ],
        columns=documents.universe.ANNOUNCEMENT_COLUMNS,
    )
    coverage = pd.DataFrame(
        [
            {
                "asset": "510700",
                "master_delist_date": "2016-05-12",
                "primary_candidate_class": "cash_or_extinguishment_candidate",
            }
        ]
    )
    selected = documents.select_documents(announcements, coverage)
    assert set(selected["announcement_title"]) == {
        "基金合同终止及基金财产清算的公告",
        "基金清算报告",
        "清算资金发放公告",
        "终止上市公告",
    }
    assert "上市初期基金份额折算结果公告" not in set(selected["announcement_title"])


def test_successor_window_does_not_select_an_old_listing_fold():
    announcements = pd.DataFrame(
        [
            _announcement(
                "159917",
                "2012-04-01",
                "上市前基金份额折算结果公告",
                ["successor_share_conversion"],
                ["conversion_to_successor"],
                "old-fold",
            ),
            _announcement(
                "159917",
                "2015-08-17",
                "关于实施转型及份额转换的公告",
                ["transformation", "successor_share_conversion"],
                ["conversion_to_successor"],
                "terminal-conversion",
            ),
            _announcement(
                "159917",
                "2015-08-25",
                "终止上市的公告",
                ["delisting"],
                ["exchange_delisting"],
                "delisting",
            ),
        ],
        columns=documents.universe.ANNOUNCEMENT_COLUMNS,
    )
    coverage = pd.DataFrame(
        [
            {
                "asset": "159917",
                "master_delist_date": "2015-08-26",
                "primary_candidate_class": "successor_share_candidate",
            }
        ]
    )
    selected = documents.select_documents(announcements, coverage)
    assert "上市前基金份额折算结果公告" not in set(selected["announcement_title"])
    assert set(selected["announcement_title"]) == {
        "关于实施转型及份额转换的公告",
        "终止上市的公告",
    }


def test_no_text_pdf_cache_is_valid_but_remains_reviewable(monkeypatch, tmp_path):
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    monkeypatch.setattr(documents, "DOCUMENT_META_DIR", tmp_path / "metadata")
    asset = "510700"
    source_url = "https://www.sse.com.cn/example.pdf"
    pdf_path = tmp_path / "documents" / "evidence.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-fixture")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": documents.SCHEMA_VERSION,
        "asset": asset,
        "source_url": source_url,
        "producer_code_sha256": documents._sha256(documents.Path(documents.__file__).resolve()),
        "collection_status": "success",
        "pdf_path": "documents/evidence.pdf",
        "pdf_sha256": digest,
        "text_status": "no_extractable_text",
        "text_path": "",
        "text_sha256": "",
    }
    meta_path = documents._meta_path(asset, source_url)
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert documents._valid_document_cache(asset, source_url) is not None

    pdf_path.write_bytes(pdf_path.read_bytes() + b"tamper")
    assert documents._valid_document_cache(asset, source_url) is None


def test_document_merge_uses_ocr_sidecar_without_promoting_it(monkeypatch, tmp_path):
    monkeypatch.setattr(document_merge, "ROOT", tmp_path)
    pdf_path = tmp_path / "evidence.pdf"
    ocr_path = tmp_path / "evidence.ocr.txt"
    pdf_path.write_bytes(b"%PDF-scanned")
    ocr_path.write_text("扫描文本" * 100, encoding="utf-8")
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    text_hash = hashlib.sha256(ocr_path.read_bytes()).hexdigest()
    base_row = {column: "" for column in documents.DOCUMENT_INDEX_COLUMNS}
    base_row.update(
        {
            "asset": "512340",
            "source_url": "https://www.sse.com.cn/scanned.pdf",
            "announcement_date": "2020-01-01",
            "pdf_path": "evidence.pdf",
            "pdf_sha256": pdf_hash,
            "text_status": "no_extractable_text",
            "text_path": "",
            "text_sha256": "",
            "historical_backtest_allowed": False,
            "model_promotion_allowed": False,
        }
    )
    sidecar = pd.DataFrame(
        [
            {
                "asset": "512340",
                "source_url": base_row["source_url"],
                "pdf_sha256": pdf_hash,
                "ocr_status": "ocr_derived_unvalidated",
                "ocr_text_path": "evidence.ocr.txt",
                "ocr_text_sha256": text_hash,
                "ocr_text_characters": 400,
                "field_validation_status": "not_started",
            }
        ]
    )
    merged, _ = document_merge.build_merged_index(pd.DataFrame([base_row]), sidecar)
    assert merged.iloc[0]["text_status"] == "ocr_derived_unvalidated"
    assert merged.iloc[0]["document_validation_status"] == "ocr_field_validation_not_started"
    assert not bool(merged.iloc[0]["historical_backtest_allowed"])
    assert not bool(merged.iloc[0]["model_promotion_allowed"])
