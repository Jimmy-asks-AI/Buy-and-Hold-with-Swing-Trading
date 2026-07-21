import hashlib

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_terminal_event_candidate_parser as parser
from strategy_lab.long_hold_v4 import pit_etf_terminal_event_validator_v2 as validator


def _write_evidence(root, stem: str, text: str):
    pdf = root / f"{stem}.pdf"
    txt = root / f"{stem}.txt"
    pdf.write_bytes(b"%PDF-fixture-" + stem.encode())
    txt.write_text(text, encoding="utf-8")
    return {
        "pdf_path": pdf.name,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "text_path": txt.name,
        "text_sha256": hashlib.sha256(txt.read_bytes()).hexdigest(),
    }


def test_strict_direct_parser_rejects_restatement_and_handles_unit_wording():
    text = validator._compact(
        "此前进行了第一次剩余财产分配，每份基金份额实际发放资金为1.2000元。"
        "本次每单位基金份额分配的剩余财产为0.0040元。"
    )
    values = validator.strict_direct_values(text)
    best = min(rank for _, rank in values)
    assert [value for value, rank in values if rank == best] == [validator.Decimal("0.0040")]


def test_validator_builds_a_pit_event_and_keeps_promotion_boundaries(monkeypatch, tmp_path):
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    cash_text = validator._compact(
        "本次剩余财产分配权益登记日为2024年8月19日，"
        "资金发放日为2024年8月20日。"
        "本次每份基金份额实际发放资金为1.2000元。"
        "本基金基金合同自2024年8月20日终止，将办理退出登记。"
    )
    cash_files = _write_evidence(tmp_path, "cash", cash_text)
    delist_files = _write_evidence(
        tmp_path,
        "delist",
        validator._compact("本基金终止上市日为2024年8月21日。"),
    )
    cash = pd.DataFrame(
        [
            {
                "candidate_event_id": "candidate-1",
                "asset": "510000",
                "asset_name": "Fixture ETF",
                "exchange": "SSE",
                "announcement_date": "2024-08-16",
                "published_at": "",
                "available_at": "2024-08-17T00:00:00+08:00",
                "available_trade_date": "2024-08-17",
                "record_date": "2024-08-19",
                "ex_date": "",
                "pay_date": "2024-08-20",
                "direct_cash_per_share": 1.2,
                "distribution_total_cash": None,
                "additional_distribution_expected": False,
                "fund_contract_terminated": True,
                "exit_registration_announced": True,
                "source_url": "https://www.sse.com.cn/cash.pdf",
                "text_status": "success",
                **cash_files,
            }
        ]
    )
    delisting = pd.DataFrame(
        [
            {
                "asset": "510000",
                "termination_date": "2024-08-21",
                "termination_date_parse_status": "unique",
                "candidate_status": "field_candidate_complete_requires_independent_validation",
                "source_url": "https://www.sse.com.cn/delist.pdf",
                **delist_files,
            }
        ],
        columns=parser.DELISTING_COLUMNS,
    )
    checks, chains, promotion = validator._validate_candidates(
        cash,
        pd.DataFrame(columns=parser.LIQUIDATION_COLUMNS),
        delisting,
    )
    assert checks.iloc[0]["validation_status"] == "pass"
    assert chains.iloc[0]["chain_status"] == "complete"
    assert len(promotion) == 1
    event = promotion.iloc[0]
    assert event["accounting_date"] == pd.Timestamp("2024-08-20")
    assert bool(event["extinguishes_position"])
    assert bool(event["historical_backtest_allowed"])
    assert not bool(event["model_promotion_allowed"])


def test_ocr_cash_text_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    evidence = _write_evidence(
        tmp_path,
        "ocr",
        validator._compact(
            "权益登记日为2024年8月19日，发放日为2024年8月20日。"
            "本次每份基金份额实际发放资金为1.2元。"
        ),
    )
    cash = pd.DataFrame(
        [
            {
                "candidate_event_id": "ocr-candidate",
                "asset": "510001",
                "asset_name": "OCR ETF",
                "exchange": "SSE",
                "announcement_date": "2024-08-16",
                "published_at": "",
                "available_at": "2024-08-17T00:00:00+08:00",
                "available_trade_date": "2024-08-17",
                "record_date": "2024-08-19",
                "ex_date": "",
                "pay_date": "2024-08-20",
                "direct_cash_per_share": 1.2,
                "distribution_total_cash": None,
                "additional_distribution_expected": False,
                "fund_contract_terminated": True,
                "exit_registration_announced": True,
                "source_url": "https://www.sse.com.cn/ocr.pdf",
                "text_status": "ocr_derived_unvalidated",
                **evidence,
            }
        ]
    )
    checks, _, promotion = validator._validate_candidates(
        cash,
        pd.DataFrame(columns=parser.LIQUIDATION_COLUMNS),
        pd.DataFrame(columns=parser.DELISTING_COLUMNS),
    )
    assert checks.iloc[0]["validation_status"] == "fail"
    assert "text_not_native_or_unavailable" in checks.iloc[0]["failure_reasons_json"]
    assert promotion.empty
