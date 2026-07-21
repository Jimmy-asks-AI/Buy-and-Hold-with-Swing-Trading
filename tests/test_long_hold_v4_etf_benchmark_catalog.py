import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_official_announcement_catalog_collector as catalog


class _Response:
    def __init__(self, payload, url="https://example.test/query"):
        self._payload = payload
        self.url = url

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def get(self, *_args, **_kwargs):
        return _Response(self.payloads.pop(0))

    def post(self, *_args, **_kwargs):
        return _Response(self.payloads.pop(0))


class BenchmarkCatalogTests(unittest.TestCase):
    def test_title_classifier_routes_initial_and_change_documents(self):
        tags, roles = catalog.classify_benchmark_title("某ETF上市交易公告书")
        self.assertIn("listing_document", tags)
        self.assertEqual(roles, ["initial_benchmark_candidate"])

        tags, roles = catalog.classify_benchmark_title("关于变更标的指数并修订基金合同的公告")
        self.assertIn("explicit_benchmark_change", tags)
        self.assertIn("benchmark_change_candidate", roles)
        self.assertIn("contract_content_review_candidate", roles)

    def test_generic_title_does_not_become_negative_evidence(self):
        tags, roles = catalog.classify_benchmark_title("某基金2025年年度报告")
        self.assertEqual(tags, [])
        self.assertEqual(roles, [])

    def test_sse_fetch_requires_every_page_and_exact_total(self):
        target = SimpleNamespace(asset="510050", asset_name="50ETF", exchange="SSE")
        session = _Session(
            [
                {"result": [{"SECURITY_CODE": "510050"}], "pageHelp": {"pageCount": 2, "total": 2}},
                {"result": [{"SECURITY_CODE": "510050"}], "pageHelp": {"pageCount": 2, "total": 2}},
            ]
        )
        artifact = catalog._fetch_sse_catalog(session, target, pd.Timestamp("2026-07-19"))
        self.assertEqual(artifact["pages"], [1, 2])
        self.assertEqual(artifact["total_rows"], 2)
        self.assertEqual(len(artifact["requests"]), 2)

    def test_cninfo_fetch_rejects_incomplete_pagination(self):
        target = SimpleNamespace(asset="159901", asset_name="深证100ETF", exchange="SZSE")
        first_page = [
            {"secCode": "159901", "announcementId": str(index)} for index in range(30)
        ]
        incomplete_pass = [
            {"announcements": first_page, "totalpages": 1, "totalAnnouncement": 31},
            {"announcements": [], "totalpages": 1, "totalAnnouncement": 31},
        ]
        session = _Session(incomplete_pass * catalog.CNINFO_MAX_PASSES)
        with patch.object(catalog, "_resolve_cninfo_identity", return_value={"org_id": "fund-id"}):
            with self.assertRaisesRegex(ValueError, "did not stabilize"):
                catalog._fetch_cninfo_catalog(session, target, pd.Timestamp("2026-07-19"))

    def test_cninfo_fetch_repairs_zero_based_totalpages(self):
        target = SimpleNamespace(asset="159901", asset_name="深证100ETF", exchange="SZSE")
        first_page = [
            {"secCode": "159901", "announcementId": str(index)} for index in range(30)
        ]
        final_page = [{"secCode": "159901", "announcementId": "30"}]
        session = _Session(
            [
                {"announcements": first_page, "totalpages": 1, "totalAnnouncement": 31},
                {"announcements": final_page, "totalpages": 1, "totalAnnouncement": 31},
                {"announcements": first_page, "totalpages": 1, "totalAnnouncement": 31},
                {"announcements": final_page, "totalpages": 1, "totalAnnouncement": 31},
            ]
        )
        with patch.object(catalog, "_resolve_cninfo_identity", return_value={"org_id": "fund-id"}):
            artifact = catalog._fetch_cninfo_catalog(session, target, pd.Timestamp("2026-07-19"))
        self.assertEqual(artifact["page_count"], 2)
        self.assertEqual(artifact["pages"], [1, 2])
        self.assertEqual(len(artifact["rows"]), 31)
        self.assertEqual(artifact["pagination_reconciliation"]["passes"], 2)

    def test_cninfo_fetch_unions_unstable_boundary_rows(self):
        target = SimpleNamespace(asset="159986", asset_name="豆粕ETF", exchange="SZSE")
        common = [
            {"secCode": "159986", "announcementId": str(index), "announcementTime": index}
            for index in range(30)
        ]
        first_variant = common + [
            {"secCode": "159986", "announcementId": "A", "announcementTime": 100}
        ]
        second_variant = common[:-1] + [
            {"secCode": "159986", "announcementId": "B", "announcementTime": 101},
            {"secCode": "159986", "announcementId": "A", "announcementTime": 100},
        ]
        payloads = []
        for rows in (first_variant, second_variant, second_variant):
            payloads.extend(
                [
                    {"announcements": rows[:30], "totalpages": 1, "totalAnnouncement": 32},
                    {"announcements": rows[30:], "totalpages": 1, "totalAnnouncement": 32},
                ]
            )
        session = _Session(payloads)
        with patch.object(catalog, "_resolve_cninfo_identity", return_value={"org_id": "fund-id"}):
            artifact = catalog._fetch_cninfo_catalog(session, target, pd.Timestamp("2026-07-19"))
        identities = {catalog._cninfo_row_identity(row) for row in artifact["rows"]}
        self.assertIn("id:A", identities)
        self.assertIn("id:B", identities)
        self.assertEqual(artifact["pagination_reconciliation"]["passes"], 3)

        metrics = catalog.summarize_cninfo_reconciliation([artifact])
        self.assertEqual(metrics["cninfo_assets_requiring_more_than_minimum_passes"], 1)
        self.assertEqual(metrics["cninfo_assets_requiring_union_reconciliation"], 1)
        self.assertEqual(metrics["cninfo_union_increment_rows"], 1)
        self.assertEqual(metrics["cninfo_union_rows_above_maximum_reported_total"], 0)

    def test_parse_uses_next_day_availability_and_filters_future_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            query_dir = Path(directory)
            data_path = query_dir / "sse" / "510050.json"
            data_path.parent.mkdir(parents=True)
            data_path.write_text("{}", encoding="utf-8")
            artifact = {
                "asset": "510050",
                "asset_name": "50ETF",
                "exchange": "SSE",
                "as_of_date": "2026-07-19",
                "fetched_at": "2026-07-19T10:00:00+08:00",
                "rows": [
                    {
                        "SECURITY_CODE": "510050",
                        "TITLE": "50ETF上市交易公告书",
                        "URL": "/a.pdf",
                        "SSEDATE": "2026-07-18",
                    },
                    {
                        "SECURITY_CODE": "510050",
                        "TITLE": "未来公告",
                        "URL": "/future.pdf",
                        "SSEDATE": "2026-07-19",
                    },
                ],
            }
            with patch.object(catalog, "ROOT", query_dir), patch.object(catalog, "QUERY_DIR", query_dir):
                frame = catalog.parse_query_artifacts([artifact])
            self.assertEqual(len(frame), 1)
            self.assertEqual(str(frame.iloc[0]["available_date"].date()), "2026-07-19")
            self.assertFalse(bool(frame.iloc[0]["historical_backtest_allowed"]))

    def test_discovery_registry_keeps_all_assets_evidence_insufficient(self):
        targets = pd.DataFrame(
            [
                {"asset": "510050", "asset_name": "50ETF", "exchange": "SSE", "list_date": "2005-02-23", "delist_date": None},
                {"asset": "159901", "asset_name": "深证100ETF", "exchange": "SZSE", "list_date": "2006-04-24", "delist_date": None},
            ]
        )
        catalog_frame = pd.DataFrame(
            [{"asset": "510050", "announcement_date": pd.Timestamp("2005-02-20")}]
        )
        candidates = pd.DataFrame(
            [{"asset": "510050", "candidate_roles_json": json.dumps(["initial_benchmark_candidate"])}]
        )
        result = catalog.build_coverage_registry(targets, catalog_frame, candidates, {"510050"})
        complete = result[result["asset"].eq("510050")].iloc[0]
        pending = result[result["asset"].eq("159901")].iloc[0]
        self.assertEqual(complete["initial_benchmark_evidence_state"], "evidence_insufficient")
        self.assertFalse(bool(complete["benchmark_history_complete"]))
        self.assertEqual(pending["discovery_state"], "query_incomplete")

    def test_atomic_write_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            calls = 0

            def replace_with_one_lock(path_self, target_path):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("transient reader lock")
                os.replace(path_self, target_path)
                return Path(target_path)

            with patch.object(type(temporary), "replace", autospec=True, side_effect=replace_with_one_lock), patch.object(
                catalog, "ATOMIC_REPLACE_SLEEP_SECONDS", 0
            ):
                catalog._atomic_bytes(b"complete", target)
            self.assertEqual(calls, 2)
            self.assertEqual(target.read_bytes(), b"complete")


if __name__ == "__main__":
    unittest.main()
