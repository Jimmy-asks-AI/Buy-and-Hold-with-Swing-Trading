from __future__ import annotations

from strategy_lab.long_hold_v4.pit_etf_source_qualification import build_qualification_checks


def _fixtures() -> tuple[dict, dict, dict]:
    recent = {"recent_cross_source_passed": True, "qualification_status": "recent-pass"}
    tencent = {
        "cross_source_content_passed": True,
        "selected_assets": 123,
        "selected_delisted_assets": 123,
        "ready_assets": 123,
        "full_market_current_final_source_passed": False,
        "version_monitoring_ready": False,
        "version_depth_coverage": 0.0,
    }
    eastmoney = {
        "selected_assets": 1701,
        "ready_assets": 1701,
        "full_market_current_final_source_passed": True,
        "version_depth_coverage": 0.0,
        "terminal_event_boundary": {"boundary_passed": True, "last_nav_date": "2017-11-07"},
    }
    return recent, tencent, eastmoney


def test_qualification_checks_keep_content_and_pit_evidence_separate() -> None:
    checks = build_qualification_checks(*_fixtures()).set_index("check")
    assert bool(checks.loc["recent_joinquant_cross_source_content", "passed"])
    assert bool(checks.loc["delisted_price_independent_coverage", "passed"])
    assert not bool(checks.loc["full_market_price_independent_coverage", "passed"])
    assert bool(checks.loc["full_lifecycle_nav_independent_coverage", "passed"])
    assert not bool(checks.loc["historical_available_date_evidence", "passed"])
    assert checks.loc["historical_available_date_evidence", "evidence_class"] == "pit_history"


def test_full_current_final_coverage_still_does_not_pass_pit_history() -> None:
    recent, tencent, eastmoney = _fixtures()
    tencent["selected_assets"] = 1701
    tencent["ready_assets"] = 1701
    tencent["full_market_current_final_source_passed"] = True
    tencent["version_monitoring_ready"] = True
    tencent["version_depth_coverage"] = 1.0
    eastmoney["version_depth_coverage"] = 1.0
    checks = build_qualification_checks(recent, tencent, eastmoney).set_index("check")
    assert bool(checks.loc["full_market_price_independent_coverage", "passed"])
    assert bool(checks.loc["price_version_monitoring_depth", "passed"])
    assert bool(checks.loc["nav_version_monitoring_depth", "passed"])
    assert not bool(checks.loc["historical_available_date_evidence", "passed"])
