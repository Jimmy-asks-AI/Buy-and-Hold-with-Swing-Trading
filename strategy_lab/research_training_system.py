#!/usr/bin/env python
"""Training and stage-gate helpers for a quant research curriculum."""

from __future__ import annotations

from typing import Any

import pandas as pd


def material_reading_route() -> pd.DataFrame:
    """Return the three-layer reading route for the local corpus."""
    rows = [
        {
            "layer": "must_read",
            "material": "00-量化研究框架.md",
            "goal": "understand the full research loop",
            "boundary": "do not jump into strategy optimization yet",
        },
        {
            "layer": "must_read",
            "material": "课程详细版/深度概念手册.md",
            "goal": "clarify indicator, factor, signal, strategy, backtest and risk",
            "boundary": "examples first, formulas second",
        },
        {
            "layer": "must_read",
            "material": "资料/技术指标回测代码/technical.py",
            "goal": "map KDJ indicator calculations to pandas operations",
            "boundary": "learn the research loop, not KDJ worship",
        },
        {
            "layer": "must_read",
            "material": "资料/技术指标回测代码/main.py",
            "goal": "separate signal validation from full backtest",
            "boundary": "future returns are labels only",
        },
        {
            "layer": "selective",
            "material": "资料/因子挖掘.md",
            "goal": "learn common factor families and evaluation language",
            "boundary": "simple factors before machine-generated factors",
        },
        {
            "layer": "selective",
            "material": "资料/卖方金工研报/海通报告",
            "goal": "learn professional factor research structure",
            "boundary": "read title, abstract and tables before formulas",
        },
        {
            "layer": "selective",
            "material": "资料/卖方金工研报/基金研究",
            "goal": "learn ETF/fund labels, benchmarks and allocation framing",
            "boundary": "match benchmark to asset pool",
        },
        {
            "layer": "defer",
            "material": "资料/卖方金工研报/华泰人工智能",
            "goal": "machine learning and overfitting extension",
            "boundary": "only after factor/backtest basics are stable",
        },
        {
            "layer": "defer",
            "material": "论文/AI金融论文整理",
            "goal": "frontier tracking and topic expansion",
            "boundary": "not an entry shortcut",
        },
    ]
    return pd.DataFrame(rows)


def concept_exam_bank() -> pd.DataFrame:
    """Return core oral-exam questions for the first quant research stage."""
    questions = [
        ("concept", "量化研究和量化交易有什么区别？"),
        ("concept", "指标和因子有什么区别？"),
        ("concept", "信号和策略有什么区别？"),
        ("concept", "信号验证和完整回测有什么区别？"),
        ("concept", "绝对收益和超额收益有什么区别？"),
        ("concept", "Alpha 和 Beta 有什么区别？"),
        ("risk", "波动和回撤有什么区别？"),
        ("risk", "过拟合和策略失效有什么区别？"),
        ("ai", "AI 辅助量化研究和 AI 自动交易有什么区别？"),
        ("ai", "为什么 AI 生成的因子更需要样本外检验？"),
        ("data", "为什么公告日比报告期更重要？"),
        ("data", "为什么未来收益可以用于评价但不能用于生成信号？"),
    ]
    return pd.DataFrame(questions, columns=["section", "question"])


def minimum_research_system_modules() -> pd.DataFrame:
    """Return the minimum directory/module map for a reusable research system."""
    rows = [
        ("research_ideas", "idea log, hypothesis, status, conclusion"),
        ("data_catalog", "data source, field, timestamp, adjustment and survivorship policy"),
        ("factor_library", "factor formula, direction, data fields, risks and validation status"),
        ("strategy_lab", "reusable code for factors, timing, portfolios, backtests and diagnostics"),
        ("backtest_configs", "asset pool, rebalance, cost, benchmark and constraints separated from code"),
        ("validation", "data checks, leakage checks, PBO, sample-out and robustness reports"),
        ("replication_reports", "paper/report reproduction and failure analysis"),
        ("ai_assistant_logs", "AI suggestions, accepted changes, rejected claims and verification evidence"),
        ("reports", "final research notes and decision records"),
    ]
    return pd.DataFrame(rows, columns=["module", "purpose"])


def assignment_template(topic: str = "etf_momentum") -> dict[str, Any]:
    """Return a structured first-research assignment template."""
    topic = topic.lower()
    common = {
        "research_question": "",
        "why_it_may_work": "",
        "research_object": "",
        "data_dictionary": "",
        "factor_or_signal_definition": "",
        "validation_method": "",
        "backtest_config": "",
        "evaluation_metrics": "",
        "risk_checks": "",
        "material_mapping": "",
        "ai_collaboration_record": "",
        "current_conclusion": "",
        "next_step": "",
    }
    if topic in {"etf_momentum", "etf_rotation"}:
        common.update(
            {
                "research_object": "ETF pool",
                "factor_or_signal_definition": "past N-day return or risk-adjusted momentum",
                "backtest_config": "monthly rebalance, top-k equal weight, benchmark=ETF equal-weight pool",
                "risk_checks": "ETF liquidity, listing age, industry concentration, parameter overfit, turnover cost",
            }
        )
    elif topic in {"low_pb", "value_factor"}:
        common.update(
            {
                "research_object": "A-share stock pool",
                "factor_or_signal_definition": "PB sorted low-to-high, low PB expected positive if value premium exists",
                "validation_method": "RankIC, quantile return, long-short, industry and size attribution",
                "risk_checks": "value trap, industry concentration, financial data announcement date, liquidity",
            }
        )
    return common


def assignment_rubric() -> pd.DataFrame:
    """Return the minimum pass rubric for a first research design."""
    rows = [
        ("research_question", "fail", "must specify object, rule, horizon and evaluation target"),
        ("data_dictionary", "fail", "must specify fields, frequency, timestamp and missing-value policy"),
        ("factor_or_signal_definition", "fail", "must specify formula, direction and no-lookahead rule"),
        ("validation_method", "fail", "must distinguish signal validation, factor validation and full backtest"),
        ("backtest_config", "warn", "should specify rebalance, execution, cost, benchmark and non-tradable handling"),
        ("evaluation_metrics", "warn", "should include return, drawdown, excess return, turnover and stability"),
        ("risk_checks", "warn", "should include cost, liquidity, concentration, overfitting and exposure checks"),
        ("ai_collaboration_record", "warn", "should record what AI produced and how it was verified"),
        ("current_conclusion", "warn", "should be restrained and state limits"),
    ]
    return pd.DataFrame(rows, columns=["field", "severity", "criterion"])


def audit_assignment(submission: dict[str, Any]) -> pd.DataFrame:
    """Audit a student/researcher assignment submission against the rubric."""
    rows = []
    for row in assignment_rubric().itertuples(index=False):
        value = submission.get(row.field)
        present = value is not None and not (isinstance(value, str) and not value.strip())
        rows.append(
            {
                "field": row.field,
                "status": "pass" if present else row.severity,
                "present": bool(present),
                "criterion": row.criterion,
            }
        )
    return pd.DataFrame(rows)


def ai_collaboration_audit_template() -> pd.DataFrame:
    """Return the audit fields required after AI-assisted research."""
    fields = [
        ("ai_did", "What did AI generate or modify?"),
        ("accepted", "Which suggestions were accepted?"),
        ("rejected", "Which suggestions were rejected and why?"),
        ("code_validation", "How were syntax, tests and smoke checks run?"),
        ("leakage_check", "How was look-ahead or label leakage checked?"),
        ("data_version", "Which data version and timestamp policy were used?"),
        ("trade_assumption", "How were execution price, cost and tradability assumptions checked?"),
        ("overfit_check", "How were sample-out, PBO or robustness checks performed?"),
        ("human_signoff", "Who approved the final research conclusion?"),
    ]
    return pd.DataFrame(fields, columns=["field", "question"])


def readiness_summary(audit: pd.DataFrame) -> pd.Series:
    """Summarize audit output into a simple readiness state."""
    fail_count = int((audit["status"] == "fail").sum())
    warn_count = int((audit["status"] == "warn").sum())
    pass_count = int((audit["status"] == "pass").sum())
    if fail_count:
        readiness = "not_ready"
    elif warn_count:
        readiness = "ready_for_guided_research"
    else:
        readiness = "ready_for_independent_research"
    return pd.Series(
        {
            "readiness": readiness,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
        }
    )
