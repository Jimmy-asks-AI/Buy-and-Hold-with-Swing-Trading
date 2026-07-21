#!/usr/bin/env python
"""Build a self-contained HTML dashboard for HIRSSM iteration results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
OUTPUT_DIR = ROOT / "outputs" / "hirssm_iteration_dashboard"
HTML_PATH = OUTPUT_DIR / "HIRSSM_ITERATION_DASHBOARD.html"
COMPARABLE_START = pd.Timestamp("2007-02-01")


VERSIONS = [
    {
        "id": "V2.0",
        "label": "V2.0 修复基线",
        "summary": ROOT / "outputs" / "hirssm_v2_0" / "cost_sensitivity_summary.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_0" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_0" / "target_weights_monthly.csv",
        "change": "建立 HIRSSM 初始可运行模型，完成回测起点、收益、现金权重和专家剪枝修复。",
        "purpose": "形成正式旧基线。",
        "status": "旧基线",
    },
    {
        "id": "V2.0S",
        "label": "V2.0S 同期基线",
        "summary": ROOT / "outputs" / "hirssm_v2_1_walk_forward" / "same_period_baseline_summary.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_0" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_0" / "target_weights_monthly.csv",
        "nav_start": COMPARABLE_START,
        "rebase_nav": True,
        "change": "把 V2.0 旧基线裁剪到 2007-02-01 之后，并把净值重新归一到 1。用于和 V2.1 之后 walk-forward/OOS 版本同区间比较。",
        "purpose": "避免 V2.0 全样本和后续 OOS 版本直接比较造成视觉误判。",
        "status": "可比基线",
    },
    {
        "id": "V2.1",
        "label": "V2.1 状态门控",
        "summary": ROOT / "outputs" / "hirssm_v2_1_walk_forward" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_1_walk_forward" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_1_walk_forward" / "walk_forward_target_weights.csv",
        "change": "新增 walk-forward 专家状态门控，用过去 5 年 RankIC 决定下一年专家启停。",
        "purpose": "降低全样本剪枝依赖。",
        "status": "未晋级",
    },
    {
        "id": "V2.2",
        "label": "V2.2 连续收缩",
        "summary": ROOT / "outputs" / "hirssm_v2_2_walk_forward" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_2_walk_forward" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_2_walk_forward" / "walk_forward_target_weights.csv",
        "change": "把专家硬启停改为连续 shrinkage multiplier。",
        "purpose": "减少专家开关跳变。",
        "status": "研究版",
    },
    {
        "id": "V2.3",
        "label": "V2.3 嵌套选择",
        "summary": ROOT / "outputs" / "hirssm_v2_3_nested_walk_forward" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_3_nested_walk_forward" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_3_nested_walk_forward" / "walk_forward_target_weights.csv",
        "change": "每年只用过去窗口选择 shrinkage 参数族。",
        "purpose": "降低全样本选参风险。",
        "status": "研究版",
    },
    {
        "id": "V2.4",
        "label": "V2.4 稳定嵌套选择",
        "summary": ROOT / "outputs" / "hirssm_v2_4_stable_nested_selection" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_4_stable_nested_selection" / "nav_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_4_stable_nested_selection" / "walk_forward_target_weights.csv",
        "change": "压缩为 3 个稳定参数族，加入多成本目标和切换惩罚。",
        "purpose": "减少过拟合自由度，作为收益参考基线。",
        "status": "收益基线",
    },
    {
        "id": "V2.5",
        "label": "V2.5 全局组合风控",
        "summary": ROOT / "outputs" / "hirssm_v2_5_portfolio_risk_overlay" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_5_portfolio_risk_overlay" / "nav_v25_overlay_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_5_portfolio_risk_overlay" / "walk_forward_target_weights.csv",
        "change": "加入状态目标波动、总暴露上限、市场/组合回撤刹车、现金替代和拥挤降权。",
        "purpose": "验证固定组合风控能否明显降回撤。",
        "status": "风险控制版",
    },
    {
        "id": "V2.6",
        "label": "V2.6 局部 sleeve 风控",
        "summary": ROOT / "outputs" / "hirssm_v2_6" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_6" / "nav_hirssm_v2_6_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_6" / "walk_forward_target_weights.csv",
        "change": "把全组合缩放改为 style/industry 局部 sleeve 风控，并加入归因输出。",
        "purpose": "降低回撤，同时减少全局现金拖累。",
        "status": "防守候选",
    },
    {
        "id": "V2.7",
        "label": "V2.7 再入场修复",
        "summary": ROOT / "outputs" / "hirssm_v2_7" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_7" / "nav_hirssm_v2_7_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_7" / "walk_forward_target_weights.csv",
        "change": "在局部风控上增加回撤后的规则化再入场机制。",
        "purpose": "提高收益恢复能力，避免低位长期降仓。",
        "status": "当前推荐",
    },
    {
        "id": "V2.8",
        "label": "V2.8 sleeve 专项修复",
        "summary": ROOT / "outputs" / "hirssm_v2_8" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_8" / "nav_hirssm_v2_8_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_8" / "walk_forward_target_weights.csv",
        "change": "保护 style 核心资产，对 industry 做更局部降权。",
        "purpose": "尝试保留核心风格收益。",
        "status": "不推荐晋级",
    },
    {
        "id": "V2.9",
        "label": "V2.9 固定混合",
        "summary": ROOT / "outputs" / "hirssm_v2_9" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_9" / "nav_hirssm_v2_9_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_9" / "walk_forward_target_weights.csv",
        "change": "固定混合 V2.4 收益基线和 V2.8 风控权重；V2.9.1 修复混合权重 asset 字段丢失。",
        "purpose": "提高收益保留，保留部分回撤改善。",
        "status": "收益保留候选",
    },
    {
        "id": "V2.10",
        "label": "V2.10 软门控试验",
        "summary": ROOT / "outputs" / "hirssm_v2_10_soft_killswitch" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_10_soft_killswitch" / "nav_hirssm_v2_10_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_10_soft_killswitch" / "walk_forward_target_weights.csv",
        "change": "取消 V2.1 硬门控直接入组合，改为连续软乘数，并加入强负证据 kill-switch。",
        "purpose": "验证专家治理是否能保留收益并减少硬开关跳变。",
        "status": "未通过自检",
    },
    {
        "id": "V2.10.1",
        "label": "V2.10.1 限定 kill-switch",
        "summary": ROOT / "outputs" / "hirssm_v2_10_1_soft_killswitch" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v2_10_1_soft_killswitch" / "nav_hirssm_v2_10_1_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v2_10_1_soft_killswitch" / "walk_forward_target_weights.csv",
        "change": "把 V2.10 中过度激进的核心专家硬 kill 改为软降权；硬 kill 只允许作用于行业趋势和行业流动性。",
        "purpose": "保留治理审计能力，同时避免核心专家被 0/1 开关破坏。",
        "status": "治理候选",
    },
    {
        "id": "V3.0",
        "label": "V3.0 相对基准目标",
        "summary": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_0" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_0" / "candidates" / "v3_0_v2_7_risk_overlay" / "nav_v3_0_v2_7_risk_overlay_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_0" / "walk_forward_target_weights.csv",
        "change": "把选型目标从绝对收益/稳健性改为相对中证全指的主动收益、回撤改善和信息比率。",
        "purpose": "确认当前候选袖套在 benchmark-relative 目标下的排序。",
        "status": "投资门槛未过",
    },
    {
        "id": "V3.1",
        "label": "V3.1 核心-卫星",
        "summary": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_1" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_1" / "candidates" / "v3_1_defensive_core" / "nav_v3_1_defensive_core_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_0_v3_1_benchmark_core" / "v3_1" / "walk_forward_target_weights.csv",
        "change": "在 V3.0 选出的增强袖套上叠加状态条件化的 000985 核心仓和卫星增强仓。",
        "purpose": "提高市场参与度，验证核心-卫星结构是否改善相对收益。",
        "status": "不推荐晋级",
    },
    {
        "id": "V3.2",
        "label": "V3.2 市场 beta 择时",
        "summary": ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "candidates" / "v3_2_recovery_attack" / "nav_v3_2_recovery_attack_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "walk_forward_target_weights.csv",
        "change": "在 V3.0 选出的 alpha 袖套上叠加独立市场 beta 择时层，用趋势、宽度、波动、回撤和修复状态决定总权益暴露；不允许杠杆和负现金。",
        "purpose": "在不新增横截面 alpha 的前提下，提高风险开启阶段的市场参与度，并在恐慌状态保留现金保护。",
        "status": "强候选但门槛未过",
    },
    {
        "id": "V3.3",
        "label": "V3.3 alpha 工厂",
        "summary": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_3" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_3" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_3" / "walk_forward_target_weights.csv",
        "change": "新增横截面 alpha 工厂：行业景气代理、相对动量、风格估值修复、低风险质量、拥挤度缓解、反弹修复和流动性确认；使用过去 5 年 RankIC 做因子门控。",
        "purpose": "检验仅靠横截面 alpha 选择能否提高相对中证全指的成本后超额。",
        "status": "未晋级",
    },
    {
        "id": "V3.4",
        "label": "V3.4 alpha+beta",
        "summary": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_4" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_4" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_4" / "walk_forward_target_weights.csv",
        "change": "把 V3.3 选中的 alpha 工厂袖套接入 V3.2 的市场 beta 择时层，按趋势、宽度、回撤和修复状态调节总权益暴露。",
        "purpose": "检验 alpha 排名和 beta 择时是否能共同改善成本后超额。",
        "status": "未晋级",
    },
    {
        "id": "V3.5",
        "label": "V3.5 稳健 ensemble",
        "summary": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "walk_forward_target_weights.csv",
        "change": "将 V3.2 beta 择时袖套与 V3.4 alpha+beta 袖套做稳健混合，加入单资产上限和调仓带，选中 beta anchor 版本。",
        "purpose": "在不过度依赖新 alpha 的情况下，提高收益、降低回撤和换手，并通过 3% 年化超额准入门槛。",
        "status": "当前最强候选",
    },
    {
        "id": "V3.6",
        "label": "V3.6 状态门控 alpha",
        "summary": ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "walk_forward_target_weights.csv",
        "change": "在 V3.5 归因基础上做组件消融，保留 V3.2 beta 择时锚，只在趋势和震荡状态加入 V3.4 alpha sleeve；在 crash/risk-off 状态关闭 alpha sleeve。",
        "purpose": "验证 V3.5 的边际收益是否来自可解释的状态条件化 alpha，而不是全局混合带来的偶然提升。",
        "status": "强候选",
    },
    {
        "id": "V3.7",
        "label": "V3.7 滚动状态门控验证",
        "summary": ROOT / "outputs" / "hirssm_v3_7_state_alpha_walkforward" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_7_state_alpha_walkforward" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_7_state_alpha_walkforward" / "walk_forward_target_weights.csv",
        "change": "用 state-only 原型组合的过往 5 年表现决定下一年 alpha sleeve 是否收缩，并保留 V3.6 精确控制组。",
        "purpose": "验证 V3.6 的状态 alpha 是否能经受滚动样本外门控，而不是仅靠全样本归因看起来有效。",
        "status": "验证版本未晋级",
    },
    {
        "id": "V3.8",
        "label": "V3.8 风险预算 overlay",
        "summary": ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay" / "oos_performance.csv",
        "nav": ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay" / "nav_selected_10bps.csv",
        "targets": ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay" / "walk_forward_target_weights.csv",
        "change": "在 V3.6 权重上测试波动率预算、相关簇约束、回撤贡献约束和换手感知风险预算；最终选择波动率预算 overlay。",
        "purpose": "不新增 alpha，只优化组合风险结构，尽量保留 V3.6 收益并小幅降低回撤和换手。",
        "status": "当前最强候选",
    },
]


COLORS = {
    "V2.0": "#5b6472",
    "V2.0S": "#344054",
    "V2.1": "#a75d5d",
    "V2.2": "#b9822b",
    "V2.3": "#8a7c2f",
    "V2.4": "#2f6b9a",
    "V2.5": "#6b5ca5",
    "V2.6": "#21867a",
    "V2.7": "#15803d",
    "V2.8": "#7c3aed",
    "V2.9": "#c2410c",
    "V2.10": "#b42318",
    "V2.10.1": "#0f766e",
    "V3.0": "#1d4ed8",
    "V3.1": "#9333ea",
    "V3.2": "#be123c",
    "V3.3": "#9a3412",
    "V3.4": "#7e22ce",
    "V3.5": "#047857",
    "V3.6": "#0f766e",
    "V3.7": "#64748b",
    "V3.8": "#0369a1",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def pct(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def pick_10bps(summary: pd.DataFrame) -> dict:
    if summary.empty:
        return {}
    if "cost_bps" in summary.columns:
        rows = summary[pd.to_numeric(summary["cost_bps"], errors="coerce").round(6).eq(10.0)]
        if rows.empty:
            rows = summary.head(1)
    else:
        rows = summary.head(1)
    row = rows.iloc[0]
    return {
        "annual_return": pct(row.get("annual_return")),
        "annual_vol": pct(row.get("annual_vol")),
        "sharpe": pct(row.get("sharpe_no_rf")),
        "max_drawdown": pct(row.get("max_drawdown")),
        "avg_cash": pct(row.get("avg_cash_weight")),
        "avg_turnover": pct(row.get("avg_trade_turnover", row.get("avg_turnover"))),
        "total_return": pct(row.get("total_return")),
    }


def load_nav(path: Path, start: pd.Timestamp | None = None, rebase: bool = False) -> list[dict]:
    df = read_csv(path)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "nav"]).sort_values("date")
    if start is not None:
        df = df[df["date"] >= start]
    if df.empty:
        return []
    if rebase:
        nav_base = pd.to_numeric(df["nav"], errors="coerce").dropna()
        if not nav_base.empty and nav_base.iloc[0] != 0:
            df["nav"] = pd.to_numeric(df["nav"], errors="coerce") / float(nav_base.iloc[0])
        if "benchmark_nav" in df.columns:
            benchmark_base = pd.to_numeric(df["benchmark_nav"], errors="coerce").dropna()
            if not benchmark_base.empty and benchmark_base.iloc[0] != 0:
                df["benchmark_nav"] = pd.to_numeric(df["benchmark_nav"], errors="coerce") / float(benchmark_base.iloc[0])
    df["drawdown"] = df["nav"] / df["nav"].cummax() - 1.0
    cols = ["date", "nav", "drawdown", "cash_weight", "turnover", "benchmark_nav"]
    rows = []
    for _, row in df.iterrows():
        item = {"date": row["date"].strftime("%Y-%m-%d")}
        for col in cols[1:]:
            item[col] = pct(row.get(col))
        rows.append(item)
    return thin_rows(rows, 1500)


def thin_rows(rows: list[dict], max_rows: int) -> list[dict]:
    if len(rows) <= max_rows:
        return rows
    step = int(np.ceil(len(rows) / max_rows))
    thinned = rows[::step]
    if thinned[-1]["date"] != rows[-1]["date"]:
        thinned.append(rows[-1])
    return thinned


def target_exposure(path: Path) -> list[dict]:
    df = read_csv(path)
    if df.empty:
        return []
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df["asset_type"] = df.get("asset_type", "").fillna("")
    grouped = df.pivot_table(index=["signal_date", "state"], columns="asset_type", values="weight", aggfunc="sum", fill_value=0.0).reset_index()
    for col in ["cash", "style", "industry"]:
        if col not in grouped.columns:
            grouped[col] = 0.0
    grouped["gross"] = 1.0 - grouped["cash"]
    out = []
    for _, row in grouped.sort_values("signal_date").iterrows():
        out.append(
            {
                "date": row["signal_date"].strftime("%Y-%m-%d"),
                "state": str(row.get("state", "")),
                "cash": pct(row.get("cash")),
                "gross": pct(row.get("gross")),
                "style": pct(row.get("style")),
                "industry": pct(row.get("industry")),
            }
        )
    return out


def weekly_kline(path: Path) -> list[dict]:
    df = read_csv(path)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_full_ohlc_bar" in df.columns:
        mask = df["is_full_ohlc_bar"].astype(str).str.lower().isin(["true", "1"])
        df = df[mask]
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    df = df[df["date"] >= pd.Timestamp("2007-01-01")]
    weekly = df.set_index("date").resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    out = []
    for _, row in weekly.iterrows():
        out.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return out


def build_data() -> dict:
    version_data = []
    for info in VERSIONS:
        metrics = pick_10bps(read_csv(info["summary"]))
        nav = load_nav(info["nav"], start=info.get("nav_start"), rebase=bool(info.get("rebase_nav", False)))
        exposures = target_exposure(info["targets"])
        version_data.append(
            {
                "id": info["id"],
                "label": info["label"],
                "color": COLORS[info["id"]],
                "metrics": metrics,
                "change": info["change"],
                "purpose": info["purpose"],
                "status": info["status"],
                "nav": nav,
                "exposure": exposures,
            }
        )
    return {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": "000985 中证全指，周 K",
        "kline": weekly_kline(ROOT / "data_raw" / "index" / "akshare_csindex" / "daily_csindex" / "000985.csv"),
        "versions": version_data,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HIRSSM 迭代可视化仪表盘</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #155e75;
      --good: #15803d;
      --bad: #b42318;
      --warn: #b7791f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 24px 28px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }
    p { margin: 0; color: var(--muted); line-height: 1.55; }
    main { padding: 20px 28px 36px; max-width: 1500px; margin: 0 auto; }
    .grid { display: grid; gap: 16px; }
    .cards { grid-template-columns: repeat(4, minmax(180px, 1fr)); }
    .card, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .card .label { color: var(--muted); font-size: 13px; }
    .card .value { font-size: 26px; font-weight: 700; margin-top: 8px; }
    .card .sub { color: var(--muted); margin-top: 6px; font-size: 12px; }
    .section-title {
      margin: 8px 0 12px;
      font-size: 18px;
      font-weight: 700;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      margin-bottom: 12px;
    }
    .check {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      font-size: 13px;
      color: var(--ink);
    }
    select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      padding: 0 8px;
      color: var(--ink);
    }
    canvas {
      width: 100%;
      height: 340px;
      display: block;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .wide { grid-column: 1 / -1; }
    .two { grid-template-columns: 1.25fr 1fr; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2) { text-align: left; }
    th { background: #f1f4f8; color: #344054; font-weight: 700; }
    tr:last-child td { border-bottom: 0; }
    .status {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: #344054;
      background: #f8fafc;
      font-size: 12px;
    }
    .status.recommended { color: #166534; background: #ecfdf3; border-color: #bbf7d0; }
    .status.base { color: #075985; background: #eef6ff; border-color: #bfdbfe; }
    .status.no { color: #991b1b; background: #fff1f2; border-color: #fecdd3; }
    .note {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.6;
      margin-top: 8px;
    }
    @media (max-width: 900px) {
      main { padding: 14px; }
      .cards, .two { grid-template-columns: 1fr; }
      th, td { font-size: 12px; padding: 7px; }
      canvas { height: 280px; }
    }
  </style>
</head>
<body>
<header>
  <h1>HIRSSM 迭代可视化仪表盘</h1>
  <p>覆盖 V2.0 到 V3.5：关键指标、净值、回撤、现金暴露，以及中证全指周 K 线上的调仓/风险暴露变化。数据口径默认使用 10bps 成本情景。</p>
</header>
<main>
  <section class="grid cards" id="summaryCards"></section>

  <section class="panel wide" style="margin-top:16px;">
    <div class="section-title">版本选择</div>
    <div class="toolbar" id="versionToggles"></div>
    <div class="note">默认展示 V2.0S 同期基线、V2.7 稳健基线、V3.2 市场 beta 择时、V3.5 当前最强候选。V3.3/V3.4 保留为 alpha 路线失败对照；V2.0 是全样本旧基线，V2.0S 是裁剪到 2007-02-01 后的可比口径。</div>
  </section>

  <section class="grid two" style="margin-top:16px;">
    <div class="panel">
      <div class="section-title">净值曲线（V2.0S 为同期归一化基线）</div>
      <canvas id="navChart" width="1200" height="420"></canvas>
    </div>
    <div class="panel">
      <div class="section-title">最大回撤曲线</div>
      <canvas id="drawdownChart" width="900" height="420"></canvas>
    </div>
  </section>

  <section class="grid two" style="margin-top:16px;">
    <div class="panel">
      <div class="section-title">现金暴露变化</div>
      <canvas id="cashChart" width="1200" height="420"></canvas>
    </div>
    <div class="panel">
      <div class="section-title">中证全指周 K + 版本调仓点</div>
      <div class="toolbar">
        <span style="font-size:13px;color:var(--muted);">K 线叠加版本：</span>
        <select id="klineVersion"></select>
      </div>
      <canvas id="klineChart" width="900" height="420"></canvas>
      <div class="note">圆点越高表示对应调仓日在指数价格越高；圆点颜色随现金仓位变化，现金越高越偏橙红。</div>
    </div>
  </section>

  <section class="panel wide" style="margin-top:16px;">
    <div class="section-title">关键指标表</div>
    <div id="metricsTable"></div>
  </section>

  <section class="panel wide" style="margin-top:16px;">
    <div class="section-title">每个版本改了什么</div>
    <div id="changeTable"></div>
  </section>
</main>
<script>
const DATA = __DATA_JSON__;

const fmtPct = (x, d=2) => x === null || x === undefined || Number.isNaN(x) ? "" : (x * 100).toFixed(d) + "%";
const fmtNum = (x, d=3) => x === null || x === undefined || Number.isNaN(x) ? "" : Number(x).toFixed(d);
const parseDate = d => new Date(d + "T00:00:00").getTime();
const defaultActive = new Set(["V2.0S", "V2.7", "V3.2", "V3.6", "V3.8"]);

function versionById(id) { return DATA.versions.find(v => v.id === id); }
function activeVersions() {
  return DATA.versions.filter(v => {
    const el = document.querySelector(`input[data-version="${v.id}"]`);
    return el && el.checked;
  });
}

function setupControls() {
  const toggles = document.getElementById("versionToggles");
  toggles.innerHTML = "";
  DATA.versions.forEach(v => {
    const label = document.createElement("label");
    label.className = "check";
    label.innerHTML = `<input type="checkbox" data-version="${v.id}" ${defaultActive.has(v.id) ? "checked" : ""}> <span style="color:${v.color};font-weight:700;">${v.id}</span><span>${v.label.replace(v.id + " ", "")}</span>`;
    toggles.appendChild(label);
  });
  toggles.addEventListener("change", drawAll);

  const select = document.getElementById("klineVersion");
  DATA.versions.forEach(v => {
    const option = document.createElement("option");
    option.value = v.id;
    option.textContent = `${v.id} ${v.label}`;
    if (v.id === "V2.7") option.selected = true;
    select.appendChild(option);
  });
  select.addEventListener("change", drawKline);
}

function statusClass(status) {
  if (status.includes("基线")) return "base";
  if (status.includes("不推荐") || status.includes("未晋级") || status.includes("未过") || status.includes("未通过")) return "no";
  if (status.includes("推荐") || status.includes("候选")) return "recommended";
  return "";
}

function renderCards() {
  const cards = document.getElementById("summaryCards");
  const v20s = versionById("V2.0S");
  const v24 = versionById("V2.4");
  const v27 = versionById("V2.7");
  const v29 = versionById("V2.9");
  const v2101 = versionById("V2.10.1");
  const v38 = versionById("V3.8");
  const items = [
    ["当前最强候选", "V3.8", `${fmtPct(v38.metrics.annual_return)} / Sharpe ${fmtNum(v38.metrics.sharpe)}`, `最大回撤 ${fmtPct(v38.metrics.max_drawdown)}`],
    ["稳健参考", "V2.7", `${fmtPct(v27.metrics.annual_return)} / Sharpe ${fmtNum(v27.metrics.sharpe)}`, `最大回撤 ${fmtPct(v27.metrics.max_drawdown)}`],
    ["同期基线", "V2.0S", `${fmtPct(v20s.metrics.annual_return)} / Sharpe ${fmtNum(v20s.metrics.sharpe)}`, `最大回撤 ${fmtPct(v20s.metrics.max_drawdown)}`],
    ["收益基线", "V2.4", `${fmtPct(v24.metrics.annual_return)} / Sharpe ${fmtNum(v24.metrics.sharpe)}`, `最大回撤 ${fmtPct(v24.metrics.max_drawdown)}`],
    ["收益保留", "V2.9", `${fmtPct(v29.metrics.annual_return)} / 回撤 ${fmtPct(v29.metrics.max_drawdown)}`, `平均现金 ${fmtPct(v29.metrics.avg_cash)}`],
  ];
  cards.innerHTML = items.map(([label, title, value, sub]) => `
    <div class="card">
      <div class="label">${label}</div>
      <div class="value">${title}</div>
      <div class="sub">${value}</div>
      <div class="sub">${sub}</div>
    </div>`).join("");
}

function renderTables() {
  const rows = DATA.versions.map(v => `
    <tr>
      <td><span style="color:${v.color};font-weight:700;">${v.id}</span></td>
      <td>${v.label}</td>
      <td><span class="status ${statusClass(v.status)}">${v.status}</span></td>
      <td>${fmtPct(v.metrics.annual_return)}</td>
      <td>${fmtNum(v.metrics.sharpe)}</td>
      <td>${fmtPct(v.metrics.max_drawdown)}</td>
      <td>${fmtPct(v.metrics.avg_cash)}</td>
      <td>${fmtPct(v.metrics.annual_vol)}</td>
      <td>${fmtPct(v.metrics.avg_turnover)}</td>
    </tr>`).join("");
  document.getElementById("metricsTable").innerHTML = `
    <table>
      <thead><tr><th>版本</th><th>名称</th><th>定位</th><th>年化收益</th><th>Sharpe</th><th>最大回撤</th><th>平均现金</th><th>年化波动</th><th>平均交易换手</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  const changes = DATA.versions.map(v => `
    <tr>
      <td><span style="color:${v.color};font-weight:700;">${v.id}</span></td>
      <td style="text-align:left;white-space:normal;">${v.change}</td>
      <td style="text-align:left;white-space:normal;">${v.purpose}</td>
      <td><span class="status ${statusClass(v.status)}">${v.status}</span></td>
    </tr>`).join("");
  document.getElementById("changeTable").innerHTML = `
    <table>
      <thead><tr><th>版本</th><th>改了什么</th><th>为什么改</th><th>结论</th></tr></thead>
      <tbody>${changes}</tbody>
    </table>`;
}

function canvasSetup(id) {
  const canvas = document.getElementById(id);
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { canvas, ctx, w: rect.width, h: rect.height };
}

function drawAxes(ctx, x, y, w, h, minY, maxY) {
  ctx.strokeStyle = "#d9dee7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yy = y + h * i / 4;
    ctx.moveTo(x, yy); ctx.lineTo(x + w, yy);
  }
  ctx.stroke();
  ctx.fillStyle = "#667085";
  ctx.font = "12px Segoe UI";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const value = maxY - (maxY - minY) * i / 4;
    ctx.fillText(value.toFixed(2), x - 8, y + h * i / 4 + 4);
  }
}

function dateRange(seriesList) {
  let min = Infinity, max = -Infinity;
  seriesList.forEach(s => s.forEach(p => {
    const t = parseDate(p.date);
    if (t < min) min = t;
    if (t > max) max = t;
  }));
  return [min, max];
}

function drawLineChart(id, field, yFormatter, fixedMin=null, fixedMax=null) {
  const {ctx, w, h} = canvasSetup(id);
  ctx.clearRect(0, 0, w, h);
  const pad = {l: 54, r: 18, t: 18, b: 36};
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const versions = activeVersions().filter(v => v.nav && v.nav.length);
  if (!versions.length) return;
  const series = versions.map(v => v.nav);
  const [minT, maxT] = dateRange(series);
  let values = [];
  versions.forEach(v => v.nav.forEach(p => { if (p[field] !== null && p[field] !== undefined) values.push(p[field]); }));
  let minY = fixedMin !== null ? fixedMin : Math.min(...values);
  let maxY = fixedMax !== null ? fixedMax : Math.max(...values);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const xScale = d => pad.l + (parseDate(d) - minT) / (maxT - minT) * plotW;
  const yScale = v => pad.t + (maxY - v) / (maxY - minY) * plotH;
  drawAxes(ctx, pad.l, pad.t, plotW, plotH, minY, maxY);
  versions.forEach(v => {
    ctx.strokeStyle = v.color;
    ctx.lineWidth = v.id === "V2.7" ? 2.4 : 1.8;
    ctx.beginPath();
    let started = false;
    v.nav.forEach(p => {
      const val = p[field];
      if (val === null || val === undefined || Number.isNaN(val)) return;
      const x = xScale(p.date), y = yScale(val);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#172033";
  ctx.textAlign = "left";
  ctx.font = "12px Segoe UI";
  let lx = pad.l, ly = h - 14;
  versions.forEach(v => {
    ctx.fillStyle = v.color; ctx.fillRect(lx, ly - 8, 18, 3);
    ctx.fillStyle = "#344054"; ctx.fillText(v.id, lx + 24, ly - 4);
    lx += 68;
  });
}

function nearestKlineIndex(date) {
  const t = parseDate(date);
  let lo = 0, hi = DATA.kline.length - 1, best = 0;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const mt = parseDate(DATA.kline[mid].date);
    if (mt <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

function cashColor(cash) {
  if (cash >= 0.35) return "#c2410c";
  if (cash >= 0.22) return "#b7791f";
  return "#15803d";
}

function drawKline() {
  const {ctx, w, h} = canvasSetup("klineChart");
  ctx.clearRect(0, 0, w, h);
  const candles = DATA.kline;
  if (!candles.length) return;
  const pad = {l: 54, r: 18, t: 18, b: 36};
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const lows = candles.map(d => d.low), highs = candles.map(d => d.high);
  const minY = Math.min(...lows) * 0.98, maxY = Math.max(...highs) * 1.02;
  drawAxes(ctx, pad.l, pad.t, plotW, plotH, minY, maxY);
  const yScale = v => pad.t + (maxY - v) / (maxY - minY) * plotH;
  const barW = Math.max(2, plotW / candles.length * 0.62);
  candles.forEach((d, i) => {
    const x = pad.l + i / (candles.length - 1) * plotW;
    const up = d.close >= d.open;
    ctx.strokeStyle = up ? "#b42318" : "#15803d";
    ctx.fillStyle = up ? "rgba(180,35,24,0.65)" : "rgba(21,128,61,0.65)";
    ctx.beginPath();
    ctx.moveTo(x, yScale(d.high)); ctx.lineTo(x, yScale(d.low)); ctx.stroke();
    const y1 = yScale(d.open), y2 = yScale(d.close);
    ctx.fillRect(x - barW / 2, Math.min(y1, y2), barW, Math.max(1, Math.abs(y2 - y1)));
  });
  const selected = versionById(document.getElementById("klineVersion").value);
  if (selected && selected.exposure) {
    selected.exposure.forEach(e => {
      const idx = nearestKlineIndex(e.date);
      const d = candles[idx];
      const x = pad.l + idx / (candles.length - 1) * plotW;
      const y = yScale(d.close);
      ctx.beginPath();
      ctx.fillStyle = cashColor(e.cash || 0);
      ctx.globalAlpha = 0.72;
      ctx.arc(x, y, 3.3 + Math.min(4, (e.cash || 0) * 8), 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  }
  ctx.fillStyle = "#344054";
  ctx.font = "12px Segoe UI";
  ctx.textAlign = "left";
  ctx.fillText(`${selected ? selected.id : ""} 调仓点：绿=低现金，橙/红=高现金`, pad.l, h - 14);
}

function drawAll() {
  drawLineChart("navChart", "nav", v => v.toFixed(2), 0.8, null);
  drawLineChart("drawdownChart", "drawdown", v => fmtPct(v), -0.6, 0.02);
  drawLineChart("cashChart", "cash_weight", v => fmtPct(v), 0, 0.5);
  drawKline();
}

window.addEventListener("resize", drawAll);
setupControls();
renderCards();
renderTables();
drawAll();
</script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    HTML_PATH.write_text(html, encoding="utf-8")
    manifest = {
        "html": str(HTML_PATH),
        "versions": [v["id"] for v in data["versions"]],
        "kline_rows": len(data["kline"]),
        "generated_at": data["generated_at"],
    }
    (OUTPUT_DIR / "dashboard_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
