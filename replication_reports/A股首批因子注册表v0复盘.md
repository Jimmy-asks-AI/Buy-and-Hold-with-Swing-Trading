# A 股首批因子注册表 v0 复盘

日期：2026-05-23

## 本轮目标

把因子工厂从“模板”推进到“可治理候选因子库”，先建立 A 股首批候选因子注册表。

## 新增产物

- `data_catalog/a_share_factor_registry_v0.csv`
- `strategy_lab/factor_registry_audit.py`
- `outputs/a_share_factor_registry_v0_audit/summary.csv`
- `outputs/a_share_factor_registry_v0_audit/family_distribution.csv`
- `outputs/a_share_factor_registry_v0_audit/issues.csv`

## 审计结果

```text
factor_count: pass, 68
family_count: pass, 21
duplicate_factor_id: pass, 0
fail_issues: pass, 0
warn_issues: pass, 0
```

## 家族分布

注册表覆盖：

- value
- dividend
- quality
- profitability
- growth
- investment
- leverage
- momentum
- reversal
- liquidity
- volatility
- technical
- intraday
- level2
- analyst
- behavior
- fund_flow
- macro
- alternative
- llm_text
- ml_deep

## 重要说明

该注册表不是“有效因子清单”，而是“候选因子清单”。每个因子只是具备可研究定义、方向、家族、持有期、数据类型和可得日期约束。是否能进入实盘模型，必须等待真实 A 股 panel 的样本外检验、成本检验和容量检验。

## 纠错

1. 没有把所有学过的因子一次性放入注册表。优先选择定义清晰、方向明确、可得日期可审计的 68 个候选。
2. 对 `market` 类型因子允许 `availability_col` 为空，因为行情数据按截面日期天然可得；对财报、分析师、文本、基金、宏观、模型输出等要求可得日期字段。
3. 将 AI/LLM/Level2 因子设为 `high` 成本层级，避免它们在早期低成本模型中被误用。

## 下一步

1. 建立真实数据字段映射：把数据源字段转换为注册表中的 `column`。
2. 实现首批低成本因子计算器：先覆盖 value、quality、dividend、momentum、reversal、liquidity、volatility。
3. 在没有 Level2/文本/基金数据前，把 `cost_tier=high` 的因子排除出第一版实证。

