# 低成本 A 股因子计算器与工厂联调复盘

日期：2026-05-23

## 本轮目标

实现第一批可由常规行情和财务数据构建的低成本因子，并验证它们能接入注册表和因子工厂 runner。

## 新增产物

- `strategy_lab/a_share_low_cost_factor_builder.py`
- `configs/factor_factory_smoke.json`
- `outputs/low_cost_factor_demo/`
- `outputs/low_cost_factor_factory_demo/`
- `outputs/low_cost_factor_factory_demo_rerun2/`

## 低成本因子计算器

已覆盖 33 个原始字段，主要包括：

- 价值：EP、BP、SP、OCFP、FCFP、EBITDA/EV。
- 红利：股息率。
- 质量：ROE、ROA、毛利率、经营利润率、现金盈利质量、应计项。
- 成长：收入同比、利润同比、销售增长加速度。
- 投资：资产增长、资本开支增长、净股本发行。
- 杠杆：资产负债率、利息保障倍数、流动比率。
- 动量/反转：6M 动量、12M 动量、1M 反转、5D 反转。
- 流动性：换手、换手波动、成交额-市值残差。
- 波动：总波动、下行波动。
- 技术：效率比、DPO。

## 验证结果

合成低成本 panel：

```text
rows: 25600
factor_cols: 33
coverage_mean: 0.9447
```

接入因子工厂：

```text
n_rows: 25600
n_assets: 80
n_dates: 320
registered_factors: 68
selected_factors: 29
families: 11
```

账本结果：

```text
demo_003_low_cost_clean_warning_free
decision: revise_and_rerun
failed_gates: win_rate
```

## 纠错

1. 合成数据上的高收益不能解释为真实 alpha，已由账本记录为 demo。
2. 因子工厂 runner 原先在常数列相关系数计算时产生 numpy warning，已在 `multi_factor_research_framework.py` 中加入 `_safe_corr`。
3. 注册表 v0 有 68 个候选，但低成本计算器只覆盖其中 33 个；未覆盖项需要分析师、基金、Level2、文本、宏观或模型输出数据。
4. 本地 `data/` 目录主要是论文 JSON，没有发现可直接用于 A 股股票因子的真实行情/财务 panel。

## Blocker

真实 A 股多因子实证需要接入 point-in-time panel。最低字段见：

- `data_catalog/factor_factory_data_contract.md`

在真实数据接入前，当前系统只能完成工程链路验证，不能声称产生可投模型。

