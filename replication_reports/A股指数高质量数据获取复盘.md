# A股指数高质量数据获取复盘

日期：2026-05-24

## 目标

建立 A 股主要指数数据层，用于后续指数择时、估值分位、宏观利率因子、ETF替代标的和高股息策略基准研究。本轮覆盖上证50、沪深300、中证500、中证1000、中证800、中证全指、中证红利、上证红利。

## 实现

新增脚本：

```text
Introduction-to-Quantitative-Finance/strategy_lab/a_share_index_data_harvester.py
```

采集数据：

- `daily_csindex/`：中证指数日线，含 OHLCV、成交额、样本数量、滚动市盈率与日线可用性标记。
- `valuation_latest_csindex/`：中证最新估值快照。
- `valuation_pe_lg/`：Legulegu 历史 PE，当前覆盖上证50、沪深300、中证500、中证1000、中证800。
- `valuation_pb_lg/`：Legulegu 历史 PB，当前覆盖上证50、沪深300、中证500、中证1000、中证800。
- `constituents_current/`：当前成分。
- `weights_latest/`：最新权重。
- `manifests/`：采集 manifest、合并 manifest 与质量摘要。

## 验证结果

采集结果：

- 合并 manifest 43 行，全部 `ok`，错误数 0。
- 指数列表 2340 行。
- 8 个指数日线全部落地，覆盖至 2026-05-22。
- 当前成分与最新权重全部落地，权重日期为 2026-04-30。
- 历史 PE/PB 覆盖 5 个指数，其中中证1000从 2014-10-17 起，其余宽基从 2005-2007 年起。

质量门槛：

- 日线、估值序列无重复日期。
- 收盘价无空值、无非正值。
- 成分和权重按 `date + asset` 复核后无重复键。
- 日线已标记 `is_full_ohlc_bar`、`is_close_only_bar`、`is_zero_volume_amount`。
- 合并质量摘要为 `quality_summary_2026-05-24T14-16-43.csv`。

## 复习与纠错

纠错 1：不能把成分股表按单独 `date` 判重。当前成分同一天天然有多只股票，质量检查已改为成分/权重表按 `date + asset` 判重。

纠错 2：不能把中证接口的基日和回溯 close-only 行当成完整日线。脚本已增加 `is_full_ohlc_bar`、`is_close_only_bar`、`is_zero_volume_amount`，后续 OHLC 信号必须过滤完整日线。

纠错 3：合并 manifest 时必须保留指数代码前导零。质量摘要层已强制把 `000852`、`000015` 等代码标准化为6位。

纠错 4：当前成分和最新权重不能用于历史 point-in-time 成分回测。它们只适合当前暴露、当前组合映射和样例研究；历史成分权重必须接入 Tushare/JoinQuant/Wind/中证历史文件。

纠错 5：不支持稳定历史 PE/PB 符号的指数，不应硬拉接口并把错误混入主数据。中证全指、中证红利、上证红利本轮只保留日线、最新估值、成分和权重。

## 结论

A 股核心指数级研究数据层已经可用，适合先开展估值分位、趋势/均线、波动率、利率敏感性和指数择时研究。严格历史成分权重回测仍需补齐 point-in-time 成分权重数据。
