# AI金融论文雷达筛选框架

## 类型

研究资料管线与论文优先级筛选工具。

## 目标

从大量 AI 金融论文摘要中筛选值得精读、复现和转化为量化研究资产的论文，避免把前沿摘要误用为投资结论。

## 输入

- 本地 arXiv radar markdown。
- 论文标题、日期、作者、标签、摘要、链接。
- 当前研究优先级：因子挖掘、组合风险、回测验证、A 股/ETF/基金适配。

## 筛选维度

高优先级：

- `Factor Mining`
- `Portfolio Optimization`
- `Risk Management`
- `Benchmark`
- `Asset Pricing`
- `Time Series`

关键词：

- alpha
- factor
- robust
- overfit
- benchmark
- CVaR
- transaction
- A 股
- Wasserstein
- Black-Litterman

降级因素：

- 非金融主任务。
- 无法获取数据。
- 只有收益、缺少风险和样本外。
- 方法太依赖黑箱且无解释。
- 不能转成可复用代码或验证框架。

## 主题桶

- `robust_optimization`
- `tail_risk`
- `network_risk`
- `reinforcement_learning`
- `derivatives_hedging`
- `portfolio_construction`
- `ai_governance_risk`
- `other`

## 实现

- `strategy_lab/ai_finance_radar.py`

核心函数：

- `parse_topic_markdown`
- `tag_counts`
- `classify_portfolio_risk_theme`
- `prioritize_papers`
- `build_watchlist`
- `topic_overview`

## 使用规则

论文进入精读前，先回答：

```text
1. 它解决哪个量化研究问题？
2. 数据是否可得或可替代？
3. 是否能复现关键实验？
4. 是否有样本外、鲁棒性、成本或风险约束？
5. 能转化成哪个代码模块、因子卡或验证工具？
6. 失败模式是什么？
```

只有通过这些问题，论文才从“雷达线索”进入“研究任务”。
