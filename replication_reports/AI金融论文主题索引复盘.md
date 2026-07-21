# AI金融论文主题索引复盘

## 复盘对象

- `factor-mining.md`
- `portfolio-risk.md`

## 解析结果

本轮用 `strategy_lab/ai_finance_radar.py` 解析出 62 篇论文：

- factor-mining：4 篇。
- portfolio-risk：58 篇。

主要标签：

- Risk Management：48。
- Portfolio Optimization：23。
- Deep Learning：20。
- Benchmark：9。
- Volatility：8。
- LLM：8。
- Asset Pricing：8。
- Reinforcement Learning：8。

## 复盘结论

因子挖掘主题当前数量少，但 LLM 代码进化 Alpha 挖掘值得后续重点跟踪。它可以成为候选因子生成器，但必须严格接入因子评价、相关性去重、PBO、样本外和成本后回测。

组合风险主题覆盖面广，最值得迁移到现有研究系统的是：

- 分布鲁棒优化。
- 尾部风险与极端事件评估。
- Black-Litterman 和观点融合。
- 网络相关性与结构风险。
- 长期资产配置模拟中的漂移不确定性。

强化学习、量子网络、DeFi 和 AI 治理论文可以作为观察方向，但不能优先投入复现，除非后续研究问题明确需要。

## 本轮固化

新增：

- `strategy_lab/ai_finance_radar.py`

验证：

- `python -X utf8 -m py_compile strategy_lab\ai_finance_radar.py`
- 小样本烟测覆盖主题解析、标签统计、主题桶分类、优先级排序、watchlist 和 topic overview。

## 纠错

- 论文摘要不能作为策略有效证据。
- 前沿模型的价值必须以可复现、可验证、可转化为准。
- 论文雷达的职责是发现候选研究，不是替代研究。
