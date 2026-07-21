# 量化知识复习与 GitHub 项目复盘

复盘日期：2026-05-23  
对应代码：`strategy_lab/multi_factor_research_framework.py`

## 本轮目标

1. 复习已有量化知识库，检查理解是否正确。
2. 学习 GitHub 公开 AI-Quant / Quant 项目的工程模式。
3. 将学习结果固化为可以支撑数百因子的量化研究框架。

## 已学习公开项目

- Microsoft Qlib
- Quantopian Alphalens
- VectorBT
- QuantConnect Lean
- Backtrader
- FinRL
- FinRL-Trading
- FinGPT
- FinRobot
- Microsoft RD-Agent
- QuantStats
- pysystemtrade
- RQAlpha
- vn.py

## 核心复盘

成熟开源项目共同指向一个结论：量化研究不是“找到一个指标然后回测”，而是一个可审计的研发系统。

本地已有资料库的优势是因子广度足够，包括基本面、行为、价量、高频、宏观、AI、文本和风险控制。但此前更偏“研究模块集合”，缺少统一总控层。本轮新增的 `multi_factor_research_framework.py` 把这些模块串成可执行框架。

## 新增代码能力

- 因子注册表。
- 可得日期审计。
- 因子方向统一。
- 横截面去极值。
- 横截面标准化。
- 行业和控制变量中性化。
- 多因子 IC 表。
- IC 稳定性汇总。
- 分组收益和高低组差。
- Top 分位换手。
- 平均相关矩阵。
- 相关性聚类。
- 因子质量分。
- 非冗余因子选择。
- 家族内合成。
- 家族间合成。
- 走步训练/测试切分。
- 目标权重生成。
- 组合收益和绩效摘要。
- GitHub 项目经验表。
- 数百因子模型质量门槛。

## 验证结果

已通过：

```text
python -X utf8 -m py_compile Introduction-to-Quantitative-Finance/strategy_lab/multi_factor_research_framework.py
```

烟测结果：

```text
smoke ok {'registry': 4, 'ic_rows': 144, 'selected': 4, 'families': 4, 'weights': 576, 'ann': 0.333}
```

烟测覆盖注册表、方向统一、去极值、标准化、中性化、IC、换手、质量分、聚类、筛选、家族合成、总分、权重和绩效。

## 纠错

1. 发现 Pandas 相关矩阵底层数组可能只读，已修正为复制矩阵后写入对角线。
2. 明确数百因子模型不是“平铺所有因子求平均”，而是“候选库很大、实用模型很克制”。
3. 明确 AI 研究代理不能取代审计，必须保存实验轨迹和失败记录。
4. 明确 LLM 金融输出必须回到结构化数据和回测。
5. 明确向量化回测只能做初筛，最终还要执行层验证。

## 下一步建议

后续要把这个框架真正用于 A 股，需要接入真实数据：

- 股票日行情、复权价、成交额、停牌和涨跌停。
- 财报公告日和入库日。
- 指数成分和行业分类历史。
- 分析师一致预期历史快照。
- 融资融券、基金持仓、Level2 或分钟数据。
- 交易成本、冲击成本和容量假设。

接入真实数据后，第一轮实证应先跑 30-50 个高置信因子，不应直接上数百因子。

