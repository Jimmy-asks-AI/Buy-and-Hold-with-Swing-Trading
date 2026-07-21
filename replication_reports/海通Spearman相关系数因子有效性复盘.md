# Replication: 海通 Spearman 相关系数因子有效性

Source: `资料/卖方金工研报/海通报告/3_选股因子研究系列（三）——从Spearman相关系数出发研究因子有效性.pdf`

Date: 2026-05-22
Status: replication plan, not yet reproduced

## Original Claim

报告认为，因子截面 Spearman 相关系数序列可以用于刻画因子有效性，但单期观测噪音很大。通过马尔科夫链 Kalman Filter 估计因子的“真实”相关性，可以比传统 p 值法更好地跟踪因子有效性，并在其组合实证中取得更好表现。

## Research Question

如何动态判断一个因子当前是否有效？

## Data Rebuild

需要：

- 历史沪深 300 成分股。
- 月末交易日。
- 月度未来收益。
- 因子池：ROE、DROE、PE 等常见因子。
- 行业分类。
- 沪深 300 指数收益。
- 可交易性、停牌、缺失、异常值处理。

## Method Rebuild

### Step 1: Cross-Sectional Rank IC

每个投资周期起始日：

```text
读取当日股票因子值
读取下一个月股票收益率
剔除缺失和异常
计算因子值与未来收益的 Spearman 相关系数
```

### Step 2: Rank IC Time Series

对每个因子形成月度 Rank IC 序列。

输出：

- 单期 Rank IC。
- 24 月滚动均值。
- 24 月滚动标准差。
- Rank IC 胜率。
- 滚动 ICIR。

### Step 3: p 值/移动平均法

复现传统方法：

```text
用过去 24 个月 Rank IC 判断因子是否有效。
```

### Step 4: Kalman Filter 方法

后续再实现。第一阶段先复现 Rank IC 序列和移动平均方法。

### Step 5: Portfolio Backtest

报告组合规则：

```text
股票池：沪深 300
周期：月度
因子选择：KF 法、p 值法、全因子法
组合：行业中性，选 50 只股票，等权
基准/对冲：沪深 300
评价：年化超额收益、年化波动率、信息比、最大回撤
```

## Key Outputs To Reproduce

- 各因子的 Rank IC 时间序列。
- 24 月移动平均。
- 因子选择结果。
- KF 法、p 值法、全因子法的组合表现。

## Falsification

必须检查：

- 因子池是否事后选择。
- 财务因子是否按公告日可得。
- 沪深 300 历史成分是否正确。
- 单期 Rank IC 是否受极端收益和异常值影响。
- p 值阈值是否样本内调参。
- KF 参数是否样本内优化。
- 组合表现能否分离因子选择和组合构建贡献。

## Verdict

当前已完成方法抽取。第一阶段复现应先做 Rank IC 时间序列与滚动统计，不急于实现 Kalman Filter。

## Reusable Artifacts

- `notes/研报阅读路线与Spearman因子有效性学习笔记.md`
- `strategy_lab/rank_ic_analysis.py`

