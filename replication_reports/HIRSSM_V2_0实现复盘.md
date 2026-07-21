# HIRSSM V2.0 实现复盘

日期：2026-05-24

## 本轮目标

把 HIRSSM V2 从设计文档推进到可运行的指数级行业轮动 + 大小盘风格切换系统。当前版本只启用本地数据可严谨支持的模块：指数价格、成交额、风险、宽度、宽基 PE/PB、规则状态、专家打分、分层风险预算、月频调仓和成本敏感性回测。

## 主要产物

- 模型脚本：`strategy_lab/hirssm_v2_model.py`
- 默认配置：`configs/hirssm_v2_default.json`
- 运行输出：`outputs/hirssm_v2_0/`
- 主报告：`outputs/hirssm_v2_0/HIRSSM_V2_MODEL_RUN_REPORT.md`
- 目标权重：`outputs/hirssm_v2_0/target_weights_monthly.csv`
- 最新权重：`outputs/hirssm_v2_0/latest_target_weights.csv`
- 成本敏感性：`outputs/hirssm_v2_0/cost_sensitivity_summary.csv`
- 专家 RankIC：`outputs/hirssm_v2_0/expert_rank_ic.csv`
- 专家消融：`outputs/hirssm_v2_0/expert_ablation_summary.csv`
- 仓位暴露：`outputs/hirssm_v2_0/monthly_target_exposure.csv`

## 实现内容

- 数据层：读取中证/上证宽基指数、申万一级与重点二级行业指数、宽基 PE/PB 估值序列。
- 特征层：收益动量、均线偏离、均线斜率、突破、RSI、波动率、下行波动、滚动回撤、成交额 z-score、相对强弱。
- 状态层：使用规则状态识别 `risk_on_trend`、`risk_on_overheat`、`range_bound`、`risk_off_decline`、`crash_rebound`，并加入状态平滑。
- 专家层：趋势延续、相对强弱、估值修复、风险收缩、震荡反转、防御、流动性确认。
- 组合层：按状态分配 style/industry/defensive/cash sleeve，使用 rank + volatility scaling，执行单资产上限、no-trade band、成本扣减。
- 验证层：输出成本敏感性、年度收益、状态收益、RankIC、专家消融、仓位暴露。

## 关键验证结果

以 10bps 单边成本情景为主：

- 策略总收益：335.87%
- 年化收益：5.74%
- 年化波动：18.73%
- 最大回撤：-55.31%
- 平均现金权重：28.66%
- 平均风险资产暴露：71.34%
- 基准总收益：547.94%
- 基准年化收益：7.34%
- 基准最大回撤：-71.48%

解释：当前 V2.0 不是收益最大化版本，而是带风险预算和现金降档的研究原型。它明显降低了最大回撤，但在全样本总收益和信息比率上仍未战胜中证全指基准。

## 纠错记录

- 修复 f-string 语法错误。
- 修复防御 sleeve 覆盖风格 sleeve 的权重合并错误，改为袖内累加。
- 修复单资产上限截断后未再分配的问题，新增 cap-and-redistribute。
- 修复 style 资产未计算相对市场强弱的问题。
- 修复 RankIC 常量序列相关性告警。
- 给反转专家增加企稳门槛，避免把持续下跌误判为均值回归。

## 重要发现

- 风险收缩和防御专家有正贡献；移除它们会降低年化收益和 Sharpe。
- 反转专家当前仍是拖累项；消融显示移除 `range_reversal` 后年化收益提升约 0.91 个百分点，Sharpe 提升约 0.045。
- 行业流动性确认的 RankIC 为正，行业趋势和相对强弱信号接近中性。
- 风格层风险收缩和估值修复 RankIC 为正，但趋势、相对强弱、流动性和反转为负或偏弱。

## 当前限制

- 行业历史估值、历史成分权重、股票级大小盘因子、宏观利率敏感性、ML ranker 暂未启用。
- 行业成分和最新权重只用于数据状态解释，不用于历史回测。
- 消融结果使用全样本诊断，不能直接当作未来生产参数优化依据。
- 当前输出完整特征面板较大，后续应增加轻量运行模式。

## 下一步

1. 将 `range_reversal` 从默认生产候选中降级为观察专家，或改为滚动 RankIC 通过后才启用。
2. 增加 point-in-time 中国/美国利率、汇率、商品和 PMI 数据，启用宏观敏感性专家。
3. 增加 walk-forward 专家权重收缩，替代固定状态先验。
4. 增加参数稳定性和 PBO/Deflated Sharpe 报告。
5. 把完整特征面板写出改为可选，默认只保存治理和交易所需的轻量结果。
