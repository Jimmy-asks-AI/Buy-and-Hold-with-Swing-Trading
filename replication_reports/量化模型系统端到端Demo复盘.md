# 量化模型系统端到端 Demo 复盘

日期：2026-05-23

## 本轮目标

把分散的因子工厂模块整合成一个可运行的量化模型系统入口，并完成端到端 demo。

## 新增产物

- `strategy_lab/quant_model_system.py`
- `configs/factor_factory_walk_forward_demo.json`
- `notes/量化模型系统使用说明.md`
- `factor_library/可治理多因子模型系统框架.md`
- `replication_reports/量化模型系统端到端Demo复盘.md`
- `outputs/quant_model_system_demo/`

## Demo 命令

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/quant_model_system.py demo --root Introduction-to-Quantitative-Finance --output-name quant_model_system_demo
```

## Demo 结果

Panel 校验：

- 必需字段：通过。
- `date-asset` 重复：0。
- 日期数量：320。
- 股票数量：80。
- 注册表可用因子列：33。
- 标签覆盖率：0.9375。

注册表审计：

- 候选因子：68。
- 家族：21。
- 重复因子：0。
- fail：0。
- warn：0。

Walk-forward：

- 行数：25600。
- 股票数：80。
- 日期数：320。
- 可用因子：33。
- split：10。
- ok split：10。
- 单边成本：20bps。

净值绩效：

```text
annual_return: 3.5350
annual_volatility: 0.2941
sharpe: 12.0210
sortino: 24.4890
max_drawdown: -0.2239
calmar: 15.7912
win_rate: 0.56
periods: 200
```

## 解释

该结果来自合成数据，不能解释为真实 A 股 alpha。它的意义是证明系统能完成：

1. 数据生成或接入。
2. 因子计算。
3. 数据契约校验。
4. 注册表审计。
5. walk-forward 样本外训练/测试。
6. 交易成本扣减。
7. 实验账本决策。
8. 摘要报告生成。

## 纠错

1. 之前系统仍偏脚本集合，现在已收束为 `quant_model_system.py` 入口。
2. 之前全样本 runner 可能造成选择偏差，现在已用 walk-forward 作为系统主路径。
3. 之前账本只支持普通 `performance.csv`，现已支持 `walk_forward_performance.csv` 的 net 指标。
4. 之前低成本因子 panel 缺少 `log_mkt_cap`，现已在因子计算器中生成。

## 下一步

继续推进方向：

1. 增加真实数据适配器。
2. 增加停牌、涨跌停、ST、新股过滤。
3. 增加行业/风格暴露约束。
4. 增加容量和冲击成本。
5. 增加 paper trading 状态文件。

