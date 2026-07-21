# Quant Research Master Index

更新时间：2026-05-24  
范围：`Introduction-to-Quantitative-Finance`

## 当前状态

- 阅读队列：188 / 188 已完成。
- 队列状态：全部 `corrected`。
- 复习状态：全部 `reviewed`。
- Artifact 状态：无缺失 artifact。
- 本地 Git 状态：当前目录不是 Git 仓库，未生成提交记录。

## 产物入口

- 学习笔记：`notes/`
- 因子卡与研究框架：`factor_library/`
- 策略与研究代码：`strategy_lab/`
- 复盘报告：`replication_reports/`
- 学习日志：`logs/research_log.md`
- 复习纠错日志：`logs/review_correction_log.md`
- PDF 文本缓存：`outputs/pdf_text_cache/`

## 当前产物数量

- `notes/`：60 个文件。
- `factor_library/`：47 个文件。
- `replication_reports/`：60 个文件。
- `strategy_lab/`：62 个文件。

## 主要能力模块

### 因子研究

- 价值、盈利、质量、成长、投资、杠杆、无形资产、分析师预期。
- 高频价量、Level2、大单、净换手率、尾盘、日内结构、LOB。
- 行为金融、情绪、关注度、薪酬、融资融券、准另类数据。
- 关系网络、供应链、共同分析师、基金隐含信息。
- 极端分位、尾部相关、因子失效、因子拥挤。

### 择时与组合

- 技术指标投票择时。
- 回归树与状态模型择时。
- 宏观敏感性、利率状态、EPU、T 值宏观因子。
- 动态风险控制、敞口边界、因子择时。
- 单因子极端组合、并联多策略组合、指数增强约束。

### 机器学习与深度学习

- GP/遗传规划因子挖掘。
- AlphaNet 类量价特征学习。
- RNN/GRU/LSTM 高频序列因子。
- 注意力与残差注意力。
- 混频深度学习。
- 正交层、RankIC 目标、滚动训练验证、样本外稳定性检查。

### 研究基础设施

- 阅读队列和学习状态管理。
- 文本提取缓存。
- 学习笔记、因子卡、复盘报告三层落盘。
- 每轮复习、反证、纠错。
- 代码语法检查与小样本烟测。

### GitHub 公开项目补充学习

- Qlib：端到端 AI quant workflow、Alpha158/Alpha360、监督学习/市场状态/RL。
- Alphalens：因子 tear sheet、IC、分组收益、换手、行业分组。
- VectorBT：向量化大规模参数和策略扫描。
- Lean / Backtrader：事件驱动、订单、费用、滑点和执行层。
- FinRL / FinRL-Trading：金融 RL 环境、智能体、目标权重接口。
- FinGPT / FinRobot：LLM 金融数据抽取、RAG、Agent 报告与工具编排。
- RD-Agent：自动化量化研发、因子和模型联合优化。
- QuantStats：绩效、风险、图表和报告分层。

新增总控代码：`strategy_lab/multi_factor_research_framework.py`。

### 因子工厂工程化

- 端到端运行器：`strategy_lab/factor_factory_runner.py`
- 实验账本与晋级规则：`strategy_lab/factor_factory_ledger.py`
- 默认配置：`configs/factor_factory_default.json`
- 因子注册表模板：`data_catalog/factor_registry_template.csv`
- 数据契约：`data_catalog/factor_factory_data_contract.md`
- 长期计划：`notes/自主多因子工厂长期运行计划.md`
- demo 输出：`outputs/factor_factory_demo/`
- 实验账本：`logs/factor_factory_experiment_ledger.csv`
- A 股候选因子注册表 v0：`data_catalog/a_share_factor_registry_v0.csv`
- 注册表审计：`strategy_lab/factor_registry_audit.py`
- 低成本因子计算器：`strategy_lab/a_share_low_cost_factor_builder.py`
- 低成本联调 demo：`outputs/low_cost_factor_factory_demo_rerun2/`
- Walk-forward 因子工厂：`strategy_lab/factor_factory_walk_forward.py`
- 一键量化模型系统入口：`strategy_lab/quant_model_system.py`
- 系统 demo：`outputs/quant_model_system_demo/`
- 系统说明：`notes/量化模型系统使用说明.md`
- 数据执行纸面跟踪说明：`notes/量化模型系统数据执行纸面跟踪说明.md`
- 真实数据接入前 SOP：`notes/真实数据接入前SOP.md`
- A 股真实数据抓取器：`strategy_lab/a_share_data_harvester.py`
- A 股真实数据获取状态：`data_catalog/a_share_data_acquisition_status.md`
- 数据凭证模板：`configs/data_credentials.example.json`
- CSV 编码与布尔转换工具：`strategy_lab/csv_io.py`
- 真实数据字段适配器：`strategy_lab/real_data_adapter.py`
- 纸面跟踪与漂移监控模块：`strategy_lab/paper_trading_monitor.py`
- point-in-time panel 构建器：`strategy_lab/a_share_panel_builder.py`
- 数据质量报告器：`strategy_lab/data_quality_report.py`
- 模型运行报告器：`strategy_lab/model_run_report.py`
- 系统 smoke test：`strategy_lab/run_quant_system_smoke_tests.py`
- 真实数据字段映射模板：`data_catalog/a_share_real_data_field_mapping_template.csv`
- 交易约束与真实数据层：`factor_library/交易约束纸面跟踪与真实数据Panel层.md`

## 已知资料修复记录

1. `2022-04-07_海通证券_选股因子系列研究（七十七）：改进深度学习高频因子的9个尝试.pdf`
   - 本地 PDF EOF/XRef 损坏。
   - 已使用外部可读 PDF 完成补学。
   - 来源：`https://bigdata-s3.wmcloud.com/researchreport/2022-04/4d413fa65d66469c8c2787f542269341.pdf`
   - Artifact：`round51_deep_highfreq_9_attempts_external_source`

2. `2023-05-14_海通证券_选股因子系列研究（八十七）：高频与日度量价数据混合的深度学习因子.pdf`
   - 本地文本层异常。
   - 已使用海通官网同名 PDF 补充学习。
   - 来源：`https://www.htsec.com/jfimg/colimg/upload/20230522/1741684713966765.pdf`
   - Artifact：`round48_duration_profit_intangible_mixedfreq`

## 下次调用方式

当需要我继续做量化研究时，可以直接指定：

```text
使用 quant-research-master，从 QUANT_RESEARCH_MASTER_INDEX.md 开始，调用已有 notes/factor_library/strategy_lab/replication_reports。
```

如果目标是做实证或回测，优先从 `strategy_lab/` 中选择对应模块，再回到 `factor_library/` 查定义和质量门槛，最后用 `replication_reports/` 检查失败场景。

## 研究质量门槛

任何新因子或策略入库前必须满足：

- 数据可得日期明确。
- 无明显未来函数和幸存者偏差。
- 原始效果、正交效果和增量效果分开报告。
- 样本内、样本外、分年度、分市场状态验证。
- 换手、交易成本、容量和流动性约束。
- 参数稳定性和失败场景记录。
- 每轮学习后必须有复习和纠错记录。

## 2026-05-24 指数数据层更新

- 新增指数采集器：`strategy_lab/a_share_index_data_harvester.py`
- 新增数据状态：`data_catalog/a_share_index_data_acquisition_status.md`
- 新增复盘报告：`replication_reports/A股指数高质量数据获取复盘.md`
- 新增原始数据层：`data_raw/index/akshare_csindex/`
- 已获取指数：上证50 `000016`、沪深300 `000300`、中证500 `000905`
- 已落地数据：中证指数日线、最新估值快照、历史PE/PB、当前成分、最新权重、manifest、quality summary
- 关键约束：OHLC信号需过滤 `is_full_ohlc_bar == True`；历史成分权重回测仍需补充 point-in-time 数据源。

### Extension Update

- 指数数据层已扩展至 8 个指数：`000015`、`000016`、`000300`、`000852`、`000905`、`000906`、`000922`、`000985`。
- 合并质量摘要：`data_raw/index/akshare_csindex/manifests/quality_summary_2026-05-24T14-16-43.csv`
- 合并 manifest：`data_raw/index/akshare_csindex/manifests/manifest_consolidated_2026-05-24T14-16-43.csv`

## 2026-05-24 行业指数数据层更新

- 新增行业指数采集器：`strategy_lab/a_share_industry_index_harvester.py`
- 新增数据状态：`data_catalog/a_share_industry_index_data_acquisition_status.md`
- 新增复盘报告：`replication_reports/A股行业指数数据获取复盘.md`
- 新增原始数据层：`data_raw/index/akshare_sw_industry/`
- 已覆盖：31个申万一级行业 + 重点二级行业，共40个行业指数。
- 已落地：申万三级行业分类、2026-05-22行业分析快照、历史日线、当前成分、最新权重、manifest、quality summary。
- 质量摘要：`data_raw/index/akshare_sw_industry/manifests/quality_summary_2026-05-24T14-29-16.csv`

## 2026-05-24 HIRSSM V2 优化版

- 优化版模型设计：`notes/HIRSSM_V2优化版模型设计.md`
- 默认配置：`configs/hirssm_v2_default.json`
- 因子治理：`factor_library/HIRSSM_V2因子治理与入库规则.md`
- 优化复盘：`replication_reports/HIRSSM_V2模型优化复盘.md`
- 核心变化：因子家族聚类、状态条件化专家集成、专家滚动 ICIR 收缩、分层风险预算、风险覆盖、PBO/Deflated Sharpe 验证要求。

## 2026-05-24 HIRSSM V2.0 实现版

- 模型脚本：`strategy_lab/hirssm_v2_model.py`
- 运行报告：`outputs/hirssm_v2_0/HIRSSM_V2_MODEL_RUN_REPORT.md`
- 复盘报告：`replication_reports/HIRSSM_V2_0实现复盘.md`
- 主要输出：`target_weights_monthly.csv`、`latest_target_weights.csv`、`cost_sensitivity_summary.csv`、`expert_rank_ic.csv`、`expert_ablation_summary.csv`
- 当前结论：V2.0 已可运行并具备成本敏感性、专家 RankIC、专家消融和仓位暴露诊断；10bps 情景年化收益 5.74%，最大回撤 -55.31%，低于中证全指收益但回撤显著更低。
- 待纠错：`range_reversal` 专家在当前样本中拖累明显，后续应降级为观察专家或接入滚动 RankIC 门控。

### Audit Update

- 审计报告：`replication_reports/HIRSSM_V2_0审计修复报告.md`
- 已修复：回测起点早于第一笔交易、总收益漏算首日/首笔成本、年度收益漏算首日、失败反转专家默认启用。
- 修复后默认禁用：`range_reversal`
- 修复后 10bps 情景：年化收益 7.26%，年化波动 19.72%，最大回撤 -54.81%，平均现金权重 23.08%。
- 仍需治理：style 趋势、style 相对强弱、style 流动性和防御 sleeve 现金替代门槛。

### Pruning Update

- 剪枝脚本：`strategy_lab/hirssm_v2_expert_pruning.py`
- 剪枝报告：`replication_reports/HIRSSM_V2_0专家剪枝评估报告.md`
- 全局剪枝输出：`outputs/hirssm_v2_pruning_full/`
- 细粒度剪枝输出：`outputs/hirssm_v2_pruning_granular/`
- 最新默认禁用：`range_reversal`、`style_trend_continuation`
- 最新 10bps 情景：年化收益 8.67%，年化波动 19.05%，Sharpe 0.455，最大回撤 -54.55%。
- 观察候选：全局 `liquidity_overlay` 禁用后收益继续提升，但因行业层流动性 RankIC 为正，暂不默认删除。
