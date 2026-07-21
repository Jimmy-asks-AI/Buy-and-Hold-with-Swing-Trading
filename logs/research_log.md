# Research Log

## 2026-05-22 学习系统升级：自动续学

### Rule Change

用户要求每一轮结束后自动开始下一轮，不再等待“继续下一轮”的提示。

后续执行规则：

```text
每轮完成 -> 校验 -> 写日志 -> 更新队列 -> 选下一批 -> 自动开始下一轮
```

停止条件仅限：

- 需要用户提供数据、权限或方向选择。
- 文件或工具损坏。
- 上下文/执行时间边界需要汇总。
- 用户明确要求暂停。

## 2026-05-22 学习系统升级：强制复习纠错

### Learned

- 用户要求持续学习整个资料库，并且每次学习后必须复习、纠错。
- 学习系统不能只产出摘要，必须能记录反证、错误修正和队列状态。

### Artifacts Created

- `notes/学习复习纠错协议.md`
- `logs/review_correction_log.md`
- `scripts/learning_queue_manager.py`

### Rule Change

每个后续学习批次必须走：

```text
学习 -> 复习 -> 反证 -> 纠错 -> 沉淀 -> 更新队列
```

## 2026-05-22 第一轮：基础研究语言

### Learned

- 量化研究的本质是把投资假设转成可检验、可复现、可比较、可复盘的证据链。
- 数据必须检查时间口径：字段含义、产生时间、公告时间、交易可得性。
- 回测必须尊重当时可得信息，未来函数是最高优先级风险。
- 信号验证不等于完整回测，完整回测必须有仓位、成本、交易限制和净值曲线。
- 因子评价不能只看收益，要看 IC、Rank IC、ICIR、分组单调性、多空收益、换手、成本和稳健性。
- 因子负责排序，组合负责把排序变成可持有资产及权重。
- 风控不是事后止损，而是研究阶段对市场、行业、风格、流动性、模型、数据、执行和 AI 风险的提前约束。

### Artifacts Created

- `notes/量化研究基础总纲.md`
- `notes/回测检查清单.md`
- `notes/因子评价检查清单.md`
- `data_catalog/第一轮资料索引摘要.md`

### Next

- 读取 `05-指标因子信号策略.md`、`07-基准和超额收益.md`、`12-量化研究报告.md`。
- 读取 KDJ 和低 PB 示例，进入最小研究闭环。
- 将现有技术指标代码转成可复用的信号验证/回测模板。

## 2026-05-22 第二轮：表达层与最小信号验证

### Learned

- 指标、因子、信号、策略必须严格区分。
- KDJ 示例中的 K、D 是指标，金叉/死叉是信号，`main.py` 做的是信号后未来收益统计。
- 信号验证没有资金曲线、仓位、成本和基准，因此不能直接称为完整策略回测。
- 基准选择决定超额收益解释，错误基准会把 Beta、行业或风格暴露误判成 Alpha。
- 研究报告的职责是保存可复现证据，包括数据版本、代码版本、AI 协作、验证和审计记录。

### Artifacts Created

- `notes/指标信号策略与研究报告规范.md`
- `strategy_lab/signal_validation_template.py`

### Next

- 用 `signal_validation_template.py` 复现 KDJ 信号验证。
- 读取 `因子定义与评价模板.md` 和 `低PB因子研究系统示例.md`。
- 建立第一个因子评价模板代码。

## 2026-05-22 第三轮：KDJ 到第一个因子研究

### Learned

- KDJ 示例完整展示了 `数据 -> 指标 -> 信号 -> 未来收益标签 -> 分组统计`。
- 从 KDJ 到因子研究的关键跃迁是：从事件触发思维，切换到横截面打分排序思维。
- 低 PB 因子需要用 `-PB` 转成越高越好的价值分数。
- 回测配置是策略说明书，没有配置的回测不可审计。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：KDJ 不是策略；低 PB 不等于安全；技术指标公式说明不能直接照搬。

### Artifacts Created

- `notes/最小研究闭环_KDJ到低PB学习笔记.md`
- `strategy_lab/factor_evaluation_template.py`
- `factor_library/20日动量因子.md`
- `factor_library/低PB因子.md`

### Next

- 读取 ETF 动量轮动示例，学习从因子到组合。
- 读取 `资料/因子挖掘.md` 和 `资料/回测.md`。
- 选择第一篇海通因子研报，开始卖方研报复现。

## 2026-05-22 第四轮：ETF 动量轮动与研究系统

### Learned

- ETF 动量轮动是完整组合回测入门案例。
- 收益评价必须结合最大回撤、波动、夏普、换手和成本。
- 最小研究系统有七层：数据、因子/指标、信号、组合、回测、评价、报告。
- AI 协作要记录输入、输出、采纳、验证和拒绝原因。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：ETF 入门不等于无偏；动量收益不等于 Alpha；收益指标不能脱离路径。

### Artifacts Created

- `notes/ETF动量轮动与研究系统学习笔记.md`
- `factor_library/ETF_60日动量因子.md`
- `strategy_lab/rotation_backtest_template.py`

### Next

- 读取 `资料/回测.md`。
- 读取 `资料/因子挖掘.md`。
- 读取 `资料/投资组合优化与风控.md`。

## 2026-05-22 第五轮：回测、因子挖掘、组合风控与海通研报

### Learned

- `资料/` 下三份核心 Markdown 是索引，不是完整知识本体。
- 回测框架、因子挖掘工具、组合优化工具都必须服从同一研究证据标准。
- 海通第一篇因子研报提供了 J/K 动量反转分组框架。
- 报告中的 A 股反转结论必须限定在其样本和方法下，后续要扩展样本复现。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：索引不等于知识；反转有效不能无条件外推；组合优化先审输入。

### Artifacts Created

- `notes/回测因子挖掘组合风控学习笔记.md`
- `factor_library/三个月反转因子.md`
- `replication_reports/海通A股动量反转效应研究复盘.md`
- `strategy_lab/performance_metrics.py`

### Next

- 建立 `jk_momentum_reversal_template.py`。
- 读取海通 Spearman/Rank IC 报告。
- 读取海通组合约束报告。

## 2026-05-22 第六轮：研报阅读路线与 Spearman 因子有效性

### Learned

- 金工研报第一遍应先抽取结构，不先陷入公式。
- Spearman/Rank IC 衡量横截面因子排序和未来收益排序的关系。
- 单期 Rank IC 噪音很大，因子有效性更像需要估计的隐状态。
- p 值法可理解为 24 个月移动平均型因子选择法，稳定但滞后。
- 海通报告用 Kalman Filter 尝试在稳定性与时效性之间折中。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：Rank IC 不是稳定真相；组合表现不能单独证明因子选择方法有效。

### Artifacts Created

- `notes/研报阅读路线与Spearman因子有效性学习笔记.md`
- `replication_reports/海通Spearman相关系数因子有效性复盘.md`
- `strategy_lab/rank_ic_analysis.py`

### Next

- 自动进入下一轮：读取海通因子正交报告和因子加权/正交/择时报告。

## 2026-05-22 第七轮：因子正交、加权与择时

### Learned

- 多因子等权相加不代表实际等暴露，因子相关性会扭曲暴露。
- 逐步正交通过回归残差剔除已知因子的线性解释部分。
- 正交顺序非常重要，先放入的因子保留优先级更高。
- z-score 条件下，复合因子 IC 加权可用 Fama-MacBeth 回归理解。
- 因子择时可理解为因子溢价对条件变量的回归。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：等权多因子不等于等暴露；正交不是无损魔法；因子择时先按回归问题理解。

### Artifacts Created

- `notes/因子正交加权择时学习笔记.md`
- `replication_reports/海通因子正交与加权择时复盘.md`
- `strategy_lab/factor_orthogonalization.py`

### Next

- 自动进入下一轮：读取个股加权方式对比报告和组合约束影响报告。

## 2026-05-22 第八轮：个股加权方式与组合约束

### Learned

- 同一因子排序可以被多种权重函数表达，权重函数本身会改变收益、换手、容量和风险暴露。
- 连续因子倾斜能更强表达信号，但通常伴随更高换手、更小容量和更高成本敏感性。
- 市值加权容量更好但 alpha 表达更弱；逆波动加权会引入低波、行业和市值暴露。
- 指数增强约束会重塑可行股票集合，因子在全市场有效不等于在约束组合内有效。
- 组合构建必须把 `signal -> base_weight -> tilt -> constraints -> trading` 分层评估。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：高 IC 因子不一定可交易；高收益权重方法不一定扣费后优；约束不是后处理，而是模型有效域的一部分。

### Artifacts Created

- `notes/个股加权与组合约束学习笔记.md`
- `replication_reports/海通个股加权与组合约束复盘.md`
- `strategy_lab/portfolio_weighting_constraints.py`

### Next

- 自动进入下一轮：读取因子失效预警和因子拥挤度改进报告。

## 2026-05-22 第九轮：因子拥挤与失效预警

### Learned

- 因子拥挤是因子投资资金追捧导致收益性或稳定性下降的内生风险。
- 估值价差、配对相关性、长期累计收益和因子波动率可组成拥挤度指标。
- 复合拥挤度在多数因子上与未来中长期因子收益负相关。
- 拥挤度与未来波动的正相关在 A 股并不普遍，不能照搬海外结论。
- 配对相关性改进算法更能预测未来波动，但可能与未来收益正相关。
- “多头/市场”因子波动率比“多空/市场”更有收益预测能力。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：拥挤度不是短线买卖信号；高拥挤不等于立即下跌；配对相关性更偏风险监控。

### Artifacts Created

- `notes/因子拥挤与失效预警学习笔记.md`
- `factor_library/因子拥挤度监控因子.md`
- `replication_reports/海通因子拥挤与失效预警复盘.md`
- `strategy_lab/factor_crowding_monitor.py`

### Next

- 自动进入下一轮：读取因子拥挤度扩展报告。

## 2026-05-22 第十轮：因子拥挤度扩展

### Learned

- 拥挤度可扩展到资产集中度和机构持仓集中度。
- 资产集中度通过 PCA/吸收比率刻画因子多头组合对共同波动的解释能力。
- 资产集中度对未来因子收益有一定预测能力，但部分因子方向相反。
- 机构持仓类指标逻辑直接，但披露滞后，原始因子集合预测能力偏弱。
- 机构持仓市值比在正交因子集合上有一定预测力。
- 大部分拥挤指标对选股空间不太敏感，但配对相关性对股票池极度敏感。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：资产集中度不是普通波动率；持仓拥挤不能忽略披露滞后；股票池是拥挤度模型参数。

### Artifacts Created

- `notes/因子拥挤扩展学习笔记.md`
- `replication_reports/海通因子拥挤扩展复盘.md`
- `strategy_lab/factor_crowding_monitor.py`
- `factor_library/因子拥挤度监控因子.md`

### Next

- 自动进入下一轮：读取北上资金与边际定价相关材料。

## 2026-05-22 第十一轮：资金流向与大资金行为

### Learned

- 资金流向因子来自委托流和逐笔成交，反映短期微观供求。
- Wind 资金流向字段可分为流入、流出、净主动买入、开盘/尾盘、主力净流入等 8 类。
- 资金流向因子有效周期短，华泰报告以 10 个交易日持仓作为测试起点。
- 资金流向因子与换手率关系最强，但残差测试显示仍可能保留增量信息。
- 大单因子与异动股票前五大买入席位占比相关，可作为大资金行为代理。
- 交易异动股票长期收益偏负，剔除或正交异动哑变量后大单因子效果改善。
- 基金重仓和基金增持在报告期前收益更好，大单因子可用于预测基金建仓行为。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：资金流向因子不能按长线因子周期评价；大单买入不是无条件利好；基金持仓数据不能前视。

### Artifacts Created

- `notes/资金流向与大资金行为学习笔记.md`
- `factor_library/资金流向与大单行为因子.md`
- `replication_reports/华泰海通资金流向与大资金行为复盘.md`
- `strategy_lab/order_flow_factors.py`

### Next

- 自动进入下一轮：读取买卖单数据中的 Alpha 和主动买入行为报告。

## 2026-05-22 第十二轮：买卖单 Alpha 与主动买入行为

### Learned

- 逐笔成交可通过叫买序号和叫卖序号还原为买卖单。
- 动态大单阈值用个股日内订单金额均值加 N 倍标准差，比固定金额阈值更稳健。
- 正交后大买成交金额占比、大买减大卖成交金额占比具有较强选股能力。
- 买卖单集中度因子不依赖大单阈值，但主要在中小盘更有效。
- BS 标志可构建主买占比和主买强度，但涨跌停分钟会导致方向解释反直觉。
- 主买占比和日内净主买强度正交后有月度 alpha，沪深 300 内多头效应更强。
- 高频因子提高调仓频率后毛收益可能上升，但必须扣成本后判断。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：不能只用 BS 标志理解逐笔数据；大单阈值不是越严格越好；高频毛收益不等于可交易净收益。

### Artifacts Created

- `notes/买卖单Alpha与主动买入行为学习笔记.md`
- `replication_reports/海通买卖单Alpha与主动买入行为复盘.md`
- `strategy_lab/order_flow_factors.py`
- `factor_library/资金流向与大单行为因子.md`

### Next

- 自动进入下一轮：读取日内分时成交、交易意愿、下跌托底相关报告。

## 2026-05-22 第十三轮：日内分时成交、交易意愿与下跌托底

### Learned

- 1 分钟成交金额、成交笔数和收益率可构建平均单笔成交金额、流入/流出金额占比和大单分钟资金流因子。
- 平均单笔流出金额占比有较强正向选股能力，正交常见风格后仍有效。
- 大单净流入率和大单驱动涨幅在报告中为负向预测，不能把大单买入机械解释为利好。
- 盘口委托快照提供未成交交易意愿，前 1 档开盘后净委买变化率最能体现集中反馈。
- 净委买变化率波动率和偏度有增量信息，但部分信号更偏中小盘，且与均值因子重合。
- 委托成交相关性把价格变化和买入意愿结合，相关性越低未来超额收益越强。
- “股价下跌、净委买上升”的托底形态后续收益最强，“股价下跌、净委买下降”最弱。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：高频委托/成交因子主要是个股截面因子；盘口挂单不等于真实成交；大单净流入不是无条件利好；生产级净委买变化率必须处理盘口档位迁移。

### Artifacts Created

- `notes/日内分时成交交易意愿与托底因子学习笔记.md`
- `replication_reports/海通日内分时交易意愿与托底复盘.md`
- `strategy_lab/order_flow_factors.py`
- `factor_library/资金流向与大单行为因子.md`

### Next

- 自动进入下一轮：继续读取高频因子梳理、知情交易与主买主卖、逐笔交易有效信息相关报告。

## 2026-05-22 第十四轮：逐笔成交、知情交易与有效信息

### Learned

- 知情交易用预期外分钟收益过滤主买主卖；残差为正时主动卖出定义为知情主卖，残差为负时主动买入定义为知情主买。
- 开盘后知情主卖占比、收盘前知情主买占比、开盘后知情净主买占比都具有截面信息，但方向不能按名称直觉解释。
- 逐笔有效因子正交后月均 IC 多在 0.03-0.04，半月调仓通常优于月度，周度并不普遍更好。
- 大买成交金额占比与大卖成交金额占比不对称；大买稳定正向，大卖整体不稳定。
- 按对手方拆分后，大卖单对小买单更偏负面；大买单被大卖单承接仍可能偏正面。
- 逐笔过滤后重构 K 线可提升部分分钟因子，但与大单因子存在信息重叠。
- 大单因子对月内早期收益预测更强，持有期越长预测力越弱。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：知情主买不是无条件正向；大买大卖不能对称解释；大单阈值不是越严格越好；单因子显著不等于组合显著改善。

### Artifacts Created

- `notes/逐笔成交知情交易与有效信息学习笔记.md`
- `replication_reports/海通逐笔成交知情交易与有效信息复盘.md`
- `strategy_lab/order_flow_factors.py`
- `factor_library/资金流向与大单行为因子.md`

### Next

- 自动进入下一轮：读取短周期高频因子、调仓收益增强和高频因子现实约束相关报告。

## 2026-05-22 第十五轮：短周期高频因子与调仓优化

### Learned

- 高频因子的低频化应优先统一到分钟级，再按日内时段聚合为日度截面信号。
- 买入意愿因子把净委买变化金额和净主动买入金额合并，开盘买入意愿强度在报告中比单独净主动买入更稳。
- 机器学习可用于搜索高频表达式，但表达式必须补经济解释、相关性约束和样本外验证。
- 高频因子信息半衰期短，直接提高调仓频率容易被成本吞噬。
- 延迟调仓是一种更实用的组合层机制：延迟卖出短期强势的计划卖出股票，延迟买入短期弱势的计划买入股票。
- 高频因子若主要贡献来自空头端，应优先做短端过滤、后置剔除或降权，而不是直接当多头信号。
- 高频因子的现实约束包括交易所字段差异、执行价格、冲击成本、容量、涨跌停和停牌不可交易。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：高频 IC 高不等于可交易；调仓频率不是越高越好；空头端有效不等于多头端可买；延迟调仓必须保持买卖金额平衡。

### Artifacts Created

- `notes/短周期高频因子与调仓优化学习笔记.md`
- `replication_reports/海通短周期高频因子与调仓优化复盘.md`
- `strategy_lab/rebalance_optimization.py`
- `factor_library/高频因子组合应用与调仓优化.md`

### Next

- 自动进入下一轮：读取高频因子空头效应、多头失效修正、剔除高频空头组合后的指数增强策略相关报告。

## 2026-05-22 第十六轮：高频空头效应与多头失效修正

### Learned

- 高频因子常见问题是整体 IC 显著但多头端失效，收益主要来自空头组。
- 空头端有效因子不应直接当作多头买入信号，更适合做示性变量、剔除、降权或延迟买入。
- 加权 IC 通过提高多头组权重，能识别真正适合多头模型的高频因子。
- 多头失效可能来自非线性关系，可用二次项、四次项或 RBF 升维修正，但过拟合风险明显。
- 沪深300多因子空头剔除中，正交因子、成分股外样本、ICIR 或等权 zscore 复合较稳；剔除比例通常控制在 4%-10%。
- 中证500中，事后剔除通常比事前剔除更稳，组合复合通常优于因子复合。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：整体 IC 不能替代多头组 IC；空头端收益不能机械转为多头 alpha；多因子空头组合不能简单取所有因子并集；非线性升维必须控制样本外风险。

### Artifacts Created

- `notes/高频空头效应与多头失效修正学习笔记.md`
- `replication_reports/海通高频空头效应与多头失效修正复盘.md`
- `strategy_lab/rebalance_optimization.py`
- `factor_library/高频空头效应与多头失效.md`

### Next

- 自动进入下一轮：继续读取因子空头收益转化、组合优化与多因子模型诊断相关材料。

## 2026-05-22 第十七轮：空头收益转化与因子敞口上限

### Learned

- A 股多数因子存在较强空头效应，空头收益占比高的因子不一定适合直接做多多头组。
- 单因子空头收益可用逆向剔除转化，即剔除短端股票后在剩余股票池内选股。
- 多因子模型中，单因子收益能否转化取决于因子 IC、因子间协方差、IC 协方差和权重。
- 正交处理能降低重复计量，并过滤成交额这类合成型因子。
- 因子敞口上限同时提高潜在收益和风险，真正决定收益的是组合实际敞口。
- 沪深300因行业市值集中，行业中性约束使实际敞口较难打满；中证500更容易打满敞口。
- 用过去实际敞口均值或中位数滚动设定上限，比直接使用历史最优上限更稳健。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：空头型因子不能直接当多头因子；单因子有效不等于多因子有效；预设敞口上限不等于实际敞口；敞口上限不是越高越好。

### Artifacts Created

- `notes/空头收益转化与因子敞口上限学习笔记.md`
- `replication_reports/海通空头收益转化与因子敞口上限复盘.md`
- `strategy_lab/portfolio_weighting_constraints.py`
- `factor_library/空头收益转化与敞口预算.md`

### Next

- 自动进入下一轮：读取风险模型、组合优化实证和机器学习组合优化相关材料。

## 2026-05-22 第十八轮：结构化风险模型与周频组合优化

### Learned

- 结构化风险模型将股票协方差拆为共同因子风险和特异性风险。
- 风格因子暴露需要去极值、缺失值填充、标准化和必要正交，正交不能过度使用。
- 因子协方差和特异性方差需要 Newey-West、特征值/PSD、压缩和偏误校准等处理。
- 偏误统计量可检验风险预测是否系统低估或高估。
- 周频 AlphaNet 必须匹配周频风险模型，月频风险模型不能直接套用。
- 风险厌恶系数提高会降低跟踪误差和回撤，但会牺牲年化超额收益。
- 风格和行业约束要服务目标：控风险时有用，约束过多会限制 alpha 表达。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：风险模型不是事后解释；短周期策略必须匹配预测期限；机器学习 alpha 必须做风格归因；约束越多不等于越好。

### Artifacts Created

- `notes/结构化风险模型与周频组合优化学习笔记.md`
- `replication_reports/华泰结构化风险模型与周频组合优化复盘.md`
- `strategy_lab/risk_model_tools.py`
- `factor_library/结构化风险模型与组合优化.md`

### Next

- 自动进入下一轮：读取基金重仓、机构持仓、价值组合和基本面因子组合相关材料。

## 2026-05-23 第十九轮：机构持仓、价值组合与盈利加速

### Learned

- 基金重仓超配因子有截面信息，但方向不稳定；不能直接当稳定正向 alpha。
- 基金持仓因子应加入滚动溢价方向门控，或使用方向延续性更强的基金分组。
- 深度价值需要 `price < NCAV`、等待期和长持有期；A股满足条件股票少，容量有限。
- 低估值组合需要拆分为纯价值暴露、行业选择收益和个股选择收益。
- 有基本面支撑的低估值组合比单纯低估值更完整，核心是低估值与盈利/增长交集，再叠加低关注度和位序估值过滤。
- 盈利加速 EAV 相比 EAA/EAP 更稳，且正交后仍有增量信息；但必须按财报披露可得日期处理。
- 高增长组合的高收益伴随小市值、高估值、SUE、TMT/制造等风格和行业暴露，入库前必须归因。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：基金重仓超配不是稳定正 alpha；低估值不是充分买入条件；NCAV 筛选方向应为股价低于 NCAV；季报因子不能用季度末日期提前获得；小组合高收益必须附容量和偏差检查。

### Artifacts Created

- `notes/机构持仓价值组合与盈利加速学习笔记.md`
- `replication_reports/海通机构持仓价值组合与盈利加速复盘.md`
- `strategy_lab/fundamental_factor_portfolios.py`
- `factor_library/机构持仓价值与盈利加速因子.md`

### Next

- 自动进入下一轮：读取因子季节效应、主动成交隐藏信息、深度学习高频特征工程及相关后续报告。

## 2026-05-23 第二十轮：季节效应、主动成交与深度学习特征工程

### Learned

- 因子季节效应适合做风格暴露微调和卫星策略开关，不适合独立作为稳定交易系统。
- 2、3、5、8月偏小盘，5、6月偏成长，基本面因子在岁末年初、年报/一季报后和中报后更强，下半年反转、低波、低换手更强。
- 节前偏确定性，高盈利、低换手、大盘蓝筹更强；节后偏弹性，小市值、高增长、反转更强。
- 买卖单主动成交度必须按订单级重构；小单主动成交度比大单/中单更有选股能力。
- 小买单主动成交度正交行业、市值、换手、反转后仍有效，但在已有深度学习高频因子的模型中边际贡献不稳定。
- 深度学习高频特征工程需要按“构建、处理、归因、筛选”拆解；偏度调整、去极值、动态特征筛选能降低噪声和冗余。
- 多颗粒度输出集成是强基线，双向AGRU进一步改善RNN遗忘问题；残差学习网络复现未显著优于输出集成。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：季节效应不能机械外推；逐笔BS方向不能替代订单主动成交度；静态全样本特征贡献度有未来信息；复杂深度模型不天然优于简单输出集成。
- 数据问题：`2023-05-14_高频与日度量价数据混合的深度学习因子.pdf` 本地文本层损坏，本轮未标记完成。

### Artifacts Created

- `notes/因子季节效应主动成交与深度学习特征工程学习笔记.md`
- `replication_reports/海通季节效应主动成交与深度学习特征工程复盘.md`
- `factor_library/季节效应主动成交与深度学习特征工程.md`
- `strategy_lab/factor_timing_seasonality.py`
- `strategy_lab/ml_feature_engineering.py`
- `strategy_lab/order_flow_factors.py`

### Next

- 自动进入下一轮：读取买入评级因子、红利投资、组合规模交易成本和组合约束后续报告，同时安排损坏PDF的OCR/重取。
## 2026-05-23 第二十一轮：分析师评级、红利投资、容量成本与模型动物园

### Learned

- 买入评级因子在 2021 年后衰减，不能再用全量买入报告粗糙入模；点评和深度报告、新增买入、SUE 基本面支持是更稳健的三层过滤。
- 分析师观点适合大盘股 Smart Beta 和指数增强的子因子，需与覆盖度、盈利支持和风格归因共同使用。
- 红利投资的核心是高分红意愿、低估值、成熟经营和现金回报；红利风格在美债利率上行、市场波动放大、社融同比下降时相对成长更优。
- 社融下降只支持红利相对价值，不保证权益绝对收益；利率下行且波动下降时，成长风格显著占优，红利不宜主配。
- 红利+成长、红利+低波、红利+分红潜力代表不同红利纯度，不能混为同一种高股息策略。
- 全市场因子 IC 会高估大规模、周频和高换手策略；需要用成交时段、组合规模、盘口流动性和大单冲击重新评估可交易 IC。
- 深度学习模型动物园显示 BiATCN、TCN、Transformer、DBiAGRU 是强候选，但复杂模型不天然更优；等权集成是更稳健的强基线。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：买入评级需要报告类型和基本面支持；红利择时是相对风格择时；红利+分红潜力不是纯红利；容量校正要区分月频全天和周频开盘半小时；深度学习模型应先做相关性和风格暴露诊断。

### Artifacts Created

- `notes/分析师评级红利投资容量成本与模型动物园学习笔记.md`
- `factor_library/分析师评级红利容量与模型集成因子.md`
- `replication_reports/海通分析师评级红利投资容量成本与模型动物园复盘.md`
- `strategy_lab/analyst_rating_factors.py`
- `strategy_lab/dividend_strategy.py`
- `strategy_lab/trading_cost_capacity.py`
- `strategy_lab/model_zoo_ensemble.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\analyst_rating_factors.py strategy_lab\dividend_strategy.py strategy_lab\trading_cost_capacity.py strategy_lab\model_zoo_ensemble.py`
- 小样本烟测覆盖：买入评级缺失月份补全、SUE 支持、红利组合权重封顶、红利宏观状态、容量 IC、大单冲击、模型集成。

### Next

- 自动进入下一轮：读取组合约束、指数增强、机器学习组合实现和剩余排队材料，并继续处理损坏 PDF 的 OCR/重取问题。
## 2026-05-23 第二十二轮：研究对象与条件期望因子择时

### Learned

- 量化研究的主线是把投资想法转为可计算、可验证、可反驳的问题，不等于量化交易或程序化交易。
- 最小研究闭环先验证信号是否有信息，完整回测再处理资金曲线、权重、成本、滑点、停牌涨跌停和基准。
- 研究对象、资产池、剔除规则和基准必须匹配；否则超额收益可能只是股票池或风格 beta。
- 因子择时的本质不是预测指数涨跌，而是预测因子收益和因子收益协方差，并据此动态调整因子权重。
- 条件期望模型把历史因子收益的无条件均值/协方差，修正为给定市场状态下的条件均值/条件协方差。
- 条件变量包括市场涨跌幅、波动率、估值、换手率和利率；波动率类变量在报告中最稳定。
- AIC 前向筛选用于控制条件变量数量，避免样本内过拟合。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：量化研究不等于自动交易；股票池和基准不匹配会制造假超额；因子择时不是仓位择时；条件变量不是越多越好；择时模型价值应同时看风格切换期表现。

### Artifacts Created

- `notes/量化研究对象与条件期望因子择时学习笔记.md`
- `factor_library/条件期望因子择时.md`
- `replication_reports/课程研究对象与条件期望择时复盘.md`
- `strategy_lab/conditional_factor_timing.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\conditional_factor_timing.py`
- 小样本烟测覆盖条件均值/协方差、最大化 IC 权重、AIC 筛选、滚动 live-safe 权重和市场条件变量构造。

### Next

- 自动进入下一轮：继续读取 2026 量化趋势、30 天课程、深度概念手册和 AI/回测过拟合相关材料，扩展通用研究治理能力。

## 2026-05-23 第二十三轮：AI量化研究治理与CSCV回测过拟合

### Learned

- AI 是量化研究助手，不是投资责任主体；它能加速资料整理、假设拆解、代码草稿和检查清单，但不能替代数据、未来函数、样本外、成本和实盘可成交验证。
- 机器学习选股的完整流程是数据获取、特征提取、数据变换、模型训练、模型选择和模型预测；在量化中必须继续落到组合、交易、成本、风险和基准超额。
- 训练过拟合与回测过拟合不同；后者来自大量策略、参数、因子和模型候选中的历史最优选择偏差。
- CSCV/PBO 用组合式样本切分衡量训练集赢家在测试集的相对排名；训练赢家若落入测试集后半区，就记为一次回测过拟合。
- 模型复杂度不是研究质量；线性、树模型、集成学习、深度学习、KNN、聚类和降维都必须通过同一套时点、样本外、过拟合、成本和风险门槛。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：AI 输出不能直接当研究结论；训练交叉验证不等于回测过拟合控制；CSCV/PBO 排名方向必须显式定义；分类准确率不等于交易收益；标准化、降维和特征选择不得使用全样本参数。

### Artifacts Created

- `notes/AI量化研究治理与CSCV过拟合学习笔记.md`
- `factor_library/回测过拟合概率与研究治理.md`
- `replication_reports/华泰AI选股框架与CSCV过拟合复盘.md`
- `strategy_lab/backtest_overfit_pbo.py`
- `strategy_lab/research_governance.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\backtest_overfit_pbo.py strategy_lab\research_governance.py`
- 小样本烟测覆盖 CSCV/PBO、manifest 审计、时间顺序切分、walk-forward、训练集标准化和二分类 AUC。

### Next

- 自动进入下一轮：继续读取课程迭代记录、常见问题、概念例子库、毕业作业和项目材料索引，补全研究训练体系与教学型知识库。

## 2026-05-23 第二十四轮：课程迭代、材料路线与研究训练体系

### Learned

- 课程迭代材料的核心价值是把初学者常见误区显式化，并形成“误区 -> 修正 -> 材料 -> 作业”的训练闭环。
- 量化研究训练应先完成概念门、数据门、因子门、回测门和审计门，再进入机器学习、LLM 因子挖掘或实盘系统。
- 项目材料必须按任务路由，不是按文件夹顺序通读；第一层建立研究语言，第二层跟任务选读，第三层作为进阶专题。
- KDJ 示例的价值是最小研究闭环，不是技术指标本身；毕业作业的价值是研究系统入门标准，不是策略上线标准。
- AI 协作记录是研究审计的一部分，应记录 AI 做了什么、采纳什么、拒绝什么以及如何验证。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：高级材料不能前置替代基础研究闭环；课程作业不是收益承诺；材料索引应作为任务路由表；AI 审计不是形式主义。

### Artifacts Created

- `notes/量化研究训练体系与材料路线学习笔记.md`
- `factor_library/研究训练质量门与作业模板.md`
- `replication_reports/课程迭代材料与研究训练体系复盘.md`
- `strategy_lab/research_training_system.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\research_training_system.py`
- 小样本烟测覆盖材料路线、测验题库、系统模块、作业模板、作业审计、AI 审计和 readiness 汇总。

### Next

- 自动进入下一轮：读取剩余 P0 中的 NLP 入门资料和基金研究框架，补全文本数据/情绪研究与基金标签体系。

## 2026-05-23 第二十五轮：NLP文本因子、基金分类标签与权益基金池

### Learned

- NLP 在量化中的核心是把非结构化文本变成可得时点明确、实体映射明确、可聚合验证的结构化信号。
- 文本因子最容易出错的是时间戳和交易可得性，而不是分词或模型本身。
- 基金评价必须先找对比较锚；合同分类、实际仓位、选股范围和投资策略不一致会制造错误比较。
- 权益基金可按主动/被动/指数增强与宽基/风格策略/行业主题形成 3x3 分类。
- 主动权益基金池应采用基础池、优选池、观察池、投资池分层，并用定量绩效、持仓归因、基金经理调研和持续跟踪交叉验证。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：文本情绪不能跳过实体和时间对齐；通用 NLP 工具不能替代金融词典；基金跨类收益不可直接比较；主题基金必须用合同、基准和持仓交叉识别。

### Artifacts Created

- `notes/NLP文本因子与基金研究框架学习笔记.md`
- `factor_library/文本情绪因子与基金标签体系.md`
- `replication_reports/NLP入门与基金池标签框架复盘.md`
- `strategy_lab/nlp_text_factor_tools.py`
- `strategy_lab/fund_research_framework.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\nlp_text_factor_tools.py strategy_lab\fund_research_framework.py`
- 小样本烟测覆盖文本可得交易日、情绪聚合、实体匹配、权益基金分类、基础池、主题识别、同类评分、画像评分和基金池迁移。

### Next

- 自动进入下一轮：P0 已清空，转入 P1 材料，优先读取剩余基金研究、回测框架或 AI 金融论文主题索引。

## 2026-05-23 第二十六轮：技术指标信号验证与因子卡模板

### Learned

- 技术指标代码展示的是最小信号验证闭环：行情、指标、信号、未来收益标签、分组统计。
- `technical.py` 的 KDJ 金叉死叉信号只用过去数据，可作为信号；`main.py` 的未来收益只能作为评价标签。
- 125 个技术指标目录适合作为候选假设库和公式翻译练习，但不能机械批量优化。
- 因子卡模板必须扩展信息可得时点、方向、样本外、成本、容量、相关性和失败模式。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：技术指标信号验证不等于完整回测；未来收益缺失不应填 False；多指标多参数必须防过拟合；KDJ 平滑口径应显式记录。

### Artifacts Created

- `notes/技术指标信号验证与因子卡模板学习笔记.md`
- `factor_library/技术指标信号验证框架.md`
- `replication_reports/KDJ技术指标代码与125指标目录复盘.md`
- `strategy_lab/technical_signal_research.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\technical_signal_research.py`
- 小样本烟测覆盖 KDJ、交叉信号、未来收益标签、信号统计、公式算子表和 125 指标目录读取。

### Next

- 自动进入下一轮：继续 P1，优先进入 AI 金融论文主题索引或华泰多因子基础系列。

## 2026-05-23 第二十七轮：AI金融论文雷达之因子挖掘与组合风险

### Learned

- AI 金融论文主题文件是候选研究雷达，不是已复现证据。
- LLM 代码进化 Alpha 挖掘可以作为因子候选生成器，但扩大搜索空间后更需要 PBO、样本外和成本约束。
- 组合风险前沿集中在分布鲁棒、尾部风险、网络风险、强化学习组合、期权对冲、Black-Litterman 和 AI 治理风险。
- 论文优先级应服务可复现代码、因子卡、组合优化器或风险监控工具，而不是追逐标题新颖性。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：论文摘要不能当结论；LLM 挖因子不能替代因子评价；组合优化收益必须和风险约束、输入误差、成本及样本外一起看。

### Artifacts Created

- `notes/AI金融论文雷达_因子挖掘与组合风险学习笔记.md`
- `factor_library/AI金融论文雷达筛选框架.md`
- `replication_reports/AI金融论文主题索引复盘.md`
- `strategy_lab/ai_finance_radar.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\ai_finance_radar.py`
- 小样本烟测覆盖 62 篇论文解析、标签统计、主题桶分类、优先级排序、watchlist 和 topic overview。

### Next

- 自动进入下一轮：继续 P1，读取华泰多因子基础系列，补齐标准单因子测试体系。

## 2026-05-23 第二十八轮：华泰多因子体系与估值成长动量单因子测试

### Learned

- 多因子模型把股票收益拆成因子暴露、因子收益和特异收益，研究流程应连接收益预测、风险预测、组合优化和归因。
- 华泰单因子测试的标准证据链是分层组合、WLS 截面回归和 IC/Rank IC，不能只看某一个指标。
- 估值因子中 `BP` 综合最强，`SP`、`EV2EBITDA`、`PEG` 也值得保留；现金流类因子需要更谨慎。
- 成长因子整体弱于估值，但 `Sales_G_q`、`Profit_G_q`、`ROE_G_q` 可作为基本面补充。
- 动量报告在 A 股中实际更接近反转效应，`exp_wgt_return_6m`、`exp_wgt_return_3m`、`wgt_return_1m`、`return_1m` 方向为低过去收益更优。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：单因子测试不等于策略回测；低值更优因子必须反向；缺失填 0 只代表标准化后中性暴露；手工 Rank correlation 替代 SciPy 依赖。

### Artifacts Created

- `notes/华泰多因子基础与估值成长动量单因子测试学习笔记.md`
- `factor_library/华泰标准单因子测试框架.md`
- `replication_reports/华泰多因子体系与前三类单因子测试复盘.md`
- `strategy_lab/huatai_single_factor_test.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\huatai_single_factor_test.py`
- 小样本烟测覆盖反转因子方向转换、行业控制 WLS、Rank IC、分层多空差和 35 个华泰因子定义。

### Next

- 自动进入下一轮：继续华泰多因子系列，读取质量、情绪、波动率和换手率因子报告，补齐更多因子类别和多因子合成前的类别内筛选规则。

## 2026-05-23 第二十九轮：华泰换手率、波动率、财务质量与一致预期因子

### Learned

- 换手率因子方向多为负，`turn_1m`、`std_turn_1m` 在回归和 IC 中突出，`bias_turn_1m`、`bias_std_turn_1m` 在分层和稳定性上突出。
- 换手率短周期样本期可能优于 1 个月，3-5 个交易日值得单独做参数敏感性测试。
- 波动率因子整体强于估值、成长，接近换手率，弱于动量反转；特质波动率相关因子更值得关注。
- 财务质量因子变化慢，适合长线组合；单季度口径通常优于当年累计和 TTM，`qfa_roe` 是核心候选。
- 一致预期 EP/BP 及同行业位序因子较强，股票池表现大致为沪深300 > 全A > 中证500。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：换手率和波动率不能重复堆叠；价量强因子必须扣成本；质量因子阶段性强；一致预期必须看覆盖率和中性化。

### Artifacts Created

- `notes/华泰换手率波动率财务质量一致预期学习笔记.md`
- `factor_library/华泰换手率波动率质量一致预期因子库.md`
- `replication_reports/华泰单因子扩展类别复盘.md`
- `strategy_lab/huatai_single_factor_test.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\huatai_single_factor_test.py`
- 小样本烟测覆盖覆盖率、类别内 Spearman 相关性、相关性汇总、行业内位序、资产级环比变化和 81 个因子定义。

### Next

- 自动进入下一轮：读取华泰因子合成、海量技术因子和历史分位数因子报告，把单因子库推进到类别内合成、技术因子筛选和时序分位数表达。

## 2026-05-23 第三十轮：华泰因子合成、海量技术因子与历史分位数

### Learned

- 因子合成主要用于降低同类因子共线性和构造大类风格因子，不是多因子模型的必备步骤。
- 最大化 IC_IR 和最大化 IC 在多数类别中表现较好，但短窗口和协方差估计误差会造成不稳定权重。
- 等权最稳定，PCA 适合高相关因子，T=12 可作为动态权重窗口起点。
- 华泰筛出的 `Alpha3/13/15/16/44/50/55` 本质是价量背离，月频优先，不能只看多空收益。
- 历史分位数 `ts_rank(F,n)` 给基本面因子加入时间序列趋势，估值、盈利能力、收益质量、营运能力效果较好。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：动态权重必须保留等权基准；技术因子必须做多重检验和成本约束；历史分位数必须验证对原始因子的增量。

### Artifacts Created

- `notes/华泰因子合成技术因子历史分位数学习笔记.md`
- `factor_library/因子合成与历史分位数技术因子框架.md`
- `replication_reports/华泰因子合成技术因子历史分位数复盘.md`
- `strategy_lab/factor_combination_tools.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\factor_combination_tools.py`
- 小样本烟测覆盖滚动 IC 权重、ICIR 权重、复合因子、PCA、ts_rank 和 Alpha101 选定公式。

### Next

- 自动进入下一轮：转入投资者情绪与行为金融核心论文，补齐情绪、风险感知、前景理论和反转/价值异象的学术基础。

## 2026-05-23 第三十一轮：行为金融基础与反向投资解释

### Learned

- Bernoulli 风险效用说明风险价值依赖财富状态和边际效用，不等于金额期望。
- 前景理论提供参考点、损失厌恶、确定性效应、概率权重和隔离效应等行为机制。
- LSV 反向投资认为价值策略有效可能来自投资者过度外推 glamour 股票增长，而不是单纯风险补偿。
- 价值策略研究必须比较过去增长、隐含预期、未来兑现和坏状态表现。
- 系统反馈风险来自参与者对同一价格/流动性信号的同步调整，压力期多样化可能失效。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：行为解释不等于证据；前景理论参数需要校准；价值因子需与风险补偿竞争解释对比；扫描 PDF 通过渲染页视觉读取。

### Artifacts Created

- `notes/行为金融基础_风险效用前景理论反向投资学习笔记.md`
- `factor_library/行为金融因子假说库.md`
- `replication_reports/行为金融经典文献复盘.md`
- `strategy_lab/behavioral_factor_models.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\behavioral_factor_models.py`
- 小样本烟测覆盖 log utility、certainty equivalent、prospect score、LSV value/glamour 分组、外推误差汇总、坏状态表现和系统反馈风险。

### Next

- 自动进入下一轮：继续投资者情绪中文核心论文，重点学习情绪指数构建、媒体情绪、网络关注和 ETF/基金资金流的实证设计。

## 2026-05-23 第三十二轮：投资者情绪指数与市场波动

### Learned

- 在线社区支持倾向不同于朴素情绪，前者是看涨/看跌预期，后者是乐观/悲观心理。
- 支持倾向指标应同时使用方向和强度，且考虑正负信息非对称。
- 支持倾向一致性可能放大市场波动，适合作为羊群或拥挤风险变量。
- 中国情绪研究常把情绪变化放入 TGARCH/GARCH-M 的收益和方差方程，检验收益与波动。
- 情绪波动也可定义为主观贴现因子、风险规避和跨期替代弹性的波动。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：情绪概念拆分；预测必须用滞后；收益和波动分开检验；文本分类效果不等于金融有效性。

### Artifacts Created

- `notes/投资者情绪指数与市场波动学习笔记.md`
- `factor_library/投资者情绪指数因子框架.md`
- `replication_reports/投资者情绪指数与波动文献复盘.md`
- `strategy_lab/sentiment_index_models.py`

### Validation

- `python -X utf8 -m py_compile strategy_lab\sentiment_index_models.py`
- 小样本烟测覆盖牛熊支持倾向、一致性、情绪变化、PCA 情绪指数、分布滞后、预测回归、波动设计和羊群一致性。

### Next

- 自动进入下一轮：继续情绪与注意力方向，读取全球风险情绪、主观信念调整、资产估值、隐性杠杆约束及更多媒体/搜索/ETF 情绪论文。
## 2026-05-23 第三十三轮：国别风险情绪、主观信念、估值状态与隐性杠杆

### Learned

- 国别风险情绪通过风险溢价和跨境权益资金流传导，对被动、开放式、ETF 等风险厌恶更强的资金更敏感。
- 主观信念调整是情绪影响收益和波动的中介变量，可用基金仓位、情绪 PCA 和主观风险溢价共同刻画。
- 高估值阶段非理性情绪更容易放大波动，低估值阶段理性情绪也可能放大波动，理性资金不等于稳定器。
- 主动基金聚合持仓 beta 可作为隐性杠杆约束和资金流动性状态代理，收益检验必须条件化到低情绪或低流动性状态。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：情绪变量要拆分；含估值/波动代理的情绪指数需要正交化；跨境资金流必须控制反向因果；基金持仓要按披露滞后；隐性杠杆只做条件定价检验。

### Artifacts Created

- `notes/国别风险情绪主观信念隐性杠杆学习笔记.md`
- `factor_library/情绪资本流与隐性杠杆约束因子框架.md`
- `replication_reports/情绪资产定价与隐性杠杆文献复盘.md`
- `strategy_lab/sentiment_index_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\sentiment_index_models.py`
- 小样本烟测覆盖面板固定效应、交互项、聚合基金 beta、条件排序收益和主观信念更新。

### Next

- 自动进入下一轮：读取华泰人工智能系列中的遗传规划、AlphaNet 和深度学习因子挖掘，把机器学习选股因子的表达、约束、训练验证和过拟合控制固化为代码与研究规范。
## 2026-05-23 第三十四轮：华泰 AI 因子挖掘、遗传规划与 AlphaNet

### Learned

- 遗传规划通过公式树搜索扩大量价因子表达式空间，但必须用复杂度惩罚、验证集和传统单因子测试约束。
- RankIC、互信息和多头超额收益对应不同因子目标，互信息尤其适合发现非线性因子。
- 非线性因子要么交给机器学习合成模型，要么用三次方残差或多项式拟合转成可排序因子。
- AlphaNet 把量化时序算子嵌入神经网络，实现因子生成和因子合成端到端优化。
- AlphaNet-v2/v3 的重点改进是比率类特征、LSTM/GRU 时序层、4:1 时间验证集和多周期特征提取。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：AI 因子不能只看训练表现；非线性因子不能只看 Top-Bottom；神经网络必须多随机种子、滚动样本外和五因子中性化复验。

### Artifacts Created

- `notes/华泰AI因子挖掘_GP与AlphaNet学习笔记.md`
- `factor_library/AI因子挖掘与端到端AlphaNet框架.md`
- `replication_reports/华泰GP与AlphaNet因子挖掘复盘.md`
- `strategy_lab/ai_factor_mining.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\ai_factor_mining.py`
- 小样本烟测覆盖截面预处理、RankIC、互信息、多头超额收益、非线性变换、验证集收敛、AlphaNet 配置、滚动训练窗口和数据图片构造。

### Next

- 自动进入下一轮：读取华泰 BERT 舆情因子、研报情感因子，以及另类交易策略中的股指期货跨品种组合和波动收敛突变趋势跟随策略。
## 2026-05-23 第三十五轮：BERT 文本情绪因子与股指期货另类策略

### Learned

- 新闻舆情因子用正面新闻数减负面新闻数，并对过去 30 个自然日线性衰减。
- 研报情感因子用 BERT 正面概率减 0.5，并对过去 90 个自然日线性衰减；负面稀缺时可构造 `senti_adj`。
- 研报入库时间是关键可获得时间，不能用创建时间直接回填。
- 跨品种策略应关注相对收益，CCA 可从因子与资产收益中寻找最优相对组合。
- 收敛突变模型用低波动识别横盘、包络线突破识别趋势，并用 SAR 式离场管理头寸。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：文本因子覆盖偏差、研报时间戳未来函数、负面情绪稀缺、CCA 过拟合和突破策略只看收益的问题。

### Artifacts Created

- `notes/BERT文本情绪与股指期货另类策略学习笔记.md`
- `factor_library/文本情绪因子与股指期货策略框架.md`
- `replication_reports/BERT情绪因子与期货另类策略复盘.md`
- `strategy_lab/text_sentiment_factors.py`
- `strategy_lab/futures_strategy_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\text_sentiment_factors.py Introduction-to-Quantitative-Finance\strategy_lab\futures_strategy_models.py`
- 小样本烟测覆盖文本情绪因子、覆盖度、残差化、CCA、突破信号、SAR 和交易指标。

### Next

- 自动进入下一轮：继续读取另类交易策略、基金研究和多因子正交化资料，补齐日内波动极值趋势跟随、基金评价体系和因子正交化方法。
## 2026-05-23 第三十六轮：日内极值趋势跟随、基金研究体系与因子正交化

### Learned

- 日内极值趋势跟随用偏移后的当日高低极值触发顺势信号，配合 SAR 式离场。
- 基金研究应组织成标签、因子、专题三层，标签保证可比性，因子判断优劣，专题做深层解释。
- 基金因子测试要剔除近 2 年基金经理变更和历史不足 2 年的产品。
- 权益、固收+、纯债基金有效因子不同，不能混池评价。
- 回归正交化只处理线性暴露，分组法更适合单因子中性化复核，非线性暴露需要额外检查。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：趋势策略评价维度不足、基金样本不可比、经理变更、线性正交化误解和子样本正交化选择问题。

### Artifacts Created

- `notes/日内极值基金研究因子正交化学习笔记.md`
- `factor_library/基金因子评价与正交化框架.md`
- `replication_reports/基金评价正交化与日内极值策略复盘.md`
- `strategy_lab/fund_research_and_orthogonalization.py`
- `strategy_lab/futures_strategy_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\fund_research_and_orthogonalization.py Introduction-to-Quantitative-Finance\strategy_lab\futures_strategy_models.py`
- 小样本烟测覆盖基金样本过滤、标签内标准化、回归/分组正交化、ICIR、分组收益和日内极值信号。

### Next

- 自动进入下一轮：读取横截面/时间序列回归辨析、短周期价量多因子、动态反转因子和 Level2 行情因子，继续补强因子检验与高频价量因子体系。
## 2026-05-23 第三十七轮：回归辨析、短周期价量、动态反转与 Level2 因子

### Learned

- 时间序列回归和横截面回归的输入输出相反：前者给定因子收益估计资产暴露，后者给定资产暴露估计截面因子收益。
- Fama-French R2 与 Barra 横截面 R2 不可直接比较，因为一个解释同期收益，一个服务未来截面预测。
- 短周期价量 alpha 必须保持因子测试目标、组合约束和执行价格一致；高 gross IC 如果不能覆盖换手和成本，不能进入实盘库。
- 动态反转的窗口应来自市场共同波段，而不是个股自身任意窗口；短于 20 日的窗口更容易表现为动量而非反转。
- Level2 因子中成交占比类证据强于主动净买入比例类，但成交占比常携带规模、反转、换手暴露，必须中性化复核。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：回归类型误用、短周期因子目标错配、动态反转短窗口污染、Level2 直觉强但证据弱、成交成本与容量遗漏等问题。

### Artifacts Created

- `notes/横截面时序回归短周期价量动态反转Level2学习笔记.md`
- `factor_library/短周期价量动态反转与Level2因子框架.md`
- `replication_reports/因子回归辨析与高频价量因子复盘.md`
- `strategy_lab/short_horizon_factor_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\short_horizon_factor_models.py`
- 小样本烟测覆盖时间序列暴露、横截面因子收益、四类短周期价量因子、动态反转窗口、Level2 成交占比/净买入比例和交易成本目标函数。

### Next

- 自动进入下一轮：读取量价结合、因子大讲坛、交易行为波动和博彩型股票预期收益，继续扩展价量行为因子和行为定价因子库。
## 2026-05-23 第三十八轮：量价相关、交易行为波动与博彩型股票

### Learned

- 半月量价相关性因子用复权收盘价与日换手率的 Pearson 相关系数度量量价同向/背离，低相关组未来收益更高，高相关组未来收益更低。
- 量价相关性因子需要结合前期涨跌拆成放量下跌、缩量上涨、缩量下跌、放量上涨；主要逻辑来自存量资金博弈下放量上涨的持续性不足和放量下跌的抛压释放。
- 因子大讲坛确认了月度行情因子整体强于季度财务因子，并强调市场涨跌状态、市值中性和反转中性对因子稳定性的影响。
- 换手率变异系数是流动性二阶矩，20 日换手率标准差/均值越高，未来收益越低；它不能被平均换手率完全解释。
- 博彩型股票因子用过去一个月最大日涨幅或 Top-N 极端涨幅均值刻画彩票偏好，高极端涨幅股票后续收益偏低，且不能简单归因于反转。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：因子方向混乱、量价相关等同反转、换手波动等同平均换手、博彩型因子等同短期反转、行为解释缺少可检验变量等问题。

### Artifacts Created

- `notes/量价行为波动与博彩型因子学习笔记.md`
- `factor_library/量价相关换手波动与博彩型行为因子框架.md`
- `replication_reports/海通量价交易行为博彩型因子复盘.md`
- `strategy_lab/short_horizon_factor_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\short_horizon_factor_models.py`
- 小样本烟测覆盖量价相关性、量价形态分类、换手率变异系数、博彩型 Top-N 极端收益、双变量排序、市场状态分层和因子库清单。

### Next

- 自动进入下一轮：读取价格形态选股、高频收益分布特征、分析师一致预期和分析师覆盖度，继续扩展技术形态、高频分布和分析师行为因子库。
## 2026-05-23 第三十九轮：价格形态、高频偏度与分析师行为因子

### Learned

- 价格形态因子补充了收盘价以外的日内信息：开盘冲高、盘低回升、均价偏离。半个月窗口更稳，且二阶项有额外信息。
- 高频收益分布中，方差和峰度不够稳，偏度最值得保留；1 分钟高频偏度优于 5 分钟版本，统一方向为低偏度更好。
- 一致预期因子必须区分原始预测、估值衍生指标和环比变化；全市场筛选后较关键的是 `Con_PB_rel` 和 `Con_PE`。
- 分析师覆盖度不能直接使用报告篇数，必须残差化为 ATOT，剥离市值、换手和前期收益后才具有选股含义。
- ATOT 同时预测未来基本面改善和未来收益，且滞后 1-4 个月仍有效。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：只用收盘价构造技术因子、高频矩不做数据质量控制、一致预期覆盖偏差、原始覆盖度误用、分析师数据未来函数等问题。

### Artifacts Created

- `notes/价格形态高频偏度与分析师因子学习笔记.md`
- `factor_library/价格形态高频偏度与分析师预期因子框架.md`
- `replication_reports/价格形态高频与分析师行为因子复盘.md`
- `strategy_lab/short_horizon_factor_models.py`
- `strategy_lab/analyst_expectation_factors.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\short_horizon_factor_models.py Introduction-to-Quantitative-Finance\strategy_lab\analyst_expectation_factors.py`
- 小样本烟测覆盖价格形态因子、日内高频矩、滚动高频偏度、一致预期估值、环比变化、ATOT、覆盖率、滞后检验和逐步 Fama-MacBeth 筛选。

### Next

- 自动进入下一轮：读取历史财务信息、因子溢价估计、选股因子研究回顾和一致预期目标价，继续扩展基本面时序、因子溢价和分析师目标价因子。
## 2026-05-23 第四十轮：历史财务、因子溢价、高频波动分解与分位数回归

### Learned

- 历史财务信息可以用 8 项类 Piotroski 指标构造 `Factor_F`，但 A 股复现必须先处理公告可得日、规模和 PB 暴露。
- 因子溢价估计不应固定依赖 24 个月均值；基于近期横截面 `R2` 的自适应 EWMA 更适合风格稳定性变化。
- 因子溢价波动率调整的目标是提高 ICIR 和组合稳定性，不是追求最高 IC 均值。
- 高频已实现波动中，上涨/下跌分解比系统性/特质性分解更可用；上涨波动率占比过高通常是负向信号。
- 分位数回归适合建模厚尾和异方差，0.1 分位点可解释为下尾收益保护，但必须和 OLS 与简单因子组合对照。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：财务因子未来函数风险、因子溢价过度均值化、高频上涨波动方向误读、分位数回归被误当万能增强器、近似优化器缺少正式收敛诊断等问题。

### Artifacts Created

- `notes/历史财务因子溢价高频波动分解分位数回归学习笔记.md`
- `factor_library/基本面综合因子溢价调整与分位数回归框架.md`
- `replication_reports/历史财务因子溢价高频波动与分位数回归复盘.md`
- `strategy_lab/fundamental_factor_portfolios.py`
- `strategy_lab/factor_premium_quantile_models.py`
- `strategy_lab/short_horizon_factor_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\short_horizon_factor_models.py Introduction-to-Quantitative-Finance\strategy_lab\fundamental_factor_portfolios.py Introduction-to-Quantitative-Finance\strategy_lab\factor_premium_quantile_models.py`
- 小样本烟测覆盖基本面综合因子、持久性检验、高频上涨波动占比、自适应 EWMA、波动率调整溢价、分位数回归系数与滚动预测。

### Next

- 自动进入下一轮：因子降维、预期质量、因子择时模型改进与择时指标筛选。
## 2026-05-23 第四十一轮：因子降维、预期质量与因子择时

### Learned

- 底层因子降维应优先使用类别内正交 IC 最高或正交 IC 加权，IC 序列 PCA 只能作为对照组。
- 一致预期数据源需要先比较覆盖度、极值错误率和清洗后预测准确度；朝阳永续覆盖高但极值多，Wind 极值少但覆盖低。
- 超预期因子必须在财报披露后构造，且要用最新可得一致预期作为基准。
- 因子择时可转化为因子收益预测问题，指标库应覆盖宏观、债券、股票市场和因子历史表现。
- 套索、弹性网和衰减加权因子择时更像风险保护机制，全区间不一定战胜基准，但在风格波动年份有补偿价值。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：把 PCA 当默认降维、忽视供应商极值误差、超预期提前使用、因子择时只看收益不看保险成本、风格概率阈值过拟合等问题。

### Artifacts Created

- `notes/因子降维预期质量与因子择时学习笔记.md`
- `factor_library/因子降维预期质量与择时模型框架.md`
- `replication_reports/因子降维预期质量与因子择时复盘.md`
- `strategy_lab/factor_reduction_timing_models.py`
- `strategy_lab/analyst_expectation_factors.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\factor_reduction_timing_models.py Introduction-to-Quantitative-Finance\strategy_lab\analyst_expectation_factors.py`
- 小样本烟测覆盖类别 IC 权重、PCA 权重、类别降维、套索择时、风格概率、一致预期误差和超预期因子。

### Next

- 自动进入下一轮：风险控制中的因子择时、预期调整类因子、宏观选股与宏观不确定性。
## 2026-05-23 第四十二轮：动态风控、预期调整与宏观敏感性

### Learned

- 因子择时不仅能通过收益预测实现，也能通过动态调整风控敞口上下限实现。
- 因子敞口边界应同时考虑因子收益预测、平均正负收益风险和投资者风险厌恶度。
- 预期调整因子应使用时间序列标准化版本，且对预测类型 3/4 的填充数据单独处理。
- MacroBeta 只刻画股票对宏观变量的方向和程度，不是直接买入信号；宏观得分必须乘以预期宏观方向。
- EPU beta 在 A 股尤其是沪深 300 中有较强证据，但必须分市值和股票池验证。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：收益预测等同因子择时、原始预期差值直接入库、MacroBeta 直接排序、使用未来宏观实际值、EPU beta 全市场外推等问题。

### Artifacts Created

- `notes/动态风险控制预期调整与宏观敏感性学习笔记.md`
- `factor_library/动态风控预期调整与宏观敏感性框架.md`
- `replication_reports/动态风控预期调整与宏观敏感性复盘.md`
- `strategy_lab/macro_sensitivity_models.py`
- `strategy_lab/risk_model_tools.py`
- `strategy_lab/analyst_expectation_factors.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\macro_sensitivity_models.py Introduction-to-Quantitative-Finance\strategy_lab\risk_model_tools.py Introduction-to-Quantitative-Finance\strategy_lab\analyst_expectation_factors.py`
- 小样本烟测覆盖动态敞口边界、预期调整标准化、MacroBeta、宏观得分、MacroBeta 稳定性、EPU beta 和截面溢价。

### Next

- 自动进入下一轮：宏观指标选股、异质动量、盈利趋势与预期底层数据处理。
## 2026-05-23 第四十三轮：宏观T值、异质动量、盈利趋势与预期底层

### Learned

- MacroBeta 不能直接排序使用，必须同时估计宏观项 t-stat，并只对显著敏感股票应用宏观方向。
- PPI、信用利差、黄金、原油、市场波动率等变量有更强可用证据，但宏观方向必须使用当期可得预测，不能用未来实际值。
- 异质动量 IMom 应剥离市场、规模、价值等共同因子；原始过去收益在 A 股容易混入高换手和高波动风险。
- IMom 在“前期下跌、下月反弹”状态中容易失效，必须保存市场状态过滤器。
- GP 盈利趋势应优先用同一财政季度的滚动 OLS 斜率，避免季节性把环比噪声误当趋势。
- 一致预期 ROE、NP、NPG、G 在时间序列标准化前必须先选择底层财年，锁定财年法逻辑最强但覆盖率更敏感。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：MacroBeta 原始排序、raw momentum 替代 IMom、盈利同比增速替代趋势斜率、预期因子忽略底层财年、宏观变量未来函数等问题。

### Artifacts Created

- `notes/宏观T值异质动量盈利趋势预期底层学习笔记.md`
- `factor_library/宏观T值筛选异质动量与盈利趋势因子框架.md`
- `replication_reports/宏观敏感性异质动量盈利趋势预期底层复盘.md`
- `strategy_lab/macro_sensitivity_models.py`
- `strategy_lab/short_horizon_factor_models.py`
- `strategy_lab/fundamental_factor_portfolios.py`
- `strategy_lab/analyst_expectation_factors.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\macro_sensitivity_models.py Introduction-to-Quantitative-Finance\strategy_lab\short_horizon_factor_models.py Introduction-to-Quantitative-Finance\strategy_lab\fundamental_factor_portfolios.py Introduction-to-Quantitative-Finance\strategy_lab\analyst_expectation_factors.py`
- 小样本烟测覆盖 MacroBeta t-stat、宏观 t 值信号、IMom、市场状态、GP 趋势、锁定财年与平滑财年预期底层选择。

### Next

- 自动进入下一轮：行业因子、质量因子、A 股五因子模型和回归树因子择时。
## 2026-05-23 第四十四轮：行业质量、五因子与回归树择时

### Learned

- 行业内多因子模型必须重新检验因子有效性和方向，不能直接复制全市场模型。
- 医药行业案例中，逐步筛选法选择边际信息显著因子，复合 RankIC 约 11%，但窄行业模型会在因子方向反转期变脆弱。
- 质量因子是 multi-signal：盈利能力、增长、稳定性、投资、净发行、资本结构都可能进入，但方向有条件差异。
- A 股投资因子在大盘与小盘中方向相反；股份净发行为负向但持有期拉长会衰减；杠杆变化比杠杆水平更有信息。
- A 股五因子定价模型中，估值采用 PE 优于 PB，盈利采用 SUE 优于 ROE，后验概率最高模型为 `{Mkt, SMB, FPE, FSUE, FTurn}`。
- 回归树因子择时适合非线性情景划分，但单独方向择时不够稳健；与因子动量一致时才保留敞口的防御性择时更可取。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：全市场因子方向外推、有效单因子重复入模、质量因子 ROE 化、资产定价模型因子堆叠、回归树择时过度切换等问题。

### Artifacts Created

- `notes/行业质量五因子与回归树择时学习笔记.md`
- `factor_library/行业质量五因子与回归树择时框架.md`
- `replication_reports/行业质量五因子与回归树择时复盘.md`
- `strategy_lab/industry_quality_factor_models.py`
- `strategy_lab/five_factor_pricing_and_tree_timing.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\industry_quality_factor_models.py Introduction-to-Quantitative-Finance\strategy_lab\five_factor_pricing_and_tree_timing.py`
- 小样本烟测覆盖行业逐步筛选、行业复合分数、质量复合因子、2x3 因子收益、五因子回归归因、BIC 后验、滚动回归树和防御性择时。

### Next

- 自动进入下一轮：上市公司关系网络、资产增长稳定性与资本结构变化、价量波动幅度、被动产品规模扩张对 alpha 的影响。
## 2026-05-23 第四十五轮：关系网络、资产稳定、价量波幅与被动扩张

### Learned

- 股价相关性网络可提取度、中心性和邻居收益溢出；度和中心性正交后仍有约 0.02 左右 IC，短周期溢出偏反转。
- 主营业务收入网络能提供业务相似信息，但行业中性后容易衰减，本质上接近行业分类增强。
- A 股资产增长本身只有弱正相关且依赖 PMI、市场情绪等状态；资产增长波动率更稳健，流动资产增长波动率表现最好。
- 资本结构变化中，杠杆率增加和股东权益比率下降在部分成长行业更有效，但必须按行业和盈利状态验证。
- 价格振幅在控制涨跌幅、换手、波动、流动性和市值后为正向补充因子；换手波幅为负向且空头效应强。
- 被动产品快速扩张期往往对应 alpha 策略表现较弱，A 股中尤其压制反转、换手等反转型因子，但只能作为状态变量使用。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：网络因子未正交、短周期溢出方向误读、资产增长过度泛化、价量振幅长持有误用、被动扩张因果化、邻接矩阵只读视图错误。

### Artifacts Created

- `notes/关系网络资产稳定价量波幅与被动扩张学习笔记.md`
- `factor_library/关系网络资产稳定价量波幅与被动扩张框架.md`
- `replication_reports/关系网络资产稳定价量波幅与被动扩张复盘.md`
- `strategy_lab/network_balance_price_passive_factors.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\network_balance_price_passive_factors.py`
- 小样本烟测覆盖相关性网络、资产增长波动率、行业稳定性、资本结构变化、价格振幅、换手波幅、正交残差、被动扩张状态和因子权重调整。

### Next

- 自动进入下一轮：日内市场微观结构、大单精细化、ROE 因子基本面改进、风格特征重分类。
## 2026-05-23 第四十六轮：日内微观结构、大单、ROE预测与风格分类

### Learned

- 高频因子的最佳计算窗口取决于因子逻辑：知情交易类因子偏开盘后 30 分钟，过度反应类因子偏剔除开盘后 30 分钟。
- 日内成交和大单成交呈 U 型，分钟波动和买卖价差呈 L 型；开盘后与收盘前虽然都放量，但信息结构不同。
- 大单阈值应基于多日单成交金额对数分布，`mean + N*std` 中 N 在 0-2 通常较稳健，绝对金额阈值可作为保险。
- 大单净买入占比和大单净买入强度强于单纯买入占比，开盘后版本在沪深 300 内仍有较强选股力。
- 最新披露 ROE 的选股能力来自业绩动量，但预测当期真实 ROE 需要加入一致预期 ROE 和 ROE 波动率置信权重。
- 基于市值、估值、盈利、关注度的风格聚类可用于类别中性化和风格动量溢出，但会改变组合的小市值暴露。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：高频因子默认全天窗口、大单阈值绝对化、一致预期 ROE 直接入库、ROE 高波动简单剔除、风格分类等同线性正交等问题。

### Artifacts Created

- `notes/日内微观结构大单ROE预测与风格分类学习笔记.md`
- `factor_library/日内大单ROE预测与风格分类框架.md`
- `replication_reports/日内大单ROE预测与风格分类复盘.md`
- `strategy_lab/intraday_bigorder_roe_style_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\intraday_bigorder_roe_style_models.py`
- 小样本烟测覆盖日内分段、高频聚合、大单因子、ROE 预测、ROE 波动率权重、风格聚类、类别中性化和风格动量。

### Next

- 自动进入下一轮：限价订单簿还原、深度学习高频因子挖掘、深度学习高频因子改进、注意力机制优化。
## 2026-05-23 第四十七轮：LOB、深度学习高频因子与注意力机制

### Learned

- 逐笔委托和逐笔成交可还原 LOB，提供快照外的排队、挂撤单、成交和盘口深度信息。
- LOB 可用于模拟撮合、TWAP 改进、限价单成交概率预测和高频因子拆解。
- 买入意愿可拆成净挂单、净撤单、净成交、被动净买入；线性复合提升有限，非线性模型更有价值。
- RNN+NN 使用 20 日高频序列预测未来 5 日收益，周均 IC 可达约 0.08；正交后仍有约 0.07，胜率提升。
- GRU 相比 LSTM 更简单，LSTM 未显著提升；特定股票池单独训练可能改善局部选股，但样本量下降。
- 10 分钟高频输入序列更长，GRU/LSTM 容易遗忘；注意力机制改善长序列信息保留，残差注意力更稳健。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：LOB 字段标准化不足、线性复合高频组件过度乐观、深度学习随机切分、模型输出未正交、注意力机制不检查换手和旧 regime 等问题。
- 第七十七篇 PDF 损坏，未作为已学习内容；已记录为待修复。

### Artifacts Created

- `notes/LOB还原深度学习高频因子与注意力机制学习笔记.md`
- `factor_library/LOB深度高频与注意力机制框架.md`
- `replication_reports/LOB深度高频与注意力机制复盘.md`
- `strategy_lab/lob_deep_highfreq_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\lob_deep_highfreq_models.py`
- 小样本烟测覆盖 LOB 事件还原、订单簿强弱、订单流强弱、成交概率目标、买入意愿分解、IC 加权复合、高频序列张量、滚动切分、正交层、注意力与残差注意力。

### Next

- 自动进入下一轮：股票久期、净利润指标改进、无形资产、高频与日度混合深度学习因子。
## 2026-05-23 第四十八轮：股票久期、净利润改进、无形资产与混频深度学习

### Learned

- 股票久期把估值、盈利、增长和折现率敏感度统一到现金流期限框架；PB/PE 只是特定现金流假设下的近似。
- 隐含久期长的股票通常高估值、低盈利、高增长、高波动，对折现率冲击更敏感；短久期因子在利率上行阶段更有利，但只能作为弱状态权重。
- 净利润相关因子需要同时处理可用性、可靠性和有效性：ROE 应由披露值、快报/预告、一致预期共同预测当期真实 ROE，SUE 应做行业有效性权重。
- 无形资产调整 PB 将研发和组织资本资本化并剔除商誉，可缓解传统 PB 对高研发、高品牌投入公司的系统性低估。
- 高频与日度混合深度学习通过双尺度序列融合捕捉非线性信息，强于线性 IC 加权，但换手、成本、容量和种子稳定性是入库前置条件。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：低 PB 等同短久期、利率信号过度择时化、ROE/SUE 回填未来财报、行业事后剔除、无形资产参数迷信、深度学习只看 IC 等问题。
- 本地第四份 PDF 文本层损坏，已用海通官网同名 PDF 补充并记录来源。

### Artifacts Created

- `notes/股票久期净利润无形资产与混频深度学习学习笔记.md`
- `factor_library/股票久期ROE_SUE无形资产混频因子框架.md`
- `replication_reports/股票久期净利润无形资产混频深度学习复盘.md`
- `strategy_lab/duration_intangible_profit_mixedfreq_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\duration_intangible_profit_mixedfreq_models.py`
- 小样本烟雾测试覆盖隐含股票久期、债券相似度、利率状态权重、ROE 预测、行业有效性权重、SUE、无形资产调整 PB、PB_INT-ROE 组合、混频序列构造和正交因子头。

### Next

- 自动进入下一轮：因子模型尾部相关性、多因子有效与失效，以及剩余海通早期因子研究。
## 2026-05-23 第四十九轮：尾部相关、极值因子与净换手率

### Learned

- 尾部相关性刻画极端市场中共同上涨/下跌概率，能补足 Beta 和普通相关在极端状态下的失效。
- 因子有效性不能只看 Pearson 或 RankIC；分组强弱指数和因子-收益尾部概率能发现两端有效、中间无序的因子。
- 净换手率用主动买入量减主动卖出量除以流通股本，是方向化换手率和短线动量因子。
- Lee-Ready 适合在报价法和逐笔法之间折中，但 A 股必须处理涨跌停修正、报价延迟和 Level2 数据缺口。
- 极值多因子筛选需要同时满足极端组收益差、极端组位置和胜率，且必须滚动估计。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：Beta 替代尾部风险、低线性相关误判因子无效、大样本显著性误读、净换手率忽略交易方向误差、极值阈值全样本优化、只看组合收益不看分位分布等问题。

### Artifacts Created

- `notes/尾部相关极值因子与净换手率学习笔记.md`
- `factor_library/尾部相关极值因子与净换手率框架.md`
- `replication_reports/尾部相关极值因子与净换手率复盘.md`
- `strategy_lab/tail_extreme_net_turnover_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\tail_extreme_net_turnover_models.py`
- 小样本烟雾测试覆盖经验尾部相关、Hill 尾指数、滚动尾部依赖、Beta 调整尾部风险、强弱指数、尾部概率、Lee-Ready、净换手率、极值筛选、极值得分和分位分布。

### Next

- 自动进入下一轮：融资融券、单因子多策略组合、上市公司薪酬、准另类数据与因子投资。
## 2026-05-23 第五十轮：融资薪酬准另类数据与技术指标择时

### Learned

- 融资因子的核心不是融资余额存量，而是融资增量相对成交额的买盘冲击；融资增速比融资余额更适合作为短中期行为因子。
- 融券余额受券源和制度约束明显，不能机械解释为稳定看空 alpha，更适合作为风险警告或融资信号折减项。
- 单因子多策略组合强调并联选择极端组股票，能够捕捉线性 IC 不强但极端分位有效的因子。
- 上市公司应付职工薪酬增长比已支付薪酬增长更具前瞻性，但必须做同季度同比和公告日对齐。
- 准另类数据可分为日内结构、公司关系网络和基金隐含信息三类；有效性来自信息扩散滞后、研究能力差异或市场状态识别。
- 经典技术指标适合通过类别投票形成指数/ETF 仓位，而不是无成本每日全仓多空切换。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录复习、反证和纠错。
- 已修正：融资余额存量误用、融券信号过度解释、极端组合忽略重叠和容量、薪酬字段季节性错配、准另类数据未剥离传统因子、技术择时忽略交易成本和换仓频率等问题。

### Artifacts Created

- `notes/融资薪酬准另类数据与技术指标择时学习笔记.md`
- `factor_library/行为准另类技术择时因子框架.md`
- `replication_reports/融资薪酬准另类数据技术指标复盘.md`
- `strategy_lab/behavior_altdata_technical_timing_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\behavior_altdata_technical_timing_models.py`
- 小样本烟测覆盖融资增速、融券警告、并联极端筛选、行业中性极端筛选、薪酬增长、关系公司动量、基金隐含信号、技术指标投票、仓位生成和择时收益。

### Next

- 自动进入阻塞资料修复：重新定位或修复 `2022-04-07_海通证券_选股因子系列研究（七十七）：改进深度学习高频因子的9个尝试.pdf`。
## 2026-05-23 第五十一轮：阻塞 PDF 修复与深度高频 9 个改进尝试补学

### Learned

- 本地第七十七篇 PDF 出现 EOF/XRef 损坏，已通过外部可读 PDF 补充学习。
- 深度高频因子的关键不是盲目增加模型复杂度，而是正交约束、特征压缩、横截面标准化、滚动切分、目标调整和样本外组合增量。
- 报告中的 9 个尝试形成了深度学习 alpha 入库检查清单：正交训练、特征压缩、标准化、输入频率、训练/验证比例、环境变量、预测目标、训练窗口和模型复杂度。
- 风险调整超额收益目标可能降低周均 IC，但提升多头超额收益，说明训练目标必须贴近最终组合目标。

### Review And Correction

- 已在 `logs/review_correction_log.md` 记录补学纠错。
- 已修正：将损坏 PDF 误标为已完成、将复杂模型等同于增量 alpha、忽略正交层、忽略高频率长序列遗忘、忽略预测目标与组合收益不一致等问题。

### Artifacts Created

- `notes/深度学习高频因子9个改进尝试补充学习笔记.md`
- `factor_library/深度学习高频因子改进检查清单.md`
- `replication_reports/深度学习高频因子9个改进尝试修复复盘.md`
- `strategy_lab/lob_deep_highfreq_models.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\lob_deep_highfreq_models.py`
- 补充函数覆盖风险调整超额收益目标和 9 项深度高频改进检查表。

### Next

- 当前阅读队列已无 `new` 或 `blocked` 项；进入总体验收和索引检查。
## 2026-05-23 总体验收：语料学习闭环完成

- 阅读队列 188 条全部为 `corrected`。
- 无 `new`、`blocked`、缺 artifact 或未复习条目。
- 已补充总索引：`QUANT_RESEARCH_MASTER_INDEX.md`。
- 关键验证：Round50 与 Round51 代码均通过 `py_compile`，Round51 风险调整目标和 9 项改进检查表通过小样本烟测。
## 2026-05-23 第五十二轮：量化知识复习与 GitHub AI-Quant/Quant 项目学习

### Learned

- 成熟开源量化项目共同强调：数据、因子、模型、组合、执行、绩效和研究治理必须形成闭环。
- Qlib 提供端到端 AI quant workflow；Alphalens 提供标准因子诊断；VectorBT 提供大规模向量化初筛；Lean/Backtrader 提供执行层和事件驱动回测；FinRL/FinRL-Trading 提供 RL 和目标权重接口；FinGPT/FinRobot 提供 LLM/Agent 金融信息处理；RD-Agent 提供自动化研发闭环；QuantStats 提供绩效报告分层。
- 已有本地知识库在因子广度上充分，但需要新增面向数百因子的总控层。

### Review And Correction

- 已修正认识：数百因子模型不是因子平铺平均，而是候选库大、实用模型克制、家族分层、冗余控制、样本外验证和执行层约束。
- 已修正代码：Pandas 相关矩阵底层数组只读时不能直接 `fill_diagonal`，改为复制数组后回写。

### Artifacts Created

- `notes/量化知识复习与GitHub公开项目学习笔记.md`
- `factor_library/数百因子量化模型构建框架.md`
- `replication_reports/量化知识复习与GitHub项目复盘.md`
- `strategy_lab/multi_factor_research_framework.py`

### Validation

- `python -X utf8 -m py_compile Introduction-to-Quantitative-Finance\strategy_lab\multi_factor_research_framework.py`
- 小样本烟测覆盖因子注册、去极值、标准化、中性化、IC、换手、质量分、聚类、筛选、家族合成、权重和绩效。
## 2026-05-23 第五十三轮：可治理因子工厂工程化启动

### Learned

- 单点因子研究要升级为因子工厂，必须增加实验运行器、数据契约、注册表模板和实验账本。
- 因子工厂的核心产物不是单个回测收益，而是一组可追踪表：审计、IC、分组收益、换手、相关性、聚类、质量分、筛选结果、组合权重、组合收益、绩效和晋级决策。
- 自主运行必须有晋级规则：`promote_to_paper`、`revise_and_rerun`、`reject_until_data_fixed`。

### Artifacts Created

- `strategy_lab/factor_factory_runner.py`
- `strategy_lab/factor_factory_ledger.py`
- `configs/factor_factory_default.json`
- `data_catalog/factor_registry_template.csv`
- `data_catalog/factor_factory_data_contract.md`
- `notes/自主多因子工厂长期运行计划.md`
- `replication_reports/因子工厂第一轮工程化复盘.md`
- `outputs/factor_factory_demo/`
- `logs/factor_factory_experiment_ledger.csv`

### Validation

- `python -X utf8 -m py_compile strategy_lab/factor_factory_runner.py`
- `python -X utf8 -m py_compile strategy_lab/factor_factory_ledger.py`
- `factor_factory_runner.py --synthetic-demo` 通过，输出 36 个注册因子、6 个家族、完整结果表。
- `factor_factory_ledger.py` 通过，demo 记录为 `promote_to_paper`。

### Next

- 自动进入下一阶段：真实 A 股 panel 字段映射、现有因子模块注册表建设、成本和容量约束接入、walk-forward 重估。
## 2026-05-23 第五十四轮：A 股首批候选因子注册表 v0

### Learned

- 因子工厂需要先把候选因子变成可审计注册表，再谈模型训练。
- 首批注册表不追求覆盖所有想法，而是优先选择定义清晰、方向明确、可得日期可审计的候选。
- 低成本行情/财务因子应先跑通，Level2、LLM、基金、另类数据因子后续按数据可得性逐步接入。

### Artifacts Created

- `data_catalog/a_share_factor_registry_v0.csv`
- `strategy_lab/factor_registry_audit.py`
- `outputs/a_share_factor_registry_v0_audit/`
- `replication_reports/A股首批因子注册表v0复盘.md`

### Validation

- `python -X utf8 -m py_compile strategy_lab/factor_registry_audit.py`
- 注册表审计通过：68 个候选因子、21 个家族、0 个重复、0 个 fail、0 个 warn。

### Next

- 自动进入下一阶段：真实数据字段映射与首批低成本因子计算器。
## 2026-05-23 第五十五轮：低成本 A 股因子计算器与因子工厂联调

### Learned

- 首批因子计算器应从低成本、低歧义数据开始，先覆盖行情、成交额、市值和基础财务。
- 注册表 v0 的 68 个候选中，低成本计算器当前可覆盖 33 个原始因子字段，其余需要分析师、基金、Level2、文本、宏观或模型输出。
- 本地目录目前没有可直接用于 A 股股票因子的真实 point-in-time panel；真实数据接入成为下一阶段 blocker。

### Artifacts Created

- `strategy_lab/a_share_low_cost_factor_builder.py`
- `configs/factor_factory_smoke.json`
- `outputs/low_cost_factor_demo/`
- `outputs/low_cost_factor_factory_demo_rerun2/`
- `replication_reports/低成本A股因子计算器与工厂联调复盘.md`

### Validation

- 低成本合成 panel：25600 行，33 个原始因子字段，平均覆盖率约 94.47%。
- 接入因子工厂 runner：入选 29 个因子、11 个家族。
- 实验账本：`demo_003_low_cost_clean_warning_free` 标记为 `revise_and_rerun`，失败 gate 为 `win_rate`。
- 全量相关系数 warning 已通过 `_safe_corr` 修正。

### Next

- 等待或接入真实 A 股 point-in-time panel 后，运行第一版真实低成本多因子模型。
## 2026-05-23 第五十六轮：端到端可运行量化模型系统

### Learned

- 可用模型系统必须以 walk-forward 为主路径，不能以全样本 runner 为主路径。
- 系统入口需要同时负责 panel 校验、注册表审计、walk-forward、成本扣减、账本记录和摘要输出。
- Demo 只能证明工程链路完整，不能证明真实 alpha。

### Artifacts Created

- `strategy_lab/factor_factory_walk_forward.py`
- `strategy_lab/quant_model_system.py`
- `configs/factor_factory_walk_forward_demo.json`
- `notes/量化模型系统使用说明.md`
- `factor_library/可治理多因子模型系统框架.md`
- `replication_reports/量化模型系统端到端Demo复盘.md`
- `outputs/quant_model_system_demo/`

### Validation

- `quant_model_system.py demo` 成功运行。
- Panel 校验通过：320 个日期、80 只股票、33 个注册表可用因子列。
- 注册表审计通过：68 个候选因子、21 个家族、0 fail、0 warn。
- Walk-forward 成功：10 个 split 全部 OK，单边成本 20bps，输出 gross/net 绩效。
- 实验账本新增 `quant_model_system_demo`，决策为 `promote_to_paper`，仅代表合成数据 demo 可进入纸面跟踪状态。

### Next

- 自动进入下一阶段：真实数据适配器、交易约束、容量模型、paper trading 状态文件。

## 2026-05-23 第五十七轮：交易约束、纸面跟踪与真实数据 Panel 层

### Learned

- 可用的量化模型系统不能只输出目标权重，必须把不可交易、成交额容量、资金规模和纸面跟踪状态纳入主流程。
- `promote_to_paper` 只代表研究和工程闸门通过，必须显式阻止被解释为实盘授权。
- 真实 A 股数据通常不是天然 panel，行情、财务、行业和状态数据需要按可得日期构造 point-in-time panel。

### Artifacts Created

- `strategy_lab/paper_trading_monitor.py`
- `strategy_lab/a_share_panel_builder.py`
- `notes/量化模型系统数据执行纸面跟踪说明.md`
- `factor_library/交易约束纸面跟踪与真实数据Panel层.md`
- `replication_reports/交易约束纸面跟踪与Panel构建器复盘.md`
- `outputs/quant_model_system_demo_v2/paper_tracking/`
- `outputs/panel_builder_demo/`

### Validation

- walk-forward 执行约束 demo 通过，9 个 split 全部 OK，单边成本 20bps，账本记录为 `promote_to_paper`。
- `quant_model_system.py demo` 已自动生成纸面跟踪状态，`live_trading_allowed=false`。
- `a_share_panel_builder.py --synthetic-demo` 生成 25600 行、80 只资产、320 个日期的 point-in-time panel。
- 全部 `strategy_lab` 模块通过 `compileall`。

### Next

- 自动进入下一阶段：为系统补充稳定的 smoke test harness、真实数据字段映射模板、模型运行报告和纸面监控日报。

## 2026-05-23 第五十八轮：量化模型系统 Smoke Test 与执行层修复

### Learned

- 系统级 smoke test 应覆盖数据层、执行层和总入口，而不是只检查脚本是否能 import。
- 容量约束不能在限幅后再无条件归一化，否则会重新突破单资产成交额上限。
- 真实 CSV 输入常见日期类型不一致，执行层合并前必须标准化键类型。

### Artifacts Created

- `strategy_lab/run_quant_system_smoke_tests.py`
- `replication_reports/量化模型系统SmokeTest复盘.md`
- `outputs/quant_model_system_smoke_test/`

### Validation

- `python -X utf8 -m compileall -q Introduction-to-Quantitative-Finance/strategy_lab` 通过。
- `run_quant_system_smoke_tests.py` 三项测试全部通过。
- 测试覆盖交易可行性与容量、point-in-time panel 构建、一键系统 demo 输出和纸面状态。

### Next

- 自动进入下一阶段：构建真实数据字段映射模板、数据质量报告和生产运行清单。

## 2026-05-23 第五十九轮：真实数据质量报告与字段映射模板

### Learned

- 真实数据接入前必须先做质量闸门，不能让字段缺失、覆盖率断层或重复 `date-asset` 进入回测。
- 字段别名和 point-in-time 规则应显式模板化，不能靠脚本在运行时猜测。
- 日期层面的股票池数量、标签覆盖率和可交易比例是 IC 与组合权重稳定性的前置条件。

### Artifacts Created

- `strategy_lab/data_quality_report.py`
- `data_catalog/a_share_real_data_field_mapping_template.csv`
- `replication_reports/真实数据质量报告与字段映射复盘.md`
- `outputs/data_quality_synthetic_demo/`

### Validation

- 合成 panel 数据质量报告通过：25600 行、80 只资产、320 个日期、0 个重复键、标签覆盖率 93.75%、33 个可用注册因子。
- 数据质量报告已接入 `run_quant_system_smoke_tests.py`。
- 第二次完整 smoke test 四项全部通过。

### Next

- 自动进入下一阶段：生成模型运行日报和纸面跟踪日报模板，把研究输出变成可持续监控系统。

## 2026-05-23 第六十轮：模型运行报告器

### Learned

- 可用系统不能要求研究者手动拼接十几张 CSV 才能判断结果，必须有单一运行报告。
- 模型报告应把账本决策、panel 校验、walk-forward 绩效、split 状态、纸面状态和人工监控清单放在同一处。
- 报告层必须再次声明 `promote_to_paper` 不等于实盘授权。

### Artifacts Created

- `strategy_lab/model_run_report.py`
- `replication_reports/模型运行报告器复盘.md`
- `outputs/quant_model_system_smoke_test/MODEL_RUN_REPORT.md`

### Validation

- `quant_model_system.py` 已在 demo 和真实 walk-forward 路径中自动生成 `MODEL_RUN_REPORT.md`。
- 完整 smoke test 四项通过，并确认模型运行报告存在。

### Next

- 自动进入下一阶段：构建纸面跟踪日报、漂移监控和真实持仓对比接口。

## 2026-05-23 第六十一轮：纸面漂移监控

### Learned

- 纸面跟踪不能只保存目标权重，还必须比较当前持仓与目标权重，识别新开仓、清仓、交易权重和最大漂移。
- 当前持仓数据格式不稳定，监控层需要支持权重和市值两种常见输入。
- 漂移监控是模型从研究进入纸面阶段后的核心治理环节。

### Artifacts Created

- `paper_trading_monitor.py` 新增 `normalize_current_holdings`、`paper_drift_report`、`save_paper_drift_report`。
- `replication_reports/纸面漂移监控复盘.md`

### Validation

- 新增 `test_paper_drift_report`。
- 系统级 smoke test 五项全部通过。

### Next

- 自动进入下一阶段：构建真实数据运行 SOP、最小实证项目目录和端到端命令清单。

## 2026-05-23 第六十二轮：真实数据接入前调试

### Learned

- 真实数据接入的主要风险不在模型本身，而在编码、字段别名、布尔状态、可得日期、重复键和坏数据闸门。
- CSV 字符串 `"False"` 不能直接 `.astype(bool)`，否则不可交易证券会被错误放行。
- walk-forward 前必须先有独立 `preflight`，把坏数据挡在回测之前。

### Artifacts Created

- `strategy_lab/csv_io.py`
- `strategy_lab/real_data_adapter.py`
- `notes/真实数据接入前SOP.md`
- `replication_reports/真实数据接入前调试复盘.md`
- `outputs/quant_model_system_preflight_smoke/`

### Validation

- `compileall` 通过。
- 系统级 smoke test 扩展为 8 项，全部通过。
- 新增覆盖：字段别名映射、布尔字符串、未来可得日期拦截、preflight 输出。

### Next

- 真实 A 股数据接入时，先按 SOP 跑字段映射、panel 构建和 preflight；无 fail 后再进入 walk-forward。

## 2026-05-24 第六十三轮：A 股真实数据获取 Pilot

### Learned

- 数据获取必须分成“广度原型数据”和“高精度生产数据”两层，不能把 AkShare 免费网页源误判为完整 point-in-time 数据库。
- 当前环境可联网访问 AkShare，但部分 Eastmoney 接口在代理下失败，需要 backend fallback。
- 当前没有 Tushare/JQData SDK 和 API 凭证，Chrome 登录态不能直接作为量化数据管线凭证使用。

### Artifacts Created

- `strategy_lab/a_share_data_harvester.py`
- `configs/data_credentials.example.json`
- `data_catalog/a_share_data_acquisition_status.md`
- `replication_reports/A股真实数据获取Pilot复盘.md`
- `data_raw/akshare/stock_list/stock_info_a_code_name.csv`
- `data_raw/akshare/calendar/trade_calendar.csv`
- `data_raw/akshare/daily_qfq/`
- `data_raw/akshare/financial_indicator/`

### Validation

- AkShare 股票列表成功：5522 只当前 A 股。
- 交易日历成功：8797 行。
- 日线 pilot 成功：5/5，覆盖 2000 年以后。
- 财务摘要 pilot 成功：5/5。
- 自动 fallback 修复：日线从 `stock_zh_a_hist` 切换到 `stock_zh_a_daily`，财务从 `stock_financial_analysis_indicator` 切换到 `stock_financial_abstract_ths`。

### Next

- 如需全量 AkShare 宽数据，直接运行 `a_share_data_harvester.py` 去掉 `--max-symbols` 并保留 `--resume`。
- 如需高精度生产数据，补充 Tushare token 或 JQData SDK 凭证后接入 point-in-time 数据源。

## 2026-05-24 第六十四轮：A股核心指数数据层

### Learned

- 指数数据必须拆成日线、估值、成分、权重和manifest，不能只保存单一行情表。
- 中证指数日线包含基日或回溯 close-only 行，OHLC信号需要显式过滤 `is_full_ohlc_bar`。
- 当前成分与最新权重只能做当前暴露和样例研究，不能替代历史 point-in-time 成分权重。

### Artifacts Created

- `strategy_lab/a_share_index_data_harvester.py`
- `data_catalog/a_share_index_data_acquisition_status.md`
- `replication_reports/A股指数高质量数据获取复盘.md`
- `data_raw/index/akshare_csindex/`

### Validation

- 上证50、沪深300、中证500共19个数据项全部采集成功，错误数0。
- 日线行数分别为5437、5914、5194；历史PE/PB均成功落地。
- 质量摘要无重复键、无非正收盘价；权重合计接近100%。

### Next

- 扩展到中证1000、中证800、中证全指、中证红利、红利指数。
- 将指数日线和PE/PB接入指数择时与估值分位模型。

### Extension Update

- 同轮扩展采集完成：`000852` 中证1000、`000906` 中证800、`000985` 中证全指、`000922` 中证红利、`000015` 上证红利。
- 合并 manifest 共 43 个数据项，全部 `ok`。
- 8 个指数均已落地日线、最新估值、当前成分和最新权重；其中 5 个指数落地历史 PE/PB。

## 2026-05-24 第六十五轮：A股申万行业指数数据层

### Learned

- 行业板块数据必须先固定供应商口径；申万、东方财富、同花顺不能混用。
- 用户自然语言行业名称需要映射到标准行业代码，例如电子元器件映射为电子/元件，基本金属映射为工业金属。
- 行业分析快照是一日多指数表，质量检查需要按 `date + index_code` 判重。

### Artifacts Created

- `strategy_lab/a_share_industry_index_harvester.py`
- `data_catalog/a_share_industry_index_data_acquisition_status.md`
- `replication_reports/A股行业指数数据获取复盘.md`
- `data_raw/index/akshare_sw_industry/`

### Validation

- 申万一级、二级、三级行业分类分别为31、131、336行。
- 40个行业指数日线共187047行，全部无重复日期、无非正收盘价。
- 40个行业当前成分共5787条，权重合计接近100%。
- manifest 84行全部 `ok`。

### Next

- 将行业指数日线接入行业轮动、估值分位、趋势和相对强弱模型。
- 补齐历史行业分类和历史成分权重，避免行业成分聚合的未来函数。

## 2026-05-24 第六十六轮：复杂版指数级行业轮动与大小盘切换模型设计

### Learned

- 复杂模型不能只是因子堆叠，必须拆成状态识别、专家模型、元模型、组合优化和风险覆盖。
- 当前数据足以支持价格、趋势、波动、回撤、成交额和宽基估值 spread；不足以支持行业历史估值分位和历史成分聚合。
- 复杂系统必须有降级路径，从 V0 工程烟测到 V6 生产研究版逐级推进。

### Artifacts Created

- `notes/复杂版指数级行业轮动与大小盘切换模型设计.md`
- `factor_library/指数级行业轮动与大小盘切换复杂因子框架.md`

### Validation

- 模型设计区分了可立即回测模块和待补数据模块。
- 明确禁止用当前行业成分权重回填历史。
- 明确行业估值快照只能用于当前状态判断，不能做历史估值分位回测。

### Next

- 实现 V1：规则状态 + 多因子专家 + 层级风险预算。
- 再实现 V2/V3：估值 spread、风险覆盖、HMM 状态识别和 walk-forward ablation。

## 2026-05-24 第六十七轮：HIRSSM V2 模型优化

### Learned

- 优化复杂模型的重点不是继续增加因子数量，而是建立因子治理、专家集成、状态收缩、组合约束和过拟合检查。
- 状态识别容易变成过拟合开关，应限制为风险预算和专家权重控制器。
- ML 排序专家必须受权重上限、walk-forward、PBO 和 Deflated Sharpe 约束。

### Artifacts Created

- `notes/HIRSSM_V2优化版模型设计.md`
- `configs/hirssm_v2_default.json`
- `factor_library/HIRSSM_V2因子治理与入库规则.md`
- `replication_reports/HIRSSM_V2模型优化复盘.md`

### Validation

- V2 明确区分当前可回测模块和待补数据模块。
- 配置文件固化了状态先验、专家上限、组合约束、风控触发和验证要求。
- 因子治理文件固定了聚类、正交、估值、ML 和晋级规则。

### Next

- 实现 `strategy_lab/hirssm_v2_model.py` 的 V2.0：价格特征、规则状态、专家打分、层级权重和月频回测。

## 2026-05-24 第六十八轮：HIRSSM V2.0 工程实现与回测验证

### Learned

- HIRSSM V2.0 已从设计推进到可运行脚本，当前启用价格、成交额、风险、估值、状态、专家、组合和成本回测的闭环。
- 组合构建层比信号层更容易出现隐蔽错误：袖内权重覆盖、上限截断后不再分配，会让现金暴露偏离配置预算。
- 当前版本降低了基准最大回撤，但全样本收益仍弱于中证全指；这说明 V2.0 是研究原型，不是可授权实盘模型。

### Artifacts Created

- `strategy_lab/hirssm_v2_model.py`
- `outputs/hirssm_v2_0/HIRSSM_V2_MODEL_RUN_REPORT.md`
- `outputs/hirssm_v2_0/cost_sensitivity_summary.csv`
- `outputs/hirssm_v2_0/expert_rank_ic.csv`
- `outputs/hirssm_v2_0/expert_ablation_summary.csv`
- `outputs/hirssm_v2_0/latest_target_weights.csv`
- `replication_reports/HIRSSM_V2_0实现复盘.md`

### Validation

- `python -X utf8 -m py_compile strategy_lab/hirssm_v2_model.py` 通过。
- 端到端回测成功，生成 2072 行月频目标权重和 3 组成本情景。
- 10bps 情景：年化收益 5.74%，年化波动 18.73%，最大回撤 -55.31%，平均现金权重 28.66%。
- 最新信号日 2026-05-22，状态为 `range_bound`，最新目标含 `000922`、`000300`、`000016` 与三个行业指数，现金约 11.59%。

### Next

- 将 `range_reversal` 降级为观察专家，或接入滚动 RankIC 门控后再恢复默认权重。
- 增加 point-in-time 宏观利率、汇率和商品数据后启用宏观敏感性专家。
- 把完整特征面板写出改为可选，默认保留轻量治理结果。

## 2026-05-24 第六十九轮：HIRSSM V2.0 审计修复

### Learned

- 原回测从 2000 年开始计入净值，但第一笔真实可执行交易在 2002-03-01，导致策略和基准比较口径不干净。
- 原收益统计使用 `final_nav / first_nav - 1`，会漏掉第一天收益和首笔成本；年度收益也存在同类问题。
- `range_reversal` 专家在 RankIC 和消融中均失败，应降级为观察专家，不能继续默认启用。

### Artifacts Created

- 修复脚本：`strategy_lab/hirssm_v2_model.py`
- 修复配置：`configs/hirssm_v2_default.json`
- 审计报告：`replication_reports/HIRSSM_V2_0审计修复报告.md`
- 修复后输出：`outputs/hirssm_v2_0/HIRSSM_V2_MODEL_RUN_REPORT.md`

### Validation

- `python -X utf8 -m py_compile strategy_lab/hirssm_v2_model.py` 通过。
- `configs/hirssm_v2_default.json` JSON 校验通过。
- 修复后回测起点为 2002-03-01，目标权重 1948 行。
- 10bps 情景：年化收益 7.26%，年化波动 19.72%，最大回撤 -54.81%，平均现金权重 23.08%。

### Next

- 对 `trend_continuation` 和 `liquidity_overlay` 做 walk-forward 门控，而不是按全样本消融直接禁用。
- 给防御 sleeve 增加现金替代门槛，避免负最终 alpha 的防御资产被强制配置。

## 2026-05-24 第七十轮：HIRSSM V2.0 专家剪枝评估

### Learned

- `range_reversal` 同时在 RankIC 和组合消融中失败，继续默认启用会污染模型。
- style 层趋势/相对强弱信号在当前样本中方向偏负，删除 `style_trend_continuation` 后收益、Sharpe 和回撤同时改善。
- `liquidity_overlay` 证据混合：行业层 RankIC 为正，style 层 RankIC 为负；全局删除有收益提升，但暂不应直接视为生产结论。

### Artifacts Created

- 剪枝脚本：`strategy_lab/hirssm_v2_expert_pruning.py`
- 细粒度禁用逻辑：`strategy_lab/hirssm_v2_model.py`
- 更新配置：`configs/hirssm_v2_default.json`
- 剪枝报告：`replication_reports/HIRSSM_V2_0专家剪枝评估报告.md`
- 剪枝实验输出：`outputs/hirssm_v2_pruning_full/`、`outputs/hirssm_v2_pruning_granular/`

### Validation

- `python -X utf8 -m py_compile strategy_lab/hirssm_v2_model.py strategy_lab/hirssm_v2_expert_pruning.py` 通过。
- JSON 配置校验通过。
- 全局组合剪枝测试 32 个变体，细粒度剪枝测试 130 个变体。
- 正式默认模型现禁用 `range_reversal` 和 `style_trend_continuation`。
- 10bps 情景：年化收益 8.67%，Sharpe 0.455，最大回撤 -54.55%。

### Next

- 将全局 `liquidity_overlay` 拆成 style/industry 两个门控，并做 walk-forward 检验。
- 新增专家启停的滚动 RankIC 规则，避免靠全样本剪枝。

## 2026-05-24 第七十一轮：HIRSSM V2.1/V2.2/V2.3/V2.4 治理化迭代

### Learned

- V2.1 的状态门控能减少全样本剪枝依赖，但单纯用离散启停仍容易让专家在样本外失稳。
- V2.2 的连续收缩比硬门控更稳，10bps 年化 7.74%、Sharpe 0.397、最大回撤 -54.54%，但 PBO 仍未过。
- V2.3 的嵌套选择小幅改善到 10bps 年化 7.89%、Sharpe 0.405、最大回撤 -54.54%，但 PBO 和 DSR 仍不足以晋级默认模型。
- V2.4 将嵌套选择压缩为 3 个稳定参数族，使用 10/20/30bps 多成本目标和切换惩罚；10bps 年化 7.73%、Sharpe 0.397、最大回撤 -54.54%，相对同期间 V2.0 基线小幅改善，但 PBO=0.353 仍未过阈值。

### Artifacts Created

- `strategy_lab/hirssm_v2_1_walk_forward.py`
- `strategy_lab/hirssm_v2_2_walk_forward.py`
- `strategy_lab/hirssm_v2_3_nested_walk_forward.py`
- `strategy_lab/hirssm_v2_4_stable_nested_selection.py`
- `outputs/hirssm_v2_1_walk_forward/`
- `outputs/hirssm_v2_2_walk_forward/`
- `outputs/hirssm_v2_3_nested_walk_forward/`
- `outputs/hirssm_v2_4_stable_nested_selection/`
- `replication_reports/HIRSSM_V2_4稳定嵌套选择与自检报告.md`

### Validation

- `python -m py_compile strategy_lab/hirssm_v2_model.py strategy_lab/hirssm_v2_4_stable_nested_selection.py` 通过。
- `configs/hirssm_v2_default.json` 中 `expert_state_stable_selection` 可解析，稳定参数族数量为 3。
- V2.4 smoke test 通过：目标权重非空、无负权重、权重和正常、无缺失资产和日期。
- V2.4 自检未全通过，失败项为 `pbo_below_0_20=False`，因此不晋级默认生产模型。

### Next

- V2.5 不继续扩大参数网格，优先实现更可解释的组合与风控机制：drawdown brake、cash substitution、状态条件化目标波动和拥挤度降权。
- 每完成一个 V2.x 版本必须保留自检结果、治理报告和是否晋级的明确结论。

## 2026-05-24 第七十二轮：HIRSSM V2.5 组合风控覆盖

### Learned

- 组合风控层可以显著降低 HIRSSM 的最大回撤，但如果回撤刹车设计不当，会变成永久高现金。
- 第一版 V2.5 使用成立以来高水位回撤作为动态刹车，自检失败：10bps 年化降至 4.50%，平均现金升至 45.25%。
- 修正为 252 交易日滚动回撤后，V2.5 通过自检：10bps 年化 5.95%，Sharpe 0.405，最大回撤 -37.35%，平均现金 31.09%。
- V2.5 牺牲了约 1.78pct 年化收益，换来约 17.18pct 最大回撤改善；它适合作为风险控制版候选，而不是直接证明底层 alpha 已生产级稳健。

### Artifacts Created

- `strategy_lab/hirssm_v2_5_portfolio_risk_overlay.py`
- `outputs/hirssm_v2_5_portfolio_risk_overlay/`
- `replication_reports/HIRSSM_V2_5组合风控覆盖与自检报告.md`

### Validation

- `python -m py_compile strategy_lab/hirssm_v2_5_portfolio_risk_overlay.py` 通过。
- `configs/hirssm_v2_default.json` 中 `portfolio_risk_overlay_v2_5` 可解析。
- V2.5 smoke test 通过：目标权重非空、无负权重、权重和正常、无缺失资产和日期。
- V2.5 self-check 全部通过。

### Next

- V2.6 做分情景归因，拆解 V2.5 收益损失来自市场状态、资产 sleeve、波动缩放、市场刹车还是组合刹车。
- 不继续靠调宽阈值追收益，优先把全组合缩放改造成局部、可解释、可审计的风险来源降权。

## 2026-05-25 第七十三轮：HIRSSM V2.6-V2.9 连续迭代

### Learned

- V2.6 将 V2.5 的全组合风控改为局部 sleeve 风控，10bps 年化 7.08%、Sharpe 0.421、最大回撤 -38.83%，自检通过。
- V2.7 增加规则化再入场，10bps 年化 7.41%、Sharpe 0.422、最大回撤 -43.58%，在收益、Sharpe、回撤和现金之间最均衡。
- V2.8 的 style 核心保护和 industry 局部降权没有优于 V2.7，10bps 年化 7.25%、Sharpe 0.417、最大回撤 -45.64%。
- V2.9 固定混合 V2.4 与 V2.8，10bps 年化 7.44%、Sharpe 0.413、最大回撤 -48.25%，收益保留更好但防守弱于 V2.7。
- V2.9 初版暴露 `asset` 字段丢失问题，导致非现金权重无法交易，平均现金异常升至 93.34%；已作为 V2.9.1 小修复并重跑通过。

### Artifacts Created

- `strategy_lab/hirssm_v2_6_to_v2_9_risk_iteration.py`
- `outputs/hirssm_v2_6/`
- `outputs/hirssm_v2_7/`
- `outputs/hirssm_v2_8/`
- `outputs/hirssm_v2_9/`
- `outputs/hirssm_v2_6_to_v2_9_iteration/`
- `replication_reports/HIRSSM_V2_6到V2_9连续迭代与自检报告.md`

### Validation

- `python -m py_compile strategy_lab/hirssm_v2_5_portfolio_risk_overlay.py strategy_lab/hirssm_v2_6_to_v2_9_risk_iteration.py` 通过。
- `configs/hirssm_v2_default.json` 中 V2.6-V2.9 配置可解析。
- V2.6/V2.7/V2.8/V2.9 smoke test 全部通过。
- V2.6/V2.7/V2.8/V2.9 self-check 全部通过。

### Next

- 暂定 V2.7 为当前“稳健可投候选”，V2.9 为“收益保留候选”。
- 下一轮不继续阈值微调，优先做 V2.7 的年度失败案例、成本容量检查和底层专家 PBO/DSR 复核。

## 2026-05-25 第七十四轮：HIRSSM V2.10/V2.10.1 专家治理软门控

### Learned

- V2.1 硬专家门控的主要价值是审计，不适合直接控制组合；它降低收益和 Sharpe，却没有明显改善最大回撤。
- V2.10 将硬门控改为连续软乘数并加入强负证据 kill-switch，但初版仍过度硬杀核心专家，10bps Sharpe 比 V2.7 低约 0.031，自检未通过。
- V2.10.1 将硬 kill 限定到行业趋势和行业流动性，趋势、估值、风险压缩、防御等核心专家只允许软降权。
- V2.10.1 10bps 年化 7.28%、Sharpe 0.404、最大回撤 -43.58%、平均现金 21.88%，自检通过；相对 V2.7 年化低 0.13 个百分点、Sharpe 低 0.019，回撤基本持平。
- 当前结论：V2.10.1 是“专家治理候选”，不是默认替代 V2.7；V2.7 仍是当前稳健可投候选。

### Artifacts Created

- `strategy_lab/hirssm_v2_10_soft_killswitch.py`
- `outputs/hirssm_v2_10_soft_killswitch/`
- `outputs/hirssm_v2_10_1_soft_killswitch/`
- `outputs/hirssm_iteration_dashboard/HIRSSM_ITERATION_DASHBOARD.html`

### Validation

- `python -m py_compile strategy_lab/hirssm_v2_10_soft_killswitch.py` 通过。
- V2.10.1 smoke test 通过：目标权重非空、无负权重、权重和正常。
- V2.10.1 self-check 全部通过。
- HTML 看板已加入 V2.10 和 V2.10.1，默认展示 V2.10.1。

### Next

- 不把 V2.10.1 直接晋级默认；下一步应做专家治理消融：仅软乘数、仅行业 kill、仅观察专家正门控，拆分每个治理组件的边际贡献。

## 2026-05-25 第七十五轮：HIRSSM V3.0/V3.1 大版本尝试

### Learned

- V3.0 将目标函数切换为相对中证全指的主动收益、回撤改善、波动降低和信息比率；候选包括 V2.4 stable、stable_balanced、stable_conservative、V2.7 风控袖套、V2.10.1 治理袖套。
- V3.0 选中 `v3_0_v2_7_risk_overlay`，说明在当前因子和组合结构下，V2.7 仍是最优 benchmark-relative 袖套。
- V3.0 10bps 年化 7.41%、中证全指年化 5.64%、年化超额 1.77%、最大回撤 -43.58%；工程检查通过，但未达到年化超额 3% 的投资门槛。
- V3.1 在 V3.0 袖套上测试 000985 核心仓 + 卫星增强仓，选中 `v3_1_defensive_core`，10bps 年化 6.40%、年化超额 0.77%、最大回撤 -58.30%。
- V3.1 结果差于 V3.0，说明简单核心-卫星并不能解决超额不足，反而引入了中证全指自身的大回撤和较低收益拖累。

### Artifacts Created

- `strategy_lab/hirssm_v3_0_v3_1_benchmark_core.py`
- `outputs/hirssm_v3_0_v3_1_benchmark_core/v3_0/`
- `outputs/hirssm_v3_0_v3_1_benchmark_core/v3_1/`
- `outputs/hirssm_iteration_dashboard/HIRSSM_ITERATION_DASHBOARD.html`

### Validation

- `python -m py_compile strategy_lab/hirssm_v3_0_v3_1_benchmark_core.py` 通过。
- V3.0/V3.1 smoke test 通过，报告文件齐全。
- V3.0/V3.1 投资门槛均未通过：10bps 年化超额未达到 3%。

### Next

- 下一个大版本不应继续做核心-卫星权重调度；应进入 V3.2：独立市场 beta 择时和进攻/防守风险预算层。
- V3.2 目标：在风险开启时提高权益 beta 和行业卫星预算，在风险关闭时降低 beta；不再用固定核心仓长期拖累收益。

## 2026-05-25 第七十六轮：HIRSSM V3.2 市场 beta 择时

### Learned

- V3.2 在 V3.0 选中的 `v3_0_v2_7_risk_overlay` 袖套上增加独立市场 beta 择时层，不新增横截面 alpha 专家，不允许杠杆和负现金。
- 择时证据由中证全指趋势、行业宽度、波动/回撤风险、深跌修复状态组成；输出 bucket 包括 `risk_on`、`recovery`、`neutral`、`cautious`、`risk_off`、`panic`。
- 选中变体为 `v3_2_recovery_attack`。10bps 年化 8.54%、Sharpe 0.449、最大回撤 -45.47%、平均现金 19.19%、相对中证全指年化超额 2.90%。
- 相比 V3.0/V2.7 的 10bps 年化 7.41%、Sharpe 0.422、最大回撤 -43.58%、年化超额 1.77%，V3.2 明显提高收益和超额，但回撤略差。
- V3.2.1 增加总权益暴露平滑后，最大回撤改善到 -43.05%，但 10bps 年化降至 8.08%、年化超额降至 2.45%，因此没有被选中。

### Artifacts Created

- `strategy_lab/hirssm_v3_2_market_beta_timing.py`
- `outputs/hirssm_v3_2_market_beta_timing/`
- `outputs/hirssm_iteration_dashboard/HIRSSM_ITERATION_DASHBOARD.html`

### Validation

- `python -m py_compile strategy_lab/hirssm_v3_2_market_beta_timing.py strategy_lab/hirssm_iteration_dashboard.py` 通过。
- V3.2 smoke test 通过：成本行齐全、无负权重、权重和不超过 1、报告文件齐全。
- V3.2 工程自检通过，但投资准入未全通过：10bps 年化超额为 2.9028%，低于 3% 门槛约 0.10 个百分点。
- HTML 看板已加入 V3.2，并默认展示 V2.0S、V2.7、V2.10.1、V3.0、V3.2。

### Next

- V3.2 是强候选但不能正式晋级默认生产版本；V2.7 仍保留为稳健基线。
- V3.3 不应继续只调 beta，应进入真实 alpha 增量：行业景气/盈利修正/估值修复的横截面 alpha 工厂，并用 walk-forward 和成本后超额验证。

## 2026-05-25 第七十七轮：HIRSSM V3.3-V3.5 alpha 工厂到稳健 ensemble

### Learned

- V3.3 新增横截面 alpha 工厂，因子包括相对动量、行业景气代理、风格估值修复、低风险质量、拥挤度缓解、反弹修复和流动性确认；使用过去 5 年 RankIC 为下一年生成因子 multiplier。
- V3.3 选中 `v3_3_value_quality_repair`，但 10bps 年化 7.76%，相对同区间中证全指年化超额 -0.26%，20bps 年化超额 -1.21%，自检未通过。
- V3.4 将 V3.3 alpha 袖套叠加 V3.2 式 beta 择时，选中 `v3_4_recovery_alpha_beta`；10bps 年化 8.38%，年化超额 0.36%，最大回撤 -50.41%，仍未达到 3% 年化超额门槛，且 20bps 超额为负。
- V3.5 将 V3.2 beta 择时袖套与 V3.4 alpha+beta 袖套做稳健混合，并加入单资产上限和调仓带；选中 `v3_5_beta_anchor`。
- V3.5 10bps 年化 8.76%、Sharpe 0.457、最大回撤 -44.86%、平均现金 18.15%、年化超额 3.12%，首次通过 3% 年化超额准入门槛；20bps 年化超额仍为 2.24%。
- 相比 V3.2，V3.5 提高年化约 0.22pct，提高 Sharpe 约 0.009，最大回撤改善约 0.61pct，换手从 0.764 降至 0.695。

### Artifacts Created

- `strategy_lab/hirssm_v3_3_to_v3_5_alpha_factory.py`
- `outputs/hirssm_v3_3_to_v3_5_alpha_factory/`
- `outputs/hirssm_iteration_dashboard/HIRSSM_ITERATION_DASHBOARD.html`

### Validation

- `python -m py_compile strategy_lab/hirssm_v3_3_to_v3_5_alpha_factory.py strategy_lab/hirssm_iteration_dashboard.py` 通过。
- V3.3/V3.4 工程 smoke 均通过，但投资准入失败，保留为失败对照。
- V3.5 smoke test 和 self-check 全部通过：成本行齐全、无负权重、权重和不超过 1、10bps 年化超额超过 3%、20bps 年化超额为正、回撤优于基准、现金不过高。
- HTML 看板已加入 V3.3/V3.4/V3.5，并默认展示 V2.0S、V2.7、V3.2、V3.5。

### Next

- 暂定 V3.5 为当前最强候选，V2.7 作为稳健基线，V3.2 作为 beta timing 强候选。
- 下一轮优先做 V3.5 的年度失败分解、相对 V3.2 的边际贡献归因，以及 V3.5 中 alpha sleeve 的真实增量检验，避免把 ensemble 的稳定性误判为 alpha 已经稳健成立。

## 2026-07-19：Long Hold V4 当前估值观察完成与血缘纠错

### Learned

- 178只当前股票候选已形成339,927行东方财富估值观察；聚宽覆盖全部候选并完成2,134次检查，BaoStock覆盖15只并完成1,545次检查。
- 33只股票存在PB尾部跨源差异。总体中心误差较低不能替代逐资产警告。
- 当前最终历史快照能够支持当下估值分位诊断，无法证明历史日期当时可见的数据值。
- 退市样本`000004`已有103个可比日期，但5只留出仅覆盖1只，不能据此通过退市股人口门槛。

### Review And Correction

- 复核发现估值函数先删除非正PE、再选最新值，导致14只当前亏损股票回退到陈旧正PE。已改为先选最新值；非正PE保留真实值、分位为空、退出行业排名并触发硬否决。
- 哈希复核发现估值采集器读取下游会重写的`research_snapshot.csv`，造成采集清单与当前快照循环失效。已改用稳定股票观察清单，并把指定资产文件纳入血缘。
- 两份本地JSON带UTF-8 BOM，严格解析失败。移除后9,898个JSON全部通过。
- 所有修复均重跑采集、交叉验证、股票快照、ETF快照和50万元当前决策，没有手工修改manifest哈希。

### Artifacts Created Or Updated

- `strategy_lab/long_hold_v4/stock_active_valuation_observation_collector.py`
- `strategy_lab/long_hold_v4/stock_active_valuation_observation_validator.py`
- `tests/test_stock_active_valuation_observation_collector.py`
- `reports/LONG_HOLD_V4_ACTIVE_VALUATION_COMPLETION_2026-07-19.md`
- `reports/PROJECT_AUDIT_FINAL_2026-07-18.md`
- `data_catalog/long_hold_v4_manual_data_interface_queue.md`
- `LONG_HOLD_V4.md`

### Validation

- `pytest` 211项通过，另有5个子测试通过；`unittest` 211项通过。
- 258个Python文件编译通过；`pip check`通过；9,898个JSON严格UTF-8解析通过。
- Agent治理检查通过：12个角色、95个运行清单；727项旧债务仍只位于35个allowlist路径。
- 六套关键manifest共8,017项哈希复核，0缺失、0不匹配。
- 当前账户仍为500,000 CNY现金、0持仓、0订单，状态`CASH_NO_ENTRY_SIGNAL`。

### Next

- 正式Gate仍为6/15通过。下一工作包应从9项阻断中选择可由现有本地证据闭合的一项，优先审查ETF跟踪指数变更或ETF历史规模流动性；无法满足PIT边界时继续保留阻断。

## 2026-07-19：ETF现金分红全市场PIT正式化

### Learned

- 现金分红事件可以独立取得PIT资格，不必等待ETF全收益价格整体闭合；事件表通过不代表价格或模型通过。
- 对全状态1,701只ETF逐只查询官方来源后，共认证930份文件。846份可直接解析为唯一完整事件，19份保留金额歧义，65份为规则调整或场外份额事件；独立验证后未留下未解释文件。
- 既有859条事件与4条新公告事件合并为863条正式记录，覆盖147只ETF。公告日是信息可得日，未来除息事件只能在除息日应用。
- 原129次启发式份额动作已经全部取得交易所或巨潮证据，当前登记表有152次受治理动作，推断动作归零。ETF全收益价格仍因当前最终快照和`511210`清算尾部保持observation。

### Review And Correction

- PIT Gate原证据契约只支持单一来源版本和固定输出，不能表达官方文件集合及验证后晋级。已加入来源版本集合哈希、输出角色认证和无循环晋级契约。
- 巨潮当前身份查询会漏掉退市ETF。新增基金全称、生命周期和证券代码精确过滤后，查询完成度从1,655/1,701修复到1,701/1,701。
- 解析器没有强行解决19份金额冲突；冲突保留到独立验证层，由既有正式事件证据仲裁。
- 直接运行ETF快照模块发现分红对齐常量定义晚于`main()`，导致13只ETF触发`NameError`。常量前移后，2026-07-19快照恢复为17只成功、13只数据缺口。
- 周末决策源清单按决策日重建，价格仍保留最近交易日，避免把证据日期和行情日期混为一谈。

### Artifacts Created Or Updated

- `strategy_lab/long_hold_v4/pit_etf_dividend_universe_coverage_collector.py`
- `strategy_lab/long_hold_v4/pit_etf_dividend_universe_validator.py`
- `strategy_lab/long_hold_v4/pit_etf_dividend_universe_events_promoter.py`
- `strategy_lab/long_hold_v4/pit_history_gate.py`
- `configs/long_hold_v4_pit_gate.json`
- `reports/LONG_HOLD_V4_ETF_DIVIDEND_PIT_2026-07-19.md`
- `reports/LONG_HOLD_V4_GATE_E2_PIT_READINESS_2026-07-19.md`
- `reports/PROJECT_AUDIT_FINAL_2026-07-19.md`
- `reports/LONG_HOLD_V4_RESUME_CHECKPOINT_2026-07-19.md`

### Validation

- 正式事件表863行、147只ETF；关键字段缺失、非正金额、日期链异常、可得日异常和重复事件均为0。
- `pytest` 270项通过，另有13个子测试通过；`unittest` 270项通过。
- Python编译通过；14,624个JSON严格解析，失败0。
- 五套关键清单共11,873个声明哈希复核，0缺失、0不匹配。
- Agent治理检查通过：12个角色、95份运行清单；727项旧债务仍只位于35个allowlist路径。
- Gate升至7/15；当前账户仍为500,000元现金、0持仓、0订单，状态`CASH_NO_ENTRY_SIGNAL`。

### Next

- 从ETF全收益价格资格继续：先闭合`511210`终止上市和清算事件，再补充覆盖退市基金的独立价格或净值来源。无法满足历史版本与可得日要求时，继续保留observation。

## 2026-07-19：ETF终止事件、独立源审计与官方事件候选

### Learned

- `511210`的112.79元/份清算分配是终止持仓的现金事件，不是可用于补齐K线的“最后价格”。正式事件可按2018-01-09公告日进入历史会计，但只覆盖这一已知对象。
- 全生命周期观察链由1,700只修复为1,701/1,701只，共1,466,663条真实行情、1,023条事件、0只隔离；`511210`只保留2013-08-16至2017-10-18的852条真实行情。
- 聚宽近期窗口对285,353条真实成交行情和5,978条净值实现100%交叉匹配，证明当前采集和字段语义一致；它不能证明2005年以来历史版本可得。
- 官方事件候选覆盖1,701只、1,466,663条价格和864条事件使用记录，确定性验证通过。独立源最早只到2025-05-06、退市覆盖6/123、来源版本深度0%，正式晋级仍阻断。

### Review And Correction

- 清算事件最初容易被误接成合成OHLC。实现改为独立现金账本，事件会终止持仓，但永不创建市场价格行。
- 聚宽有29条成交量和成交额均为0、OHLC沿用前值的无交易标记，Sina不输出这些行。审计明确排除无交易标记后再比较真实成交日，没有降低价格阈值。
- 正式事件与供应商事件仅发现`512390`一处金额精度差：0.18182对0.1818。候选采用官方精度，并将1,344条锚定复权价微小变化列为预期差异。
- 候选验证器的资产数和行数曾写死。已改为从认证候选清单动态读取，并把旧序列差异检查改为“原始价一致、复权差异资产与现金差异资产一致”的通用约束。
- 当前决策清单在采集器代码更新后出现1项哈希不匹配。按2026-07-19重跑当前决策后恢复为265项全匹配，决策仍为`CASH_NO_ENTRY_SIGNAL`。

### Artifacts Created Or Updated

- `strategy_lab/long_hold_v4/pit_etf_terminal_cash_event_collector.py`
- `strategy_lab/long_hold_v4/pit_etf_terminal_cash_event_validator.py`
- `strategy_lab/long_hold_v4/pit_etf_terminal_cash_event_promoter.py`
- `strategy_lab/long_hold_v4/pit_etf_joinquant_validation_collector.py`
- `strategy_lab/long_hold_v4/pit_etf_price_nav_validator.py`
- `strategy_lab/long_hold_v4/pit_etf_total_return_candidate_builder.py`
- `strategy_lab/long_hold_v4/pit_etf_total_return_candidate_validator.py`
- `reports/LONG_HOLD_V4_ETF_TERMINAL_EVENT_2026-07-19.md`
- `reports/LONG_HOLD_V4_ETF_PRICE_NAV_AUDIT_2026-07-19.md`
- `reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_CANDIDATE_2026-07-19.md`

### Validation

- 候选验证结论：`PASS_DETERMINISTIC_CANDIDATE_FULL_HISTORY_SOURCE_BLOCKED`；事件对齐失败0、收益恒等式失败0、重复主键0、合成清算行情0。
- `pytest` 284项通过，另有13个子测试通过；`unittest` 284项通过；Python编译通过。
- 14,640个JSON严格解析，失败0。
- 8套关键清单15,198项哈希复核，0缺失、0不匹配。
- Agent治理检查通过：12个角色、95份运行清单；727项旧债务仍只位于35个allowlist路径。
- Gate仍为7/15；当前账户仍为500,000元现金、0持仓、0订单，状态`CASH_NO_ENTRY_SIGNAL`。

### Next

- 获取覆盖2005年以来和全部123只退市ETF的独立价格/净值历史，并建立可验证的历史来源版本深度；并行盘点其余122只退市ETF终止事件。未通过前不生成正式全收益表，不运行Long Hold V4 walk-forward。

## 2026-07-19：ETF全生命周期当前最终来源闭合与PIT边界复核

### Learned

- 腾讯未复权ETF价格完成1,701/1,701只、1,469,617行，东方财富单位净值完成1,701/1,701只、1,483,802行；两者均覆盖2005-02-23至2026-07-17和全部123只退市ETF。
- 全市场价格与Sina相交1,466,119行，Sina行覆盖99.962773%，收盘价一跳内99.987996%；聚宽近期285,353条真实行情和5,978条净值仍保持逐值匹配。
- 当前最终全历史内容可以检查值、生命周期和退市覆盖，不能证明历史交易日当时的数据版本。腾讯和东方财富的真实`available_date`都是2026-07-19，价格和净值版本深度均为0%。
- 统一来源资格为`PASS_CURRENT_FINAL_PRICE_NAV_CONTENT_PIT_BLOCKED`；官方事件候选继续为`PASS_DETERMINISTIC_CANDIDATE_FULL_HISTORY_SOURCE_BLOCKED`，正式晋级保持false。

### Review And Correction

- 腾讯续采曾触发WAF 501。沿用同一run ID恢复，复用581只已完成资产并清除25个临时错误，最终零错误完成；没有覆盖第一次失败证据。
- 东方财富两条记录缺单位净值。解析器保留原始响应和缺失计数，丢弃这2行但不填0、不借累计净值代替；累计净值缺失7条按原样保留。
- 总体匹配率掩盖了尾部。验证器新增重大价差表：8行、7只ETF；`510180`与`510880`在2008-01-02的腾讯原始响应严重失真，东方财富单位净值支持Sina，主价格未被腾讯值改写。
- `511230`只有63.90%日期覆盖。缺失的74个Sina交易日均为10至90份微量成交，报告不再把1,701只写成逐日全覆盖。
- 上游运行曾只认当前工作区源码。新增SHA-256内容寻址代码封存，采集器、验证器和声明依赖可从当前文件或封存副本认证。

### Artifacts Created Or Updated

- `strategy_lab/long_hold_v4/pit_etf_tencent_price_collector.py`
- `strategy_lab/long_hold_v4/pit_etf_tencent_price_validator.py`
- `strategy_lab/long_hold_v4/pit_etf_eastmoney_nav_collector.py`
- `strategy_lab/long_hold_v4/pit_etf_eastmoney_nav_validator.py`
- `strategy_lab/long_hold_v4/pit_etf_source_qualification.py`
- `strategy_lab/long_hold_v4/pit_source_code_archive.py`
- `strategy_lab/long_hold_v4/pit_etf_total_return_candidate_validator.py`
- `reports/LONG_HOLD_V4_ETF_PRICE_NAV_AUDIT_2026-07-19.md`
- `reports/LONG_HOLD_V4_ETF_TOTAL_RETURN_CANDIDATE_2026-07-19.md`

### Validation

- `pytest` 319项通过，另有13个子测试通过；`unittest` 284项通过；Python编译通过。
- 5,914个当前权威JSON严格解析，失败0；16份关键清单20,558项声明哈希复核，缺失0、不匹配0。
- Agent治理检查通过：12个角色、95份运行清单；727项旧债务仍只位于35个白名单路径。
- Gate仍为7/15；当前账户仍为500,000元现金、0持仓、0订单，状态`CASH_NO_ENTRY_SIGNAL`。

### Next

- 当前最终价格/净值内容不再列为缺口。下一工作包先逐只盘点其余122只退市ETF的终止事件；并按不同采集日积累腾讯和东方财富不可变快照。历史PIT仍需带真实发布版本的授权数据，不能用2026-07-19采集日回填。
