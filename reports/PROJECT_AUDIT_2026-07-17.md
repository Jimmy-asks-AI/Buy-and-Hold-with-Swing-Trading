# 项目完整审查报告

> 本文件保留修复前缺陷和反例取证。整改后的当前结论见`PROJECT_AUDIT_FINAL_2026-07-17.md`，不得继续引用本文件末尾的旧评级作为当前状态。

审查日期：2026-07-17  
审查范围：仓库治理、Long Hold Dividend V4、历史 HIRSSM、旧因子工厂、数据层、回测层、组合与执行、多 Agent 治理、测试与依赖。  
审查方式：静态阅读、全量语法编译、正式测试、当前数据重跑、输入哈希核验、最小反例复现。未修改策略代码。

## 整改状态更新

同日完成 Gate A 整改。下列原始审查发现保留，用于说明缺陷和复现证据；当前代码状态以本节及`LONG_HOLD_V4_GATE_A_REMEDIATION_2026-07-17.md`为准。

- P0-1 已修复：所有必需数值字段必须为有限数，非法布尔值也会阻断数据门禁。
- P0-2 已修复：核心权重按行业聚合限额，T仓同时受组合、行业和最低现金约束；订单规划器再次执行组合T仓和现金预算。
- P0-3 已修复：缺失、契约失败或陈旧的必需价格都会计入失败；陈旧快照会阻断系统及`data_steward`。
- P0-4 已修复：回测验证单标的、核心仓、T仓、行业、现金和标的数量约束；持仓期间缺行情立即失败；没有独立walk-forward证据时不能自我晋级。
- 回归测试由16项增加到30项并全部通过；真实178只股票快照重跑仍为`CASH_NO_ENTRY_SIGNAL`，零订单。
- 已新增78个月的当前成分条件化历史诊断和入场筛选，但它明确禁止回测晋级。P1仍未完成：无幸存者偏差历史PIT股票池、ETF真实数据、成交回报与账户更新、真实独立多Agent及版本治理。

## 一、结论

项目已经具备一套可运行的当前截面研究原型，但尚未达到可投、可回测晋级或可持续实盘运行的标准。

- **当前研究可用性：有限可用。** 可以对当前银行、保险、公用事业股票做数据门禁、耐久性评分和等待状态分类。
- **历史绩效证据：不可用。** V4 没有历史 PIT 股票池、历史目标权重生成器或完整 walk-forward 结果。
- **订单与做 T：不可用。** 当前只有研究意向生成，没有成交回报、账户更新、全局 T 仓约束和持仓生命周期。
- **多 Agent：治理说明可用，运行时不成立。** V4 的九个 Agent 是顺序规则和状态标签，不是隔离上下文、独立执行、结构化交接的真实 Agent 流程。
- **版本治理：阻断。** `strategy_lab/`、`tests/`、`configs/`、`reports/` 当前在 Git 中的已跟踪文件数均为 0；研究成果主要依赖本机未跟踪文件。

Gate A 已修复本报告四项 P0，但历史PIT、walk-forward和成交闭环仍未完成。系统仍不应生成可执行买卖建议，不应报告 V4 预期收益，也不应把输入契约通过解释为策略已晋级。

## 二、当前真实状态

截至 2026-07-17，重新运行唯一入口成功：

```text
python -X utf8 -m strategy_lab.long_hold_v4.cli --as-of 2026-07-17
status = CASH_NO_ENTRY_SIGNAL
```

当前数据与决策：

| 项目 | 结果 |
|---|---:|
| 当前股票快照 | 178 只 |
| 银行 / 保险 / 公用事业 | 42 / 5 / 131 |
| 数据门禁通过 | 79 |
| 数据门禁阻断 | 99 |
| 耐久性合格 | 33 |
| `KEEP_CASH` | 160 |
| `WAIT_DEEP_VALUE` | 11 |
| `WAIT_DEEP_DRAWDOWN` | 6 |
| `WAIT_STABILIZATION` | 1 |
| `BUILD_*` | 0 |
| 当前持仓 / 订单意向 | 0 / 0 |
| 账户现金 | 500,000 CNY |

当前分数靠前的合格标的仍全部处于等待状态，例如皖天然气、江阴银行、常熟银行、成都银行、中国太保和中国平安。这个结果只表示当前规则没有入场信号，不代表这些标的一定低风险或未来收益确定。

数据边界：

- 178 行全部标记为 `current_universe_only=true`、`historical_backtest_allowed=false`。
- 当前成分表抓取于 2026-05-24，距研究日 54 天；四个指数代理止于 2026-05-22，数据龄 56 天。
- 股票复权价格文件 178 个，当前均可解析；177 个止于 2026-07-16，1 个止于 2026-07-17。
- 99 行主要因自由现金流分红覆盖、分红增长、利润增长等字段缺失而被阻断。

## 三、P0 阻断问题

### P0-1 数值坏数据可以绕过价值陷阱硬门槛

证据位置：

- `strategy_lab/long_hold_v4/core.py:139-147`
- `strategy_lab/long_hold_v4/core.py:203-208`
- `strategy_lab/long_hold_v4/core.py:249-291`

`audit_snapshot()` 只把 `NaN` 视为缺失，不验证字段能否转换为有限数值，也不验证单位和合理区间。随后 `_number()` 把非法字符串转成 `NaN`，而 `NaN > 上限` 和 `NaN < 下限` 均为 False，硬门槛因此失败开放。

已复现：将银行 `npl_ratio` 设置为 `NOT_A_NUMBER` 后，结果仍为：

```text
data_gate=pass, hard_veto=false, durable_eligible=true, final_score=75.8628
```

影响：缺失或格式错误的不良率、偿付能力、资本充足率、利息覆盖等关键字段可能被当作“没有触发风险”，直接污染候选池。

必须修复：建立字段级 schema，要求所有硬门槛数值为 finite，并按比例、金额、年数和布尔类型验证合理区间；任何解析失败必须进入 `blocked`。

### P0-2 行业上限和组合 T 仓上限没有被真正执行

证据位置：

- `strategy_lab/long_hold_v4/core.py:694-727`
- `strategy_lab/long_hold_v4/core.py:730-757`
- `strategy_lab/long_hold_v4/pipeline.py:206-216`

行业剩余额度被逐标的复制，同一行业多只标的可同时使用同一份额度。反例中三只银行的完整目标权重合计为 36%，超过 30% 行业上限；第一批建仓因只乘 30% 暂时不报错，到第三批会直接抛出 `AssertionError: sector cap violated`。

`target_t_weight_cap` 也按每只标的分别取 10% 上限，没有约束全组合之和。八只股票反例产生 14.4% 的 T 仓买入意向，并把现金压到 5.6%，同时违反 10% T 仓上限和 10% 最低现金要求。

必须修复：使用带共享行业预算的 capped allocation；T 仓在订单规划前做组合级预算分配；订单生成后再次校验核心仓、T 仓、行业、单标的和现金约束。

### P0-3 数据 readiness 会在缺文件或陈旧快照时错误报通过

证据位置：

- `strategy_lab/long_hold_v4/pipeline.py:279-319`
- `strategy_lab/long_hold_v4/pipeline.py:409-448`
- `strategy_lab/long_hold_v4/pipeline.py:251-264`

价格文件缺失或价格契约失败时，代码不会向 `investable_price_freshness` 追加 False。只要其他文件新鲜，`all(existing_flags)` 仍为 True。

已复现：两只股票中删除一只价格文件后，readiness 仍输出：

```text
fresh_investable_prices=true, blocking_reasons=[]
```

同时，`snapshot_ready` 只判断文件存在，不判断快照审计是否通过。使用陈旧快照时，系统错误输出 `CASH_NO_DURABLE_CANDIDATE`，`data_steward=pass`，且没有 blocking reason，而不是 `CASH_DATA_BLOCKED`。

必须修复：readiness 必须统计预期资产、成功加载资产、失败资产和陈旧资产；数据 Agent 状态必须来自真实 data gate 汇总，不能来自“文件存在”布尔值。

### P0-4 回测的 `promotion_allowed` 含义错误，且缺失价格会冻结资产价值

证据位置：

- `strategy_lab/long_hold_v4/backtest.py:23-64`
- `strategy_lab/long_hold_v4/backtest.py:109-115`
- `strategy_lab/long_hold_v4/backtest.py:182-210`

当前 `promotion_allowed` 只表示价格口径属于 `qfq_adjusted/hfq_adjusted/total_return`，不检查历史股票池、PIT 财务、样本外验证、成本场景、单标的上限、行业上限、T 仓上限或现金上限。

已复现：单只股票目标权重 90%，配置上限 12%，回测仍返回 `promotion_allowed=true`。

缺失开收盘价被转换为 0% 收益。反例中持仓资产停止报价后，90 元持仓价值在后续日期永久冻结，而不是进入停牌、退市、强制退出或不可估值状态。这会高估含停牌、退市和数据断档样本的净值。

必须修复：删除或重命名该布尔值；晋级必须由独立验证报告决定。回测需引入交易日历、停牌/退市状态、最后可交易日、退市回收规则，并对未知缺价 fail closed。

## 四、P1 高优先级问题

### P1-1 V4 只有回测引擎，没有 V4 策略历史回测

仓库中除测试外没有代码调用 `run_weight_backtest()`。当前 178 行快照全部禁止历史回测，也没有按历史公告日生成财务快照、历史股票池、目标权重和 T 仓状态的流水线。

因此：

- 当前不能回答 V4 年化收益、Sharpe、最大回撤、胜率或预期收益。
- 当前不能证明三批建仓优于一次建仓。
- 当前不能证明 T 策略在账户成本后能降低持仓成本。
- 当前不能用 V3.10 指数轮动结果替代 V4 股票/ETF 长持策略结果。

### P1-2 ETF 支持停留在接口和配置层

V4 有 ETF 字段、评分函数和成本测试，但没有 ETF 数据构建器、ETF 当前快照、底层指数全收益、规模/费率/跟踪误差数据，也没有 ETF 质量门槛测试。当前快照 178 行全部为股票。

项目目标明确包含 ETF，因此 ETF 不能被视为已实现。

### P1-3 账户与成交闭环不存在

证据位置：`strategy_lab/long_hold_v4/pipeline.py:76-89,130-248,330-360`

- 账户只校验现金非负和 holdings 是列表，不校验负持仓、重复标的、资产类型、成本价、股份整数或核心/T 分账一致性。
- 建仓阶段依赖手工维护的 `core_fraction_of_full_target`，T 仓依赖手工维护的 `t_fraction_of_full_target` 和 `t_holding_days`。
- 没有成交回报导入、订单状态、部分成交、撤单、账户更新或交易日志。
- 已持有标的一旦离开当前股票池，pipeline 可能因找不到现价而中断。
- 配置中的组合回撤复核和降风险阈值没有执行代码。

当前系统只能生成研究意向，不能持续管理真实持仓。

### P1-4 当前决策不可完整审计，输出 schema 不稳定

`candidate_decisions.csv` 没有输出实际用于入场判断的 `latest_price_date`、当前 `drawdown_3y`、`stabilized`、`falling_knife`、`range_regime`、`zscore20`、`ma20` 和 `ma60`。用户无法从候选表复核为什么某只股票处于 `WAIT_DEEP_DRAWDOWN` 或 `WAIT_STABILIZATION`。

无订单时，`order_intents.csv` 只有 BOM 和空行，`pandas.read_csv()` 直接报 `EmptyDataError`。所有输出即使为空也应保留固定列。

当前 `run_manifest.json` 的 185 个输入哈希均可核验，但没有代码哈希、Git commit/dirty 状态、Python/依赖版本，也没有底层财务、分红、估值原始文件哈希，不能完成端到端复现。

### P1-5 文档已与运行状态漂移

- `LONG_HOLD_V4.md:52-54` 仍写“缺少 PIT 快照、数据门禁阻断”。
- `reports/LONG_HOLD_V4_REFACTOR.md:29-38` 仍写快照不存在、状态为 `CASH_DATA_BLOCKED`。

实际状态是快照存在、33 只耐久性合格、无入场信号。错误文档会让后续 Agent 和用户基于过期边界工作。

### P1-6 V4 九 Agent 与旧十二 Agent 是两套未统一的治理系统

V4 配置定义 9 个角色，旧 `strategy_lab/agents/README.md` 定义 12 个角色。旧 `agent_framework_check.py` 审查十二角色，V4 pipeline 只检查九角色名称和四个非空列表。

V4 的 `agent_decision_log.csv` 由 `_agent_log()` 一次性合成，没有独立 Agent 执行、上下文隔离、输入白名单、独立产物或逐级 veto 交接。更严重的是，只要存在任何订单，`t_execution_analyst` 就会被标记为 pass，即使订单全部是核心仓订单。

结论：当前是“模块化规则引擎附带 Agent 标签”，不是运行中的多 Agent 量化研究系统。

### P1-7 依赖和 CI 不能复现完整项目

`requirements-long-hold-v4.txt` 只有 NumPy 和 pandas，但当前数据构建器需要 AkShare，历史采集脚本还使用 Tushare、JoinQuant、SciPy 和 OpenAI SDK。CI 运行 CLI 时没有本地数据，实际只验证 fail-closed 空数据路径，不验证 178 只股票的数据构建和当前研究路径。

所有 V4 工作流、代码、测试、配置和报告目前也未被 Git 跟踪，因此新 CI 文件不会在远端触发。

## 五、历史系统审查

### 5.1 HIRSSM V3.10

V3.10 的 10bps 历史结果为：年化 8.76%、Sharpe 0.455、最大回撤 -54.56%、相对基准年化 +0.74%，但信息比率为 -0.037。其 manifest 已明确标记为治理基线，不是 alpha 晋级模型。

这些结果属于指数轮动控制基线，与 V4 的稳定分红股票/ETF 长持加 T 不是同一策略，不能用于证明 V4 表现。

### 5.2 旧因子工厂

证据位置：

- `strategy_lab/multi_factor_research_framework.py:511-545`
- `strategy_lab/a_share_low_cost_factor_builder.py:29-41`
- `strategy_lab/factor_factory_runner.py:143-162`

两个已复现的重大错误：

1. 权重先 clip 再归一化，归一化后重新突破上限。设置 `max_weight=5%` 的三资产反例最终最大权重为 33.33%。
2. 默认 20 日前瞻收益按每日样本生成，随后当成独立期收益计算组合绩效。若按日连续复利，会重复计算高度重叠的未来区间。

V4 文档已经把旧因子工厂隔离为不可引用绩效，这是正确边界；隔离标记不能替代修复和回归测试。

### 5.3 旧 crowding 结果

`strategy_lab/hirssm_v2_model.py:258` 对完整时间面板直接 `rank(pct=True)`，而不是在每个交易日横截面排名。V2.5 至 V3.8 依赖该值的结果存在跨时点分布信息混入，应继续保持不可晋级。

### 5.4 旧测试与 Agent 债务

- `run_quant_system_smoke_tests.py` 从仓库根目录运行时失败，因为 `ROOT = Path("Introduction-to-Quantitative-Finance")` 重复拼接路径。
- `agent_framework_check.py` 返回 pass，但原始错误数为 727，全部被 35 个旧路径的 debt allowlist 豁免。这个 pass 只代表没有未豁免的新错误，不代表历史 Agent 产物健康。

## 六、通过项

以下设计应保留：

1. V4 将长期可持有性、价值陷阱、估值、价格企稳、核心仓和 T 仓拆开，方向符合目标。
2. 财务年报使用公告日过滤，当前快照明确禁止历史回测，没有把当前成分伪装成 PIT 历史股票池。
3. 收盘信号在下一交易日开盘执行，正式回测不复利重叠标签。
4. 股票与 ETF 佣金按用户账户配置，无最低佣金；股票卖出印花税、股票双向过户费、ETF无印花税的结构正确。
5. 当前价格文件显式声明 `return_basis=qfq_adjusted`，价格指数代理明确禁止晋级。
6. 16 个 V4 单元测试全部通过；215 个 Python 文件全部可解析编译；V4 CLI 当前可运行。
7. 当前 run manifest 的 185 个已列输入哈希全部匹配，当前账户仍为 500,000 CNY 现金、零持仓、零订单。
8. 本地真实凭证文件已被 Git 忽略，未发现真实凭证进入已跟踪文件；但凭证仍以明文保存在本机。

## 七、测试结果

| 检查 | 结果 |
|---|---|
| `python -m unittest discover -s tests -p test_*.py -v` | 16/16 通过 |
| `python -m compileall -q strategy_lab tests scripts` | 通过 |
| V4 当前 CLI | 通过，`CASH_NO_ENTRY_SIGNAL` |
| 当前 manifest 输入哈希 | 185/185 匹配 |
| `git diff --check` | 通过，仅有 LF/CRLF 提示 |
| 旧量化系统 smoke test | 失败，根路径错误 |
| 旧 Agent framework check | 表面通过，727 条错误被债务白名单豁免 |
| ETF 数据与端到端测试 | 不存在 |
| V4 历史 walk-forward | 不存在 |

当前测试的主要盲区正是本报告反例触发的路径：非法数值、多标的行业预算、全局 T 仓、最低现金、部分价格缺失、陈旧快照、退市缺价、持仓离开股票池、账户更新、ETF 和历史 PIT。

## 八、仓库与工程规模

| 项目 | 数量 |
|---|---:|
| 非 Git 文件 | 12,694 |
| 本地体积 | 5.586 GB |
| Python 文件 | 215 |
| Python 行数 | 96,621 |
| HIRSSM 版本脚本 | 93 |
| 正式测试文件 | 1 |
| Git 已跟踪文件 | 503，主要为原始资料仓库 |
| Git 未跟踪文件 | 675 |
| Git 已跟踪 `strategy_lab/tests/configs/reports` | 0/0/0/0 |

`data_raw` 约 2.66 GB，`outputs` 约 2.52 GB，均已被忽略。这适合避免提交大文件，但当前没有外部对象存储、数据版本号或可恢复快照，研究复现依赖本机磁盘和上游接口仍可访问。

## 九、整改顺序

### Gate A：恢复正确性，完成前禁止订单意向

1. 为 snapshot、account、target、order 建立严格 schema 和 finite/range 校验。
2. 修复共享行业预算、组合 T 仓上限、最低现金及订单后约束复核。
3. 修复 readiness 汇总，任何预期文件缺失、契约失败或陈旧都必须显式阻断。
4. 删除误导性的 `promotion_allowed`，修复缺价、停牌和退市估值逻辑。
5. 把本报告六个反例全部加入自动化回归测试。
6. 输出固定空表 schema，并把真实时机特征写入候选表。

### Gate B：建立可验证的 V4 历史研究链

1. 构建历史证券主表、历史行业分类、历史 ST/停牌/退市和历史成分。
2. 按公告日及修订可得日构建财务、分红、估值和国债收益率 PIT 面板。
3. 生成逐日或逐月 V4 历史目标，执行三批建仓和核心/T 分账状态机。
4. 使用总收益或严格复权收益，加入 5/10/20bps、真实账户成本、涨跌停和容量。
5. 做 anchored walk-forward、purge/embargo、参数稳定性、行业留出、时期留出和失败案例。
6. 与不择时买入、单次买入、无 T、红利全收益指数和现金基准分别比较。

### Gate C：补齐 ETF 和实盘研究闭环

1. 接入 ETF 规模、成交额、费率、跟踪误差、分红和底层指数全收益。
2. 建立成交回报导入、部分成交、撤单、持仓成本、核心/T 子账和账户快照更新。
3. 已持有标的必须独立于当前候选股票池持续估值和风控。
4. 实现组合回撤复核、风险降档和人工确认工作流。

### Gate D：统一治理与可复现工程

1. 选择一套正式 Agent roster，并给 V4 建立真实任务 brief、独立输入、独立输出和 veto 交接。
2. 将研究代码、测试、配置、契约和报告纳入 Git，避免继续累积未跟踪版本文件。
3. 拆分 current、historical、legacy 三个命名空间；旧脚本默认不可执行晋级。
4. 建立完整依赖锁、最小脱敏数据 fixture、数据构建 CI 和端到端验收。
5. run manifest 增加代码/配置/原始数据哈希、Git 状态、环境和依赖版本。

## 十、最终评级

| 维度 | 评级 | 说明 |
|---|---|---|
| 目标匹配 | B | V4 方向已转向稳定分红、深度低估、分批建仓和做 T |
| 当前数据 | C+ | 当前截面可用，但 99/178 被阻断，股票池和代理存在陈旧项 |
| PIT 历史 | D | 当前股票池明确不可历史回测 |
| 因子与评分 | C | 经济逻辑可解释，但阈值未经样本外校准，坏数值可绕过硬门槛 |
| 组合风控 | D | 行业和全局 T 仓约束存在已复现错误 |
| 回测可信度 | D | V4 无历史回测，回测 promotion 和缺价处理不可靠 |
| 做 T 可用性 | D | 只有规则，没有历史证据和成交/账户闭环 |
| ETF 完整度 | F | 没有 ETF 数据和端到端路径 |
| 多 Agent 运行 | D | 契约和标签存在，独立 Agent 运行不存在 |
| 测试 | C- | 正常路径 16 项通过，高风险组合和数据异常路径缺失 |
| 可复现性 | D | 研究代码未纳入 Git，依赖和 provenance 不完整 |

**总评：C- 级研究原型，D 级可投系统。** 当前最正确的状态仍是保持现金和继续研究，但理由不仅是“没有入场信号”，还包括系统自身尚未通过可投验收。
