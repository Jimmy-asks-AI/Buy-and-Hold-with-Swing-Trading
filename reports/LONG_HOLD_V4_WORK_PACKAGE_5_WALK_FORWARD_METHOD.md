# 工作包 5：Purged/Embargoed Walk-forward 方法

当前状态：`BLOCKED_NOT_RUN`

证据用途：研究流程与失败路径审查

晋级状态：`promotion_allowed=false`

## 正式入口加固

工作包 5 后续审查补上了原实现不能满足正式评估的四个缺口：

1. PIT 使用清单不再自行声明可得日。每条历史使用记录必须解析到版本化源文件中的 `source_row_key + asset + available_date`；源行缺失、重复或字段不一致时 Gate 失败。
2. 候选表不再提交 `validation_score` 和 `validation_p_value`。正式运行器对所有预登记候选重新执行验证窗口，按20bps额外滑点后的主动收益计算显著性、Holm校正、PBO和Deflated Sharpe。
3. 绩效基准固定为中证全指全收益 `000985.CSI`。验证期和独立测试期使用两份物理分离的基准文件，价格指数不能替代。
4. 独立测试消费凭证固定为同一输出根目录下的 `holdout_consumption/independent-test.json`。更换 `run_id` 不能重复读取同一测试期；所有验证窗口先成功，才会登记消费并读取独立文件。

正式入口为 `strategy_lab.long_hold_v4.formal_walk_forward`，运行产物可用其 `verify` 子命令重新验证。验签会核对代码、配置、正式输入、窗口产物、消费凭证、人工复核要求和禁止实盘状态。

## 窗口定义

窗口按资产级交易日历的有效交易日计数，日期不能用自然日近似。正式运行先冻结日历与 target manifest，再生成带 SHA-256 的 `plan.json`。

| 区间 | 长度 | 用途 |
|---|---:|---|
| 训练窗口 | 756 个交易日 | 因子计算、模型拟合和参数候选训练 |
| purge | 20 个交易日 | 切断训练标签与随后验证区间的重叠 |
| 验证窗口 | 126 个交易日 | 模型选择、因子筛选和参数比较 |
| 步长 | 126 个交易日 | 滚动到下一个开发窗口 |
| embargo | 20 个交易日 | 隔离最后一个验证标签与独立测试 |
| 独立测试窗口 | 252 个交易日 | 参数冻结后的最终评估，只允许一次 |
| 标签期限 | 20 个交易日 | purge 和 embargo 都不得短于该期限 |

训练采用固定 756 日滚动窗口。每个开发窗口都要求：

`train_end + label_horizon < validation_start`

最终开发窗口还要求：

`validation_end + label_horizon < independent_test_start`

程序对训练、purge、验证、embargo 和独立测试的索引集合做交集检查。任何重叠或标签跨界都会阻断，不会缩短隔离区继续运行。

## 调参与独立测试

参数候选必须预先登记，最多 24 组，多重检验校正固定为 Holm。候选表只允许登记参数、训练分数和 `train+validation` split role；不得自报验证分数或显著性。正式运行器用验证窗口重新计算结果。字段名或使用记录中出现 `test`、`holdout` 或 `independent_test` 会直接失败。

独立测试由只写一次的消费账本控制。账本绑定：

- 首次消费的 walk-forward `run_id`；
- plan SHA-256；
- 独立测试数据清单 SHA-256；
- `FINAL_EVALUATION_ONLY` 用途。

账本文件名不随 `run_id` 改变。账本存在后，使用其他运行编号再次评估也会失败。独立测试结果不能回流到参数候选表，也不能修改因子、阈值、成本口径或资产池。

## PIT Gate 绑定

每个窗口开始前重新验证 PIT Gate manifest、seal 和全部绑定文件，至少包括：

- PIT Gate run_id 与 manifest SHA-256；
- target manifest SHA-256；
- PIT 使用清单 SHA-256，以及使用记录和版本化源行的一致性；
- 验证/独立行情、目标、基准、日历和候选登记表的 SHA-256；
- 全部数据文件和数据 manifest SHA-256；
- target 生成代码 commit 与文件 SHA-256；
- target 配置 SHA-256；
- Gate 代码和配置 SHA-256。

数据、清单、代码或配置在 Gate 后发生漂移，窗口不能运行。

## 交易与资产级日历

信号在收盘后形成，订单最早在该资产下一次可成交开盘执行。正式价格状态需要逐资产登记：

- `has_market_data`；
- `is_suspended`；
- `is_limit_up`；
- `is_limit_down`；
- `is_delisted`；
- `list_date` 与 `delist_date`；
- `available_date`。

当前实现对涨停和跌停都采取保守的当日不成交规则。停牌、缺失行情和退市状态同样不成交。未成交尝试写入订单审计表，持仓只有在 fill 发生后才改变；未成交目标权重不能产生收益。持仓遇到退市而没有可执行退出或正式终止现金事件时，窗口失败。

## 窗口产物

每个成功窗口保存并封印：

- 数据清单和 PIT Gate 绑定；
- 代码 commit、代码哈希和配置哈希；
- 冻结训练参数；
- 目标权重；
- 订单尝试和成交；
- 账户与 NAV；
- 成本假设；
- 风险暴露；
- 核心仓/T 仓归因；
- 5bps、10bps、20bps 成本情景；
- 相对中证全指全收益的策略、基准和主动收益；
- 候选验证收益、Holm校正、PBO与Deflated Sharpe；
- window manifest 与独立 seal。

窗口目录只写一次。相同 `run_id/window_id` 不能覆盖。

## 核心仓与 T 仓归因

归因使用实际成交和每日持仓市值，不使用未成交目标权重：

- 核心仓表现：`initial_cash + core_net_pnl`；
- 核心仓加 T：`initial_cash + core_net_pnl + t_net_pnl`；
- T 仓毛收益：T 仓市值加累计卖出额减累计买入额；
- T 仓交易成本：T 仓实际成交记录的佣金、税费和滑点；
- T 仓净增益：`t_gross_pnl - t_trading_cost`。

归因后的核心仓加 T 净值必须逐日与账户 NAV 对平。对平差异超过 `1e-7` 时窗口失败。

成本情景是在账户已记录成本之外，按双边实际成交额追加 5bps、10bps、20bps 滑点。情景不会改写原始成交和账户，只生成独立复核表。

## 专项偏差检查

每次正式运行都需要四项单独结论：

| 风险 | 通过条件 |
|---|---|
| 幸存者偏差 | 使用历史成分、全生命周期证券和退市资产；当前成分回填为硬失败 |
| 未来函数 | 每行 `available_date <= decision_date`；财务修订按新 revision 进入 |
| 重复调参 | 只用训练/验证；独立测试消费次数不超过一次 |
| 多重检验 | 候选族预登记；运行器重算验证结果；Holm、PBO和DSR在独立测试前通过 |

任一项失败，窗口或整体流程进入 `BLOCKED`。

## 当前失败窗口与数据限制

2026-07-23 的 PIT Gate 运行 `wp5-gap-audit-20260723-v1` 产生 74 个失败检查，正式回测许可为 false。实际交易日历、正式目标权重和独立测试数据都没有生成，因此：

| 窗口 | 状态 | 原因 |
|---|---|---|
| 全部训练/验证窗口 | `BLOCKED_NOT_RUN` | `PIT_GATE_BLOCKED` |
| 独立测试窗口 | `BLOCKED_NOT_RUN` | `PIT_GATE_BLOCKED` |

这里没有失败窗口绩效可披露。数据限制详见 `LONG_HOLD_V4_WORK_PACKAGE_5_PIT_GATE_2026-07-23.md` 和两份工作包 5 数据清单。

加固提交 `517a45d` 上又使用原冻结缺口清单执行了
`wp5-hardening-blocked-20260723-v1`。结果仍为
`BLOCKED_PIT_GATE`，共87项失败，manifest SHA-256为
`947dd0efa33043e6a29e50c89ea30f68f5092ff1ea48ff7fd9058fe72f901c61`。
新增失败来自源行PIT使用清单、8份正式输入、验证/独立全收益基准，
以及旧清单绑定的代码和配置已过期。该运行只验证新门禁会继续失败闭锁，
没有生成正式绩效。

## 晋级规则

代码不会自动晋级。即使 PIT Gate、全部窗口、独立测试、成本解释、失败窗口披露和人工签字都齐全，程序仍输出 `promotion_allowed=false` 与 `manual_promotion_action_required`，等待仓库外的明确治理动作。当前证据结论只能写为 `RESEARCH_ONLY`、`BLOCKED` 或 `NO_ACTION`。
