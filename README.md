# A股长线持有与中低频做T量化研究框架

这是一套面向A股低波动、稳定分红资产的量化研究系统。研究对象以银行、保险、电力及其他公用事业股票为主，同时覆盖红利、价值、行业类ETF。

系统先判断一项资产能否长期持有，再等待深度低估、大幅下跌和跌后企稳同时出现。核心仓建立后，才允许使用小比例中低频T仓降低持仓成本。数据不完整、时间戳不可靠或条件不足时，默认保持现金。

> 本仓库用于研究、回测治理和纸面交易，不连接券商，不构成投资建议。当前历史数据门禁尚未全部通过，仓库中的旧回测收益不得用于推断未来表现。

## 研究目标

- 筛选盈利与分红可持续、资产负债表可靠、长期增长相对稳定的股票和ETF。
- 识别自身历史估值与行业相对估值均处于极低位置的阶段。
- 等待大幅下跌后的波动收缩和价格企稳，避免机械抄底。
- 分三批建立核心仓，控制单标的、单行业和组合总风险。
- 核心仓稳定后，用有限T仓做中低频价差交易。
- 将未来函数、幸存者偏差、复权污染、数据修订和交易成本纳入统一治理。

## 决策流程

```mermaid
flowchart LR
    A[全状态股票与ETF主表] --> B[PIT数据与来源认证]
    B --> C[长期质量筛选]
    C --> D[价值陷阱否决]
    D --> E[深度低估判断]
    E --> F[大幅下跌与企稳确认]
    F --> G[三批核心仓]
    G --> H[中低频T仓]
    H --> I[账户、成本与回撤风控]
```

### 1. 长期质量

通用指标包括连续盈利、分红连续性、派息率、ROE稳定性、利润波动和现金流覆盖。银行、保险、公用事业使用不同的行业会计门槛，不用一套通用PE规则处理所有行业。

### 2. 价值陷阱否决

下列风险属于硬约束：银行不良资产恶化、保险偿付能力不足、公用事业杠杆和利息覆盖恶化、自由现金流长期无法覆盖分红。硬约束失败后，低PE和高股息不再构成买入理由。

### 3. 深度低估

估值判断同时参考：

- 自身历史PE、PB和股息率分位；
- 同行业横向估值；
- 股息率相对中国10年国债收益率的利差；
- 盈利质量与估值压缩是否一致。

### 4. 跌后企稳

入场条件强调“大幅下跌后的盘整”，主要观察三年回撤、近期创新低速度、20日波动收缩、均线斜率和价格路径效率。单独出现低PE或超跌，不会直接触发建仓。

### 5. 核心仓与T仓

- 核心仓按目标仓位的 `30% -> 60% -> 100%` 分批建立。
- 信号在收盘后形成，最早在下一交易日执行。
- T仓只有在核心仓达到完整目标的80%后才能启用。
- T仓不超过单标的完整仓位的20%，组合T仓不超过10%。
- 股票按T+1约束处理，不模拟当日回转。
- 核心仓卖出需要人工复核，模型不会自动清空长期仓位。

## 默认风险与费用参数

| 项目 | 默认值 |
|---|---:|
| 股票佣金 | 0.008%，无最低佣金 |
| ETF佣金 | 0.005%，无最低佣金 |
| 核心仓总上限 | 80% |
| 最低现金 | 10% |
| 单只股票上限 | 12% |
| 单只ETF上限 | 25% |
| 单行业上限 | 30% |
| T仓组合上限 | 10% |
| 最大标的数量 | 10只 |

费用、资金规模和仓位参数都可以在 [`configs/long_hold_v4.json`](configs/long_hold_v4.json) 中调整。公开仓库中的账户文件仅为纸面交易示例，不代表任何真实账户。

## 数据治理

金融回测最容易在数据时间上出错。本项目把以下规则作为硬门禁：

- 所有可用于历史决策的数据必须满足 `available_date <= trade_date`。
- 当前最终快照不能回填为历史时点已知数据。
- 财务报表按实际披露和修订时间进入因子计算。
- 指数成分、行业归属、证券状态、分红和公司行为按历史版本处理。
- 退市资产不能从历史股票池中消失。
- 前复权价格只有在复权因子的历史可得性得到证明后才能进入正式回测。
- 所有候选历史表、正式历史表和模型晋级状态分开保存。
- 工程测试通过只证明程序可运行，不证明策略有效。

不满足门禁的数据仍可进入观察层，但会固定为 `historical_backtest_allowed=false`。

## 当前研究状态

项目的权威检查点为 `2026-07-19`：

- 当前决策链返回 `CASH_NO_ENTRY_SIGNAL`，没有建仓意向。
- PIT历史数据门禁通过 `7/15`，其余8类数据仍阻断。
- ETF现金分红事件链已经完成；ETF全收益历史仍缺少可证明的历史版本深度。
- 当前没有可晋级的walk-forward持仓路径。
- 所有Long Hold V4历史绩效引用和模型晋级保持关闭。

详细证据见：

- [`LONG_HOLD_V4.md`](LONG_HOLD_V4.md)
- [`reports/PROJECT_AUDIT_FINAL_2026-07-19.md`](reports/PROJECT_AUDIT_FINAL_2026-07-19.md)
- [`reports/LONG_HOLD_V4_RESUME_CHECKPOINT_2026-07-19.md`](reports/LONG_HOLD_V4_RESUME_CHECKPOINT_2026-07-19.md)
- [`data_catalog/performance_evidence_registry.csv`](data_catalog/performance_evidence_registry.csv)

## 目录结构

```text
configs/                         模型、数据门禁和Agent契约
data_catalog/                    数据契约、字段定义和证据注册表
factor_library/                  因子卡片与研究定义
portfolio_lab/long_hold_v4/      账户和成交回执模板
reports/                         审查、验证和失败案例报告
strategy_lab/long_hold_v4/       Long Hold V4核心代码
strategy_lab/agents/             多Agent治理结构与任务契约
tests/                           单元测试、PIT测试和回归测试
```

`data_raw/`、`outputs/`、真实账户状态、API凭据和大型生成队列不会上传到GitHub。

## 安装

建议使用 Python 3.13。Windows PowerShell 示例：

```powershell
git clone https://github.com/Jimmy-asks-AI/Buy-and-Hold-with-Swing-Trading.git
Set-Location Buy-and-Hold-with-Swing-Trading

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-long-hold-v4.txt
```

## 本地配置

先建立本地凭据、成交回执文件，并显式初始化纸面账户：

```powershell
Copy-Item configs/data_credentials.example.json configs/data_credentials.json
Copy-Item portfolio_lab/long_hold_v4/pending_fills.example.csv portfolio_lab/long_hold_v4/pending_fills.csv
Copy-Item portfolio_lab/long_hold_v4/account_events.example.json portfolio_lab/long_hold_v4/pending_account_events.json
python -X utf8 -m strategy_lab.long_hold_v4.execution --initialize-account --initial-as-of 2026-07-22
```

在 `configs/data_credentials.json` 中填写自己有权使用的数据接口。该文件已被 `.gitignore` 排除。不要把Token、密码、交割单或真实持仓提交到公开仓库。

## 运行测试

核心CI命令：

```powershell
python -X utf8 -m unittest discover -s tests -p "test_long_hold_v4*.py" -v
python -m compileall strategy_lab/long_hold_v4 tests
```

使用pytest运行更广的本地回归：

```powershell
python -X utf8 -m pytest -q tests
```

## 运行研究链

完整研究链需要先采集本地数据。数据准备和逐项命令见 [`LONG_HOLD_V4.md`](LONG_HOLD_V4.md)。数据门禁通过后，可运行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.pit_history_gate --as-of 2026-07-19
python -X utf8 -m strategy_lab.long_hold_v4.cli --as-of 2026-07-19
```

主要输出位于 `outputs/long_hold_v4/`：

- `pit_gate/readiness.json`：历史数据是否允许进入回测；
- `runs/<run_id>/`：每次完整运行的不可变产物目录；
- `current`：仅指向最后一次完整运行的原子 JSON 指针；
- `runs/<run_id>/candidate_decisions.csv`：候选标的与阻断原因；
- `runs/<run_id>/target_weights.csv`：该次运行的目标权重快照；
- `runs/<run_id>/order_intents.csv`：纸面订单意向；
- `runs/<run_id>/run_manifest.json` 与 `run_manifest_seal.json`：产物清单及独立封印。

运行完整性验证、失败恢复和旧 `current/` 目录迁移说明见 [`data_catalog/long_hold_v4_run_artifact_integrity.md`](data_catalog/long_hold_v4_run_artifact_integrity.md)。

研究入口只生成意向，不修改账户，也不连接券商。成交回执与日终记账由 `execution.py` 和 `accounting.py` 独立处理。

订单使用版本化 `OrderEnvelope`，全部执行字段都参与 SHA-256，生命周期另存于本地订单状态账本。成交时会重新校验订单、账户版本、运行清单、配置和交易日历，并在模拟成交后复查全部组合硬约束。旧账户迁移、估值快照格式、手工批准和崩溃恢复说明见 [`data_catalog/long_hold_v4_order_execution_security.md`](data_catalog/long_hold_v4_order_execution_security.md)。

## 多Agent结构

项目将数据、因子、基本面、技术面、组合、风险、执行和审计拆成独立职责，并用任务简报、固定输入输出、禁止事项和验收标准约束协作。当前公开版本主要实现的是可审计治理结构；部分角色仍由确定性程序串联，不应描述为九个持续在线、完全独立上下文的智能体。

## 已知限制

- 历史行业版本、逐版本财务报表和全市场历史估值仍不完整。
- ETF基准变更、历史规模流动性、费率和跟踪误差链尚未全部关闭。
- 部分行情源只能取得当前最终历史，不能证明过去每个交易日看到的版本。
- 当前没有足够证据报告策略年化收益、Sharpe、最大回撤或做T增益。
- 实盘接入、订单路由、券商风控和税务处理不在当前范围内。

这些限制会让系统保持现金或阻断回测，不会通过降低数据标准来生成更好看的结果。

## 许可

代码采用 [MIT License](LICENSE)。行情、财务、指数、公告和研报等第三方数据仍受各自来源条款约束，本仓库不重新分发原始数据或付费资料。
