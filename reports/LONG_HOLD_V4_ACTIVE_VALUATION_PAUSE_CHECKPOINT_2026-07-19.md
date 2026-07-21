# Long Hold V4 在市股票估值观察暂停检查点

后续状态：本检查点已于2026-07-19恢复并完成。完成结果、纠错和验证证据见`reports/LONG_HOLD_V4_ACTIVE_VALUATION_COMPLETION_2026-07-19.md`；本文件保留暂停时的原始状态，不再作为当前进度口径。

## 暂停状态

- 暂停时间：2026-07-19 00:57:49 +08:00
- 数据截止日：2026-07-17
- 目标候选股：178 只
- 已完成：57 只
- 待采集：121 只
- 新观察缓存：52 只
- 复用旧验证缓存：5 只
- 合并观察行数：108,373
- 供应商熔断：未触发
- 后台采集进程：0

逐股缓存采用原子写入；被终止批次已经通过一次 `max-fetch=0` 无网络重建纳入最新清单。

## 本轮已完成

1. 修正东方财富退市样本验证口径。`000004` 的缓存现已进入独立交叉验证，共 103 个可比日期；其余 4 只退市样本仍无第二来源。
2. 新增退市验证的证券数量门槛，防止单只股票的长时间序列满足总体检查数。
3. 新增 `stock_active_valuation_observation_collector.py`，支持断点续采、旧缓存复用、原子写入、逐股状态、哈希清单和供应商熔断。
4. 历史行的 `available_date` 使用实际抓取日；数据继续标记为观察用途，禁止历史回测和模型晋级。
5. 专项测试通过：估值/PIT 测试 91 项；新采集器测试 4 项。

## 恢复命令

在项目根目录执行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.stock_active_valuation_observation_collector --as-of 2026-07-17 --universe candidates --max-fetch 25 --sleep-seconds 1.2 --retry-attempts 1 --max-consecutive-failures 2
```

每批结束后检查：

- `data_raw/long_hold_v4/manifests/stock_active_valuation_observation_latest.json`
- `data_raw/long_hold_v4/pit_history/observations/stock_active_valuation_history_eastmoney_status.csv`
- `data_raw/long_hold_v4/pit_history/observations/stock_active_valuation_history_eastmoney.csv.gz`

## 后续顺序

1. 完成剩余 121 只候选股采集；发生限流时保留熔断结果，不连续重试。
2. 建立东方财富观察与 BaoStock、聚宽重叠窗口的独立交叉验证报告。
3. 明确记录退市股第二估值源、历史股息率和版本化财务报表仍然缺失。
4. 重跑 PIT Gate、当前 50 万元现金模型、全量测试、编译和哈希审计。
5. 更新项目总审查报告与人工数据接口队列。

## 边界

东方财富序列来自当前最终快照，且候选池由当前研究快照选出。它可以用于当前低估判断和数据质量诊断，不能替代全生命周期、逐日可得的 PIT 估值数据。正式历史回测 Gate 继续保持阻断。
