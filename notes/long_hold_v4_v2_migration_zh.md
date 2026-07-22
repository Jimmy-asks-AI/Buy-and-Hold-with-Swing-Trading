# Long Hold V4 v2 迁移说明

## 字段迁移

`expected_reversion_edge` 更名为 `ma20_reversion_distance`，含义保持为 `max((MA20-close)/close, 0)`。

- 新生成的价格特征、诊断表和运行产物只写 `ma20_reversion_distance`。
- 读取端在一个兼容周期内接受旧字段：从 v2 发布日起至 2026-10-31，或目标 schema v3 发布之日，以较早者为准。
- 同时出现新旧字段且数值一致时，以新字段为准；两者冲突时硬失败。
- 兼容期结束后，旧字段读取分支应删除。旧文件应在此之前重算或改列名。

## 目标表迁移

旧目标表没有明确语义，不能直接进入正式回测。迁移步骤：

1. 确认每个目标日是否确实覆盖完整资产目标。
2. 完整目标写 `target_schema_version=2`、`target_semantics=FULL_SNAPSHOT`，并为该日每一行写相同的 `snapshot_asset_count`。
3. 上期非零资产本期若要退出，增加显式零目标行；不要删行表示清仓。
4. 稀疏表只能标记为 `DELTA`，且只供显式开启增量语义的诊断流程使用。

无法证明是完整快照的历史文件一律按 `DELTA` 处理，不能补标签后进入正式回测。

## 价格与账户迁移

研究价格继续放在 `data.price_directory`，须补 `price_basis`，并与 `return_basis` 一致。真实账户另建 `data.execution_price_directory`，文件使用未复权 OHLC，声明：

```text
return_basis=unadjusted_executable
price_basis=unadjusted_executable
```

账户 JSON 的股数、现金、平均成本和事件账本不改写。迁移后第一次盯市前应完成三项检查：执行价文件覆盖全部现有持仓；复权口径全部被拒绝；拆并股、转增和换股已通过账户事件调整股数。缺一项就保持账户阻断状态。

成交估值快照同样必须增加 `price_basis=unadjusted_executable`。旧快照无此字段时失败，不自动推断。

## 财务产物迁移

五年 CAGR 旧值若只用了五个端点，应作废重算。新产物增加 `_start_date`、`_end_date`、`_span_years` 和连续性状态字段。财政年度不足、缺失或重复时保留空指标与阻断原因，不沿用旧值。

## 迁移验收

- 正式回测中不存在 `DELTA`。
- 目标快照声明数与行数一致，退出资产有显式零行。
- 账户盯市输入全部为 `unadjusted_executable`。
- 新产物不再出现 `expected_reversion_edge`。
- 五年、三年 CAGR 分别具备六个、四个连续端点，并记录真实时间跨度。
