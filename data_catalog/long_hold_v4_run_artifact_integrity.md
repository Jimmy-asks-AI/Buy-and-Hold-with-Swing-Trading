# Long Hold V4 运行产物与快照血缘契约

## 目标

本契约处理两类工程风险：运行中断导致半成品被误当成当前结果，以及股票、ETF快照被并发覆盖或跨日期混合。它不证明策略有效，也不提高任何数据或模型的晋级等级。

## 不可变运行目录

研究链先写入：

```text
outputs/long_hold_v4/runs/<run_id>.tmp/
```

业务产物全部写完后，系统生成`run_manifest.json`和`run_manifest_seal.json`，再以目录原子改名发布到`runs/<run_id>/`。已存在的正式运行目录不会被覆盖。最后一步才原子更新`outputs/long_hold_v4/current`指针；中断运行和孤立的已发布目录都不能自动成为当前运行。

`run_manifest.json`逐项记录业务产物的相对路径、SHA-256、文件大小和schema版本，同时记录输入文件、代码、配置、账户版本和运行环境。`run_manifest_seal.json`保存最终清单文件的SHA-256。清单和seal本身不列入业务产物清单，避免自哈希递归。

## 订单绑定与哈希循环

订单文件既要绑定运行，又要作为运行产物被清单哈希。如果订单直接保存最终`run_manifest.json`的文件哈希，就会形成“清单包含订单哈希、订单又包含清单哈希”的循环依赖。

本版本将订单字段`run_manifest_sha256`定义为清单中`order_binding.sha256`：它是加入输出清单前，对固定运行上下文做规范化JSON后的SHA-256。最终订单文件写入后，`run_manifest.json`记录订单文件的实际字节哈希，seal再封印最终清单。执行端必须先验证seal、全部产物和外部输入，再用`order_binding.sha256`校验订单。单独持有订单文件、上下文哈希或seal中的任何一个，都不足以绕过完整验证。

## 快照血缘

股票和ETF builder各有一把全程锁，防止同类任务覆盖自己的缓存清单和summary；最终分部发布还共用`.snapshot_build.lock`，防止两个builder同时重建combined。

锁文件记录PID和创建时间。进程异常退出后，后续builder会超时阻断，不会自行删除锁。确认记录的进程已经结束后，才能人工移走对应`.stock_snapshot_builder.lock`、`.etf_snapshot_builder.lock`或`.snapshot_build.lock`；自动清理可能误删仍在工作的进程锁。

每个分部快照都有独立manifest，记录：

- 唯一`as_of_date`；
- 分部文件SHA-256、大小和schema版本；
- builder配置哈希；
- builder代码文件哈希。

只有股票和ETF分部都存在、各自通过哈希校验且`as_of_date`相同时，系统才生成`research_snapshot.csv`和`combined_snapshot_manifest.json`。缺失分部不会从旧combined恢复；日期不一致会阻断合并。研究pipeline把combined manifest和combined snapshot都纳入输入验证。

## 验证与恢复

验证当前运行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.run_artifacts verify --output-root outputs/long_hold_v4
```

隔离一个确认失败的临时运行：

```powershell
python -X utf8 -m strategy_lab.long_hold_v4.run_artifacts quarantine-temp --output-root outputs/long_hold_v4 --run-id <run_id>
```

命令只把`runs/<run_id>.tmp`原子移动到`quarantine/`，不删除证据，也不改动`current`。

## 旧目录迁移

旧版本把产物直接写在`outputs/long_hold_v4/current/`目录。新版本检测到该目录时会硬阻断，不会删除或覆盖。确认旧产物已备份后，手工执行：

```powershell
Move-Item outputs/long_hold_v4/current outputs/long_hold_v4/legacy_current_before_immutable_runs
```

随后重新运行研究链。首次完整发布会创建新的`current`指针文件。旧产物仅用于审计，不能自动转为已封印运行。

## 威胁边界

能够同时改写代码、输入数据和所有清单的本机高权限攻击者不在本地SHA-256模型的防护范围内；对该风险需要只读归档、代码签名或外部可信存储。当前机制防护的是误操作、文件漂移、局部篡改、并发覆盖、日期混合和进程中断。验证通过只说明本地运行产物与记录一致，不代表PIT数据合格、回测有效或系统可实盘交易。
