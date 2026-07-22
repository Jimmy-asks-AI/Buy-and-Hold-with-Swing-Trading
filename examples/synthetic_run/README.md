# 合成纸面交易 replay

本目录只使用程序生成的虚构证券、虚构财务字段和虚构行情。`000000` 与 `999999` 在本示例中只是测试标识，名称均带有“非真实证券”；这里没有证券研究结论、收益承诺或实盘建议。

在仓库根目录执行：

```powershell
python -m strategy_lab.long_hold_v4.synthetic_replay
```

命令会离线重建 `output/`，完整走过 `snapshot → candidate → FULL_SNAPSHOT target → order → fill → account/ledger → NAV`，再逐文件校验 `expected_manifest.json`。它不读取 `data_raw/`、真实账户或外部 API。

连续运行相同代码、配置和合成输入，应输出相同的 `bundle_sha256`。只有维护者有意更新合成契约时才使用 `--write-expected`，并应审查预期哈希的变化。
