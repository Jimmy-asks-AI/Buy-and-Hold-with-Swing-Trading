# 真实数据接入前 SOP

日期：2026-05-23

## 目标

真实 A 股数据不得直接进入 walk-forward。必须先完成字段映射、point-in-time panel 构建、数据质量预检、注册表审计和 panel validation。

## 1. 字段映射

如果原始字段不是系统 canonical 名称，先运行字段映射：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/real_data_adapter.py \
  --input-csv data/raw_market.csv \
  --source-table market \
  --output-csv data/canonical_market.csv \
  --report-csv data/canonical_market_mapping_report.csv
```

`source-table` 可选：

- `market`
- `financial`
- `industry`

字段规则来自：

```text
Introduction-to-Quantitative-Finance/data_catalog/a_share_real_data_field_mapping_template.csv
```

## 2. 构建 point-in-time panel

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_panel_builder.py \
  --market-csv data/raw_market.csv \
  --financial-csv data/raw_financial.csv \
  --industry-csv data/raw_industry.csv \
  --output Introduction-to-Quantitative-Finance/outputs/real_panel/point_in_time_panel.csv
```

默认会自动按字段映射模板规范化字段。若输入已经是 canonical 字段，可加：

```text
--no-auto-map
```

## 3. 预检，不跑回测

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/quant_model_system.py preflight \
  --panel-csv Introduction-to-Quantitative-Finance/outputs/real_panel/point_in_time_panel.csv \
  --registry-csv Introduction-to-Quantitative-Finance/data_catalog/a_share_factor_registry_v0.csv \
  --config-json Introduction-to-Quantitative-Finance/configs/factor_factory_smoke.json \
  --output-dir Introduction-to-Quantitative-Finance/outputs/real_preflight_001
```

必须检查：

- `panel_validation.csv`
- `data_quality/gates.csv`
- `data_quality/factor_coverage.csv`
- `data_quality/availability_audit.csv`
- `registry_summary.csv`
- `SYSTEM_RUN_SUMMARY.md`

任一 `fail` 不得进入回测。

## 4. 运行 walk-forward

只有 preflight 无 fail 后，才能运行：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/quant_model_system.py walk-forward \
  --panel-csv Introduction-to-Quantitative-Finance/outputs/real_panel/point_in_time_panel.csv \
  --registry-csv Introduction-to-Quantitative-Finance/data_catalog/a_share_factor_registry_v0.csv \
  --config-json Introduction-to-Quantitative-Finance/configs/factor_factory_smoke.json \
  --output-dir Introduction-to-Quantitative-Finance/outputs/real_run_001 \
  --experiment-id real_run_001 \
  --hypothesis "first real A-share governed multi-factor run"
```

## 强制拦截项

- 无效交易日期。
- 重复 `date-asset`。
- 可得日期晚于交易日期。
- 缺失已存在因子的 availability column。
- 负成交额。
- 非正市值。
- panel validation 出现 fail。

## 注意

`promote_to_paper` 只允许进入纸面跟踪，不允许直接实盘。真实数据首次接入后，必须至少完成数据质量复核、样本外 walk-forward、纸面漂移监控和人工复盘。
