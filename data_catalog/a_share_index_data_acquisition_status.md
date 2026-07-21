# A股指数高质量数据获取状态

日期：2026-05-24  
数据根目录：`Introduction-to-Quantitative-Finance/data_raw/index/akshare_csindex/`

## 已落地指数

本轮已覆盖 8 个 A 股常用指数：

| 指数代码 | 指数 | 日线行数 | 日线覆盖 | 完整OHLC行 | close-only行 | 当前成分数 | 最新权重日期 | 权重合计 |
|---|---:|---:|---|---:|---:|---:|---|---:|
| 000015 | 上证红利 | 5194 | 2000-01-01 至 2026-05-22 | 5191 | 3 | 50 | 2026-04-30 | 100.000 |
| 000016 | 上证50 | 5437 | 2000-01-01 至 2026-05-22 | 5434 | 3 | 50 | 2026-04-30 | 99.997 |
| 000300 | 沪深300 | 5914 | 2000-01-01 至 2026-05-22 | 5191 | 723 | 300 | 2026-04-30 | 100.003 |
| 000852 | 中证1000 | 5193 | 2000-01-01 至 2026-05-22 | 5192 | 1 | 1000 | 2026-04-30 | 100.001 |
| 000905 | 中证500 | 5194 | 2000-01-01 至 2026-05-22 | 5191 | 3 | 500 | 2026-04-30 | 100.010 |
| 000906 | 中证800 | 5194 | 2000-01-01 至 2026-05-22 | 5191 | 3 | 800 | 2026-04-30 | 99.999 |
| 000922 | 中证红利 | 5194 | 2000-01-01 至 2026-05-22 | 5191 | 3 | 100 | 2026-04-30 | 100.001 |
| 000985 | 中证全指 | 5194 | 2000-01-01 至 2026-05-22 | 5191 | 3 | 5061 | 2026-04-30 | 100.030 |

补充数据：

- 中证指数全量列表：`index_list/index_csindex_all.csv`，2340 行。
- 日线：`daily_csindex/*.csv`，已包含 `is_full_ohlc_bar`、`is_close_only_bar`、`is_zero_volume_amount`。
- 最新中证估值快照：`valuation_latest_csindex/*.csv`，每个指数 20 行，覆盖 2026-04-24 至 2026-05-22。
- 历史 PE/PB：`valuation_pe_lg/*.csv`、`valuation_pb_lg/*.csv`。
  - 已支持：上证50、沪深300、中证500、中证1000、中证800。
  - 未拉取：中证全指、中证红利、上证红利；当前 Legulegu 接口未配置对应稳定符号，先不把错误结果混入主数据层。
- 当前成分：`constituents_current/*.csv`，成分日期 2026-05-22。
- 最新权重：`weights_latest/*.csv`，权重日期 2026-04-30。

## 质量复核

合并 manifest 与质量摘要：

```text
Introduction-to-Quantitative-Finance/data_raw/index/akshare_csindex/manifests/manifest_consolidated_2026-05-24T14-16-43.csv
Introduction-to-Quantitative-Finance/data_raw/index/akshare_csindex/manifests/quality_summary_2026-05-24T14-16-43.csv
```

复核结论：

- 43 个数据项全部为 `ok`。
- 日线、估值序列无重复日期、无非正收盘价。
- 成分和权重按 `date + asset` 检查重复，全部通过。
- 权重合计接近 100%，可用于当前截面暴露检查。
- 所有指数日线已打标 close-only 行，避免 OHLC 信号误用基日/回溯数据。

关键口径：

- 中证指数日线接口会保留指数基日或回溯收盘值；这类记录可用于收盘价收益序列，但不适合计算依赖开高低价的信号。
- 若策略使用开盘、最高、最低、振幅、日内结构等特征，必须过滤 `is_full_ohlc_bar == True`。
- 当前 AkShare/中证接口提供的是当前成分与最新权重，不是完整历史成分权重。做严格 point-in-time 历史回测时，需要补充 Tushare Pro、JoinQuant、Wind 或中证指数历史权重文件。

## 复现命令

核心三指数：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_index_data_harvester.py \
  --symbols 000016,000300,000905 \
  --datasets index_list,daily,valuation_latest,valuation_pe_lg,valuation_pb_lg,constituents,weights \
  --start-date 20000101 \
  --end-date 20260524 \
  --daily-backend auto \
  --sleep-seconds 0.2
```

扩展指数池：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_index_data_harvester.py \
  --symbols 000852,000906 \
  --datasets daily,valuation_latest,valuation_pe_lg,valuation_pb_lg,constituents,weights \
  --start-date 20000101 \
  --end-date 20260524 \
  --daily-backend auto \
  --sleep-seconds 0.2 \
  --resume

python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_index_data_harvester.py \
  --symbols 000985,000922,000015 \
  --datasets daily,valuation_latest,constituents,weights \
  --start-date 20000101 \
  --end-date 20260524 \
  --daily-backend auto \
  --sleep-seconds 0.2 \
  --resume
```

## 下一步

- 将 `daily_csindex` 和 `valuation_pe_lg/pb_lg` 接入指数择时模型，先做 close-only 可用区间，再做完整 OHLC 区间。
- 获取历史成分权重后，再做指数增强、成分股聚合因子和行业/权重漂移回测。
- 若继续扩展指数池，优先补齐创业板指、科创50、国证红利、红利低波、300价值、500价值等风格/行业/红利类指数。
