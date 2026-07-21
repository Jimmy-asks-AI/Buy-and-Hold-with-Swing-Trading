# A 股真实数据获取状态

日期：2026-05-24

## 当前已落地数据

数据根目录：

```text
Introduction-to-Quantitative-Finance/data_raw/akshare/
```

已成功获取：

- 当前 A 股证券列表：`stock_list/stock_info_a_code_name.csv`
  - 行数：5522
  - 字段：`asset,name,fetched_at`
- 交易日历：`calendar/trade_calendar.csv`
  - 行数：8797
  - 覆盖：1990-12-19 至 2026-12-31
- pilot 日线前复权数据：`daily_qfq/*.csv`
  - 股票：`000001`、`000002`、`000004`、`000006`、`000007`
  - 起始：2000-01-01
  - 字段：`date,open,high,low,close,volume,amount,outstanding_share,turnover,asset,turnover_pct,adj_close,adjust,data_source,fetched_at`
- pilot 财务摘要：`financial_indicator/*.csv`
  - 股票：`000001`、`000002`、`000004`、`000006`、`000007`
  - 数据源：`akshare.stock_financial_abstract_ths`

## 已验证的数据链路

运行命令：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_data_harvester.py \
  --datasets stock_list,trade_calendar,daily,financial_indicator \
  --start-date 20000101 \
  --end-date 20260524 \
  --start-year 2000 \
  --adjust qfq \
  --daily-backend auto \
  --financial-backend auto \
  --max-symbols 5 \
  --sleep-seconds 0.2 \
  --resume
```

验证结果：

```text
daily_qfq:
000001 ok 6230 rows
000002 ok 6178 rows
000004 ok 6029 rows
000006 ok 6190 rows
000007 ok 5678 rows

financial_indicator:
000001 ok 121 rows
000002 ok 117 rows
000004 ok 117 rows
000006 ok 119 rows
000007 ok 119 rows
```

## 发现的问题

- AkShare 的东财日线接口 `stock_zh_a_hist` 当前在本机代理下会被远端断开。
- 已改为自动 fallback 到 `stock_zh_a_daily`，pilot 成功。
- AkShare 的 `stock_financial_analysis_indicator` 对部分股票失败。
- 已改为自动 fallback 到 `stock_financial_abstract_ths`，pilot 成功。

## 全量获取命令

获取当前 A 股列表中所有股票的 2000 年后前复权日线和财务摘要：

```text
python -X utf8 Introduction-to-Quantitative-Finance/strategy_lab/a_share_data_harvester.py \
  --datasets stock_list,trade_calendar,daily,financial_indicator \
  --start-date 20000101 \
  --end-date 20260524 \
  --start-year 2000 \
  --adjust qfq \
  --daily-backend auto \
  --financial-backend auto \
  --sleep-seconds 0.2 \
  --resume
```

注意：按 pilot 速度估算，全量 5522 只股票可能需要数小时，并会产生数 GB 级别文件。脚本支持 `--resume`，中断后可继续。

## 量化精度分层

AkShare 当前用途：

- 适合广度覆盖、快速原型、横向字段探索。
- 不应直接作为生产级 point-in-time 财务数据库。
- 当前证券列表更接近当前仍可查询股票池，可能存在退市股票覆盖不足，存在幸存者偏差风险。

Tushare / JoinQuant 推荐用途：

- 生产级日线、复权因子、停复牌、涨跌停、ST、上市退市状态。
- 财务三表、财务指标、分红送转、公告日或入库日。
- 完整历史股票池，降低幸存者偏差。

## 凭证状态

本机当前状态：

- 未发现 `TUSHARE_TOKEN` 环境变量。
- 未发现 `JQ` / `JOINQUANT` 环境变量。
- `tushare` SDK 未安装。
- `jqdatasdk` SDK 未安装。
- Chrome 登录态不能被当前 Python 数据管线直接、安全地当成 API 凭证使用。

已提供凭证模板：

```text
Introduction-to-Quantitative-Finance/configs/data_credentials.example.json
```

真实 token/密码不得写入 git 或公共报告。
