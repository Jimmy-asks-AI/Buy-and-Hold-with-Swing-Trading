# A 股真实数据获取 Pilot 复盘

日期：2026-05-24

## 目标

按用户要求开始自行获取真实 A 股数据，优先验证 2000 年后宽口径数据抓取链路，并区分“广度数据”和“高精度量化数据”。

## 执行过程

1. 检查本机 SDK：
   - `akshare` 已安装。
   - `tushare` 未安装。
   - `jqdatasdk` 未安装。
2. 检查环境变量：
   - 未发现 Tushare / JoinQuant 相关 token。
3. 联网测试：
   - AkShare 股票列表接口可用，返回 5522 只当前 A 股。
4. 创建抓取器：
   - `strategy_lab/a_share_data_harvester.py`
5. 运行 pilot：
   - 股票列表。
   - 交易日历。
   - 前 5 只股票 2000 年后前复权日线。
   - 前 5 只股票财务摘要。

## 已落地产物

- `data_raw/akshare/stock_list/stock_info_a_code_name.csv`
- `data_raw/akshare/calendar/trade_calendar.csv`
- `data_raw/akshare/daily_qfq/000001.csv`
- `data_raw/akshare/daily_qfq/000002.csv`
- `data_raw/akshare/daily_qfq/000004.csv`
- `data_raw/akshare/daily_qfq/000006.csv`
- `data_raw/akshare/daily_qfq/000007.csv`
- `data_raw/akshare/financial_indicator/000001.csv`
- `data_raw/akshare/financial_indicator/000002.csv`
- `data_raw/akshare/financial_indicator/000004.csv`
- `data_raw/akshare/financial_indicator/000006.csv`
- `data_raw/akshare/financial_indicator/000007.csv`
- `data_catalog/a_share_data_acquisition_status.md`

## 结果

日线 pilot 全部成功：

- `000001`：6230 行。
- `000002`：6178 行。
- `000004`：6029 行。
- `000006`：6190 行。
- `000007`：5678 行。

财务摘要 pilot 全部成功：

- `000001`：121 行。
- `000002`：117 行。
- `000004`：117 行。
- `000006`：119 行。
- `000007`：119 行。

## 问题与修正

- 问题 1：`stock_zh_a_hist` 东财接口被当前代理断开。
  - 修正：新增日线 backend fallback，自动切到 `stock_zh_a_daily`。
- 问题 2：`stock_financial_analysis_indicator` 对样本失败。
  - 修正：新增财务 backend fallback，自动切到 `stock_financial_abstract_ths`。
- 问题 3：AkShare 股票池不保证覆盖全部退市股票。
  - 修正：在状态报告中明确幸存者偏差风险，高精度生产数据仍需 Tushare/JQData。

## 复习与纠错

- 不能把“能下载当前股票列表”解释为“完整历史全市场股票池”。
- 不能把 AkShare 财务摘要直接当作 point-in-time 财务数据库。
- 全量抓取需要 resumable manifest，否则网络中断后无法治理。
- 生产级量化数据必须补充复权因子、停复牌、涨跌停、ST、上市退市状态和公告日。
