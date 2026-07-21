# A-share Single-Stock PIT Intake V3.93

Purpose: register the first single-stock data readiness pilot for the quant research assistant framework.

Strict boundary: this catalog describes data usability only. It is not a stock-picking model, backtest, trade signal, or portfolio recommendation.

## Universe

- Stocks: 000001, 000002, 000004, 000006, 000007
- Output directory: `outputs/agent_runs/v3_93/single_stock_pit_intake`

## Source Decisions

- `tushare_raw_daily`: 5 stock-source rows approved only for PIT raw price-state features.
- `akshare_daily_qfq` and `akshare_financial_indicator`: 10 stock-source rows remain research-only.
- Blocked source rows: 0

## Key Data Restrictions

- AkShare `financial_indicator` lacks `available_date`; it cannot enter historical fundamental factor backtests.
- AkShare `daily_qfq` is adjusted but lacks row-level availability control; it is latest technical research data, not a governed historical adjusted-return label.
- Tushare daily data has `available_date`, but is `none_raw`; it is isolated to raw price-state features until corporate-action adjustment data is added.

## Scan Metadata

- Files scanned: 6889
- Scan mode: full_line_filter
- Prefix limit hits: 0
- Parse errors: 0

## Artifacts

- `single_stock_research_universe.csv`
- `single_stock_price_coverage.csv`
- `single_stock_fundamental_snapshot.csv`
- `single_stock_pit_readiness.csv`
- `single_stock_data_gap_register.csv`
- `single_stock_boundary_audit.csv`
- `single_stock_intake_decision.md`
- `data_dictionary.csv`
- `agent_run_manifest.json`

Data gap rows recorded: 16
