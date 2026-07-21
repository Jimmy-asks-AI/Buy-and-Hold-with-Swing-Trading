# A-share Market Participation Breadth Feature Layer V3.67

## Dataset Role

V3.67 converts accepted Tushare daily-only raw OHLCV diagnostics into market participation breadth features.

## Governance

- Raw daily prices are used only for cross-sectional breadth and activity diagnostics.
- Stock return labels, adjusted returns, portfolio outputs, and model promotion are blocked.
- Feature timing is after-close for next trade-date research.

## Produced Shape

- Feature rows: `6395`
- Date range: `20000104` to `20260528`
- History-sufficient rows: `6335`
- Trailing window: `252`