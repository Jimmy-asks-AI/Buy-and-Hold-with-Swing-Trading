# Factor: 低 PB

Domain: value
Universe: A-share equities
Frequency: monthly candidate
Data Fields: PB, market cap, industry, trading status, ST status, listed days, turnover, amount

## Rationale

价格相对账面净资产较低的公司可能存在估值修复、均值回归或风险补偿。

## Formula

```text
value_score = -PB
```

方向：`value_score` 越高，代表 PB 越低。

## Universe Filter

- 剔除 ST 和 *ST。
- 剔除上市不足 180 个交易日。
- 剔除停牌和不可交易股票。
- 剔除 PB <= 0 或 PB 缺失。
- 剔除成交额过低股票。

## Evaluation

- IC。
- Rank IC。
- 5 组或 10 组分组收益。
- 最便宜组 - 最贵组。
- 多头组相对中证 500、中证 800 或全 A 等权。
- 换手率。
- 成本敏感性。

## Robustness

- 分年度。
- 牛市、熊市、震荡市。
- 分行业。
- 分市值。
- 财报公告日口径。
- 样本内/样本外。

## Failure Modes

- 价值陷阱。
- 行业集中，尤其金融、地产、周期。
- 盈利恶化。
- 资产质量差。
- 低流动性。
- 市场长期不给估值修复。

## Implementation Notes

低 PB 不等于安全。它必须和质量、盈利稳定性、行业暴露、流动性和组合约束一起看。

