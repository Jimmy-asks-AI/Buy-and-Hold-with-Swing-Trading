# 买卖单 Alpha 与主动买入行为学习笔记

Source:
- `资料/卖方金工研报/海通报告/2019-11-07_海通证券_选股因子系列研究（五十六）：买卖单数据中的Alpha.pdf`
- `资料/卖方金工研报/海通报告/2020-01-13_海通证券_选股因子系列研究（五十七）：基于主动买入行为的选股因子.pdf`

Date: 2026-05-22
Domain: Tick Data / Order Flow Alpha
Priority: P1
Status: corrected

## One-Line Takeaway

逐笔数据不仅能用 BS 标志，还能用叫买/叫卖序号还原买卖单；大买占比、买卖单集中度、主买占比和主买强度都能形成短周期 alpha，但必须正交低频因子并严控成本。

## Research Question

- 从逐笔成交的“笔”还原为买卖“单”后，能否更好刻画投资者交易结构？
- 动态大单成交金额占比和买卖单集中度是否有选股能力？
- 主动买入金额和主动卖出金额能否构建稳定的主买占比和主买强度因子？

## Data

- 逐笔成交：成交编号、成交价格、成交数量、BS 标志、叫买序号、叫卖序号。
- 分钟级主动买入金额和主动卖出金额。
- 行业、市值、中盘、换手、反转、波动、估值、盈利、盈利增长等常规控制因子。
- 不同股票池：全 A、中证 800、中证 500、沪深 300。

## Method

### 从笔到单

同一笔委托可能被对手盘拆成多笔成交。用叫买序号和叫卖序号可把逐笔成交重新合成为买单和卖单。基于单维度而不是笔维度，能更接近投资者订单行为。

### 动态大单阈值

报告不采用固定金额阈值，而是在每个股票、每个交易日内用：

```text
big_order_threshold = mean(order_amount) + N * std(order_amount)
```

其中 `N=1` 和 `N=3` 都被测试。过严阈值会导致股票之间缺乏区分度，报告中 `N=1` 通常更有效。

### 大单成交金额占比

```text
big_sell_amount_ratio = big_sell_amount / total_amount
big_buy_amount_ratio = big_buy_amount / total_amount
big_buy_minus_sell_ratio = big_buy_amount_ratio - big_sell_amount_ratio
big_order_amount_ratio = big_buy_amount_ratio + big_sell_amount_ratio
```

正交剔除行业、市值、中盘、换手、反转、波动、估值、盈利、盈利成长后，大买占比和大买减大卖占比有显著正向选股能力。

### 买卖单集中度

```text
buy_concentration = sum(buy_order_amount_k^2) / total_amount^2
sell_concentration = sum(sell_order_amount_k^2) / total_amount^2
concentration_diff = buy_concentration - sell_concentration
concentration_sum = buy_concentration + sell_concentration
```

集中度越高，说明成交更集中在少数订单上，大单特征更明显。报告建议对偏态分布做对数调整。

### 主动买入行为

BS 标志中，`B` 表示主动买入，`S` 表示主动卖出。报告将逐笔数据降频到分钟级，构建：

```text
active_buy_ratio_day = active_buy_amount / daily_total_amount
active_buy_ratio_session = active_buy_amount / session_total_amount
intraday_active_buy_intensity = mean(active_buy_amount) / std(active_buy_amount)
intraday_active_net_buy_intensity = mean(active_buy_amount - active_sell_amount) / std(active_buy_amount - active_sell_amount)
```

时段划分：

- 全天：9:30-14:56。
- 开盘后：9:30-9:59。
- 盘中：10:00-14:26。
- 收盘前：14:27-14:56。

## Key Findings

- 大买成交金额占比在正交后表现强，1 倍标准差阈值下优于 3 倍标准差。
- 大买减大卖占比也有正向选股能力。
- 集中度因子在原始和正交后均有效，但主要集中于中小盘股票。
- 大单成交金额占比类因子在沪深 300、中证 500、中证 800 内仍较稳定。
- 月度有效的大单和集中度因子在更高调仓频率下仍有效，ICIR 往往提高，但毛收益未扣成本。
- 主买占比原始因子较弱，正交后全天、开盘后、盘中主买占比表现较好。
- 收盘前主买占比呈现反转效应，可能来自尾盘成交占比，而不是主动买入本身。
- 日内净主买强度比日间强度更强；开盘后日内净主买强度表现突出。
- 主动买入类因子在沪深 300 内多头效应更强。

## Testable Hypothesis

1. 大买成交金额占比正交后，在中大盘股票池中仍有稳定月度选股能力。
2. 买卖单集中度在中小盘中更有效，在沪深 300 中弱化。
3. 主买占比和日内净主买强度正交后有正向收益预测能力。
4. 高频因子调仓频率提升会提高毛收益和 ICIR，但扣费后最优频率可能不是最高频。

## Factor/Signal/Strategy Extraction

### Required Controls

正交控制因子至少包括：

- 行业。
- 市值和中盘。
- 换手率。
- 反转。
- 系统波动和特质波动。
- 估值、盈利、盈利增长。

### Trading Rules

- 月度、半月、周度、2 日、1 日频率都可测试。
- 窗口长度要随调仓频率调整，但窗口与调仓频率不应被硬绑定，后续需独立优化。
- 高频因子的收益报告必须扣交易成本和冲击成本。

## Risks And Failure Modes

- BS 标志在涨停/跌停分钟会反直觉，涨停成交可能被识别为主动卖出，跌停成交可能被识别为主动买入。
- 大单阈值过严会降低截面区分度。
- 调仓频率越高，毛收益越高并不代表净收益越高。
- 逐笔数据清洗错误会直接制造伪 alpha。
- 集中度因子可能只是小盘、低流动性或低换手暴露。

## Replication Plan

1. 用叫买/叫卖序号重建买卖单。
2. 用均值 + N 倍标准差识别大单，测试 `N=1` 和 `N=3`。
3. 计算大买占比、大卖占比、大买减大卖、大单占比。
4. 计算买卖单集中度及对数集中度。
5. 用 BS 标志计算主买占比和主买强度。
6. 剔除涨跌停分钟。
7. 对常规低频因子正交后测试 IC、ICIR、分层、多空、换手和成本。
8. 分股票池和调仓频率复测。

## Reusable Knowledge

- 逐笔因子不能只看 BS 标志，订单号能提供更接近真实订单行为的信息。
- 高频 alpha 必须先归入短周期交易模块，再决定是否服务于做 T 或指数增强。
- 正交前后方向变化本身是重要信息，说明原始因子含有强烈风格暴露。
