#!/usr/bin/env python
"""LOB reconstruction and deep high-frequency factor research helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


def _side_masks(side: pd.Series) -> tuple[pd.Series, pd.Series]:
    if pd.api.types.is_numeric_dtype(side):
        numeric = pd.to_numeric(side, errors="coerce")
        return numeric > 0, numeric < 0
    text = side.astype(str).str.upper()
    buy = text.isin({"B", "BUY", "1", "TRUE", "买", "买入", "BID"})
    sell = text.isin({"S", "SELL", "-1", "卖", "卖出", "ASK", "OFFER"})
    return buy, sell


def reconstruct_lob_from_events(
    events: pd.DataFrame,
    timestamp_col: str,
    order_id_col: str,
    side_col: str,
    price_col: str,
    quantity_col: str,
    event_type_col: str,
    levels: int = 10,
) -> pd.DataFrame:
    """Reconstruct a simple limit order book from add/cancel/trade events.

    This is a research-grade simulator for normalized event data. Exchange-
    specific details such as market-order price completion should be handled
    before calling this function.
    """
    if levels <= 0:
        raise ValueError("levels must be positive.")
    data = events.sort_values(timestamp_col).copy()
    book: dict[object, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    def snapshot(ts: object) -> dict[str, object]:
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        for order in book.values():
            qty = float(order["quantity"])
            if qty <= EPS:
                continue
            price = float(order["price"])
            if order["side"] == "buy":
                bids[price] = bids.get(price, 0.0) + qty
            elif order["side"] == "sell":
                asks[price] = asks.get(price, 0.0) + qty
        row: dict[str, object] = {"timestamp": ts}
        for i, (price, qty) in enumerate(sorted(bids.items(), reverse=True)[:levels], start=1):
            row[f"bid_price_{i}"] = price
            row[f"bid_qty_{i}"] = qty
        for i, (price, qty) in enumerate(sorted(asks.items())[:levels], start=1):
            row[f"ask_price_{i}"] = price
            row[f"ask_qty_{i}"] = qty
        return row

    for _, row in data.iterrows():
        order_id = row[order_id_col]
        event_type = str(row[event_type_col]).lower()
        qty = float(pd.to_numeric(pd.Series([row[quantity_col]]), errors="coerce").iloc[0] or 0.0)
        price = float(pd.to_numeric(pd.Series([row[price_col]]), errors="coerce").iloc[0] or 0.0)
        buy_mask, sell_mask = _side_masks(pd.Series([row[side_col]]))
        side = "buy" if bool(buy_mask.iloc[0]) else "sell" if bool(sell_mask.iloc[0]) else None
        if event_type in {"add", "new", "insert", "委托", "新增"} and side is not None:
            book[order_id] = {"side": side, "price": price, "quantity": qty}
        elif event_type in {"cancel", "delete", "撤单", "删除"}:
            if order_id in book:
                book[order_id]["quantity"] = max(0.0, float(book[order_id]["quantity"]) - qty)
                if book[order_id]["quantity"] <= EPS:
                    book.pop(order_id, None)
        elif event_type in {"trade", "fill", "成交"}:
            if order_id in book:
                book[order_id]["quantity"] = max(0.0, float(book[order_id]["quantity"]) - qty)
                if book[order_id]["quantity"] <= EPS:
                    book.pop(order_id, None)
        rows.append(snapshot(row[timestamp_col]))
    return pd.DataFrame(rows)


def lob_relative_strength(
    snapshots: pd.DataFrame,
    bid_qty_prefix: str = "bid_qty_",
    ask_qty_prefix: str = "ask_qty_",
    levels: int = 10,
    output_col: str = "lob_relative_strength",
) -> pd.DataFrame:
    """Order-book relative strength from bid and ask depth."""
    out = snapshots.copy()
    bid_cols = [f"{bid_qty_prefix}{i}" for i in range(1, levels + 1) if f"{bid_qty_prefix}{i}" in out.columns]
    ask_cols = [f"{ask_qty_prefix}{i}" for i in range(1, levels + 1) if f"{ask_qty_prefix}{i}" in out.columns]
    bid = out[bid_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if bid_cols else 0.0
    ask = out[ask_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1) if ask_cols else 0.0
    out[output_col] = (bid - ask) / (bid + ask).replace(0.0, np.nan)
    return out


def order_flow_relative_strength(
    flow: pd.DataFrame,
    buy_col: str,
    sell_col: str,
    output_col: str,
) -> pd.DataFrame:
    """Generic relative strength for order, trade, or cancel flow."""
    out = flow.copy()
    buy = pd.to_numeric(out[buy_col], errors="coerce").fillna(0.0)
    sell = pd.to_numeric(out[sell_col], errors="coerce").fillna(0.0)
    out[output_col] = (buy - sell) / (buy + sell).replace(0.0, np.nan)
    return out


def limit_order_execution_probability_target(
    intervals: pd.DataFrame,
    buy_executed_col: str,
    buy_total_col: str,
    sell_executed_col: str,
    sell_total_col: str,
    output_col: str = "limit_order_execution_probability",
) -> pd.DataFrame:
    """Target proxy: buy limit execution ratio minus sell limit execution ratio."""
    out = intervals.copy()
    buy_ratio = pd.to_numeric(out[buy_executed_col], errors="coerce") / pd.to_numeric(out[buy_total_col], errors="coerce").replace(0.0, np.nan)
    sell_ratio = pd.to_numeric(out[sell_executed_col], errors="coerce") / pd.to_numeric(out[sell_total_col], errors="coerce").replace(0.0, np.nan)
    out[output_col] = buy_ratio - sell_ratio
    return out


def decompose_buying_intention_lob(
    intervals: pd.DataFrame,
    net_add_col: str,
    net_cancel_col: str,
    net_trade_col: str,
    passive_net_buy_col: str,
    total_amount_col: str,
    output_prefix: str = "lob_intention",
) -> pd.DataFrame:
    """Decompose buying intention into add, cancel, trade, and passive components."""
    out = intervals.copy()
    total = pd.to_numeric(out[total_amount_col], errors="coerce").replace(0.0, np.nan)
    components = {
        "net_add": net_add_col,
        "net_cancel": net_cancel_col,
        "net_trade": net_trade_col,
        "passive_net_buy": passive_net_buy_col,
    }
    for name, col in components.items():
        out[f"{output_prefix}_{name}_ratio"] = pd.to_numeric(out[col], errors="coerce") / total
    component_cols = [f"{output_prefix}_{name}_ratio" for name in components]
    out[f"{output_prefix}_equal_weight"] = out[component_cols].mean(axis=1)
    return out


def ic_weighted_composite(
    df: pd.DataFrame,
    date_col: str,
    component_cols: Sequence[str],
    forward_return_col: str,
    lookback: int = 24,
    min_history: int = 6,
    output_col: str = "ic_weighted_composite",
) -> pd.DataFrame:
    """Build a trailing IC-weighted composite from multiple components."""
    out = df.sort_values(date_col).copy()
    out[output_col] = np.nan
    dates = pd.Index(sorted(out[date_col].dropna().unique()))
    comps = list(component_cols)
    for pos, date in enumerate(dates):
        hist_dates = dates[max(0, pos - lookback) : pos]
        hist = out[out[date_col].isin(set(hist_dates))]
        if len(hist_dates) < min_history:
            weights = pd.Series(1.0 / len(comps), index=comps)
        else:
            ics = {}
            for col in comps:
                values = []
                for _, group in hist[[date_col, col, forward_return_col]].dropna().groupby(date_col):
                    if group.shape[0] >= 10:
                        values.append(group[col].rank().corr(group[forward_return_col].rank()))
                ics[col] = float(np.nanmean(values)) if values else 0.0
            weights = pd.Series(ics).fillna(0.0)
            denom = weights.abs().sum()
            weights = weights / denom if denom > EPS else pd.Series(1.0 / len(comps), index=comps)
        idx = out.index[out[date_col] == date]
        out.loc[idx, output_col] = out.loc[idx, comps].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) @ weights.reindex(comps).to_numpy(dtype=float)
    return out


def make_highfreq_sequence_dataset(
    panel: pd.DataFrame,
    date_col: str,
    asset_col: str,
    feature_cols: Sequence[str],
    target_col: str,
    lookback: int = 20,
) -> dict[str, object]:
    """Create an N x T x F tensor and target vector for high-frequency sequence models."""
    features = list(feature_cols)
    data = panel.sort_values([asset_col, date_col]).copy()
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    index_rows: list[dict[str, object]] = []
    for asset, group in data.groupby(asset_col, sort=False):
        feat = group[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        target = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=float)
        dates = group[date_col].to_numpy()
        for pos in range(lookback, len(group)):
            window = feat[pos - lookback : pos]
            if np.isnan(window).any() or np.isnan(target[pos]):
                continue
            x_list.append(window)
            y_list.append(float(target[pos]))
            index_rows.append({"asset": asset, "date": dates[pos]})
    x = np.stack(x_list, axis=0) if x_list else np.empty((0, lookback, len(features)))
    return {"x": x, "y": np.asarray(y_list, dtype=float), "index": pd.DataFrame(index_rows), "features": features}


def rolling_train_validation_splits(
    dates: Iterable[object],
    train_periods: int = 100,
    validation_periods: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """Create rolling train/validation date splits for weekly model iteration."""
    unique_dates = pd.Index(sorted(pd.Series(list(dates)).dropna().unique()))
    rows: list[dict[str, object]] = []
    end = train_periods + validation_periods
    split_id = 0
    while end <= len(unique_dates):
        train = unique_dates[end - validation_periods - train_periods : end - validation_periods]
        valid = unique_dates[end - validation_periods : end]
        rows.append(
            {
                "split": split_id,
                "train_start": train[0],
                "train_end": train[-1],
                "valid_start": valid[0],
                "valid_end": valid[-1],
                "n_train_dates": len(train),
                "n_valid_dates": len(valid),
            }
        )
        split_id += 1
        end += step
    return pd.DataFrame(rows)


def rank_ic_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    """Negative rank IC, matching the report's IC-oriented training objective."""
    pred_rank = pd.Series(prediction).rank(method="average")
    target_rank = pd.Series(target).rank(method="average")
    corr = pred_rank.corr(target_rank)
    return float(-corr) if not pd.isna(corr) else np.nan


def orthogonalization_layer(
    predictions: pd.DataFrame,
    date_col: str,
    pred_col: str,
    control_cols: Sequence[str],
    output_col: str = "orthogonal_prediction",
    min_count: int | None = None,
) -> pd.DataFrame:
    """Cross-sectional residual layer for model outputs."""
    controls = list(control_cols)
    min_count = min_count or len(controls) + 8
    out = predictions.copy()
    out[output_col] = np.nan
    for _, group in out.groupby(date_col, sort=True):
        work = group[[pred_col, *controls]].apply(pd.to_numeric, errors="coerce").dropna()
        if work.shape[0] < min_count:
            continue
        x = np.column_stack([np.ones(work.shape[0]), work[controls].to_numpy(dtype=float)])
        y = work[pred_col].to_numpy(dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        out.loc[work.index, output_col] = y - x @ beta
    return out


def risk_adjusted_excess_return_target(
    returns: pd.DataFrame,
    date_col: str,
    asset_col: str,
    return_col: str,
    benchmark_return_col: str | None = None,
    volatility_lookback: int = 20,
    output_col: str = "risk_adjusted_excess_return",
) -> pd.DataFrame:
    """Build a risk-adjusted target for deep high-frequency stock selection."""
    out = returns.sort_values([asset_col, date_col]).copy()
    raw = pd.to_numeric(out[return_col], errors="coerce")
    if benchmark_return_col is None:
        bench = out.groupby(date_col)[return_col].transform("mean")
    else:
        bench = pd.to_numeric(out[benchmark_return_col], errors="coerce")
    excess = raw - bench
    vol = (
        out.assign(_excess=excess)
        .groupby(asset_col)["_excess"]
        .transform(lambda s: s.rolling(volatility_lookback, min_periods=max(5, volatility_lookback // 2)).std())
    )
    out[output_col] = excess / vol.replace(0.0, np.nan)
    return out


def deep_highfreq_improvement_attempts() -> pd.DataFrame:
    """Research checklist from Haitong's 2022 deep high-frequency follow-up."""
    rows = [
        (
            "orthogonal_training",
            "Add an orthogonal layer or penalty during training so output is less correlated with industry, style, and low-frequency technical factors.",
            "Promising; report stronger weekly selection and better CSI500 enhancement contribution.",
        ),
        (
            "feature_compression",
            "Compress high-frequency input features by economic logic before training.",
            "Useful; reduced feature set did not materially weaken selection ability.",
        ),
        (
            "feature_standardization",
            "Compare raw features, time-series standardization, and cross-sectional standardization.",
            "Cross-sectional standardization performed best in the report.",
        ),
        (
            "higher_frequency_input",
            "Raise input frequency to 10-minute bars, increasing sequence length.",
            "Not automatically better; simple GRU/LSTM may forget long sequences.",
        ),
        (
            "train_validation_ratio",
            "Shorten validation window so more recent samples enter training.",
            "Promising, but must be checked against validation overfitting.",
        ),
        (
            "environment_variables",
            "Add market environment variables to the input tensor.",
            "Weak in the report when added naively; regime variables need explicit design.",
        ),
        (
            "target_adjustment",
            "Use risk-adjusted excess return instead of raw forward return.",
            "Can lower weekly IC slightly while improving long-only excess return.",
        ),
        (
            "longer_training_window",
            "Extend rolling training history.",
            "Promising if market microstructure is stable enough.",
        ),
        (
            "model_complexity",
            "Increase model parameter count.",
            "Not useful alone; expand information content before increasing complexity.",
        ),
    ]
    return pd.DataFrame(rows, columns=["attempt", "implementation", "lesson"])


def attention_pool(hidden_states: np.ndarray, query: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention pooling for one sequence of hidden states."""
    states = np.asarray(hidden_states, dtype=float)
    if states.ndim != 2:
        raise ValueError("hidden_states must be T x H.")
    query_vec = states[-1] if query is None else np.asarray(query, dtype=float)
    scores = states @ query_vec / np.sqrt(max(states.shape[1], 1))
    scores = scores - np.nanmax(scores)
    weights = np.exp(scores)
    weights = weights / max(weights.sum(), EPS)
    pooled = weights @ states
    return pooled, weights


def residual_attention_pool(hidden_states: np.ndarray, query: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Residual attention pooling: last state plus attention-pooled history."""
    pooled, weights = attention_pool(hidden_states, query=query)
    return hidden_states[-1] + pooled, weights


def factor_autocorrelation(
    factor_panel: pd.DataFrame,
    asset_col: str,
    date_col: str,
    factor_col: str,
    lag: int = 1,
) -> float:
    """Average cross-sectional factor autocorrelation by date."""
    data = factor_panel.sort_values([asset_col, date_col]).copy()
    data["_lag"] = data.groupby(asset_col)[factor_col].shift(lag)
    values = []
    for _, group in data[[date_col, factor_col, "_lag"]].dropna().groupby(date_col):
        if group.shape[0] >= 10:
            values.append(group[factor_col].corr(group["_lag"]))
    return float(np.nanmean(values)) if values else np.nan


def deep_highfreq_model_checklist() -> pd.DataFrame:
    rows = [
        ("lob", "Normalize exchange-specific order, cancel, trade, and market-order fields before LOB reconstruction."),
        ("execution", "LOB-based execution tests need latency, queue position, forced fill, and order-size limits."),
        ("features", "LOB components should be tested individually before nonlinear composition."),
        ("dataset", "Sequence tensor must use only trailing high-frequency features and forward returns."),
        ("split", "Use rolling train/validation splits; never random-shuffle time-series cross sections."),
        ("objective", "IC or rank-IC loss aligns better with cross-sectional alpha than MSE alone."),
        ("orthogonal_layer", "Neutralize model output to known low-frequency and high-frequency factors before claiming novelty."),
        ("nine_attempts", "Track the nine improvement attempts: orthogonalization, feature compression, standardization, frequency, split ratio, environment variables, target adjustment, longer training, and complexity."),
        ("attention", "Attention helps longer high-frequency sequences but can reduce older-regime performance; test residual attention."),
        ("turnover", "Report factor autocorrelation and top-quantile turnover together with IC."),
    ]
    return pd.DataFrame(rows, columns=["gate", "requirement"])
