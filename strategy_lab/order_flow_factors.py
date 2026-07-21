#!/usr/bin/env python
"""Order-flow and big-order factor helpers.

Use this module to aggregate tick/order data into daily cross-sectional factors.
It assumes a normalized trade/order table and should be adapted to the data
vendor's exact field definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def order_size_bucket(amount: pd.Series) -> pd.Series:
    """Classify order amount using common A-share money-flow thresholds."""
    bins = [-np.inf, 40_000, 200_000, 1_000_000, np.inf]
    labels = ["small", "medium", "large", "super_large"]
    return pd.cut(amount.astype(float), bins=bins, labels=labels, right=False).astype(str)


def session_bucket(time: pd.Series) -> pd.Series:
    """Classify intraday timestamp into open, close, or regular session."""
    parsed = pd.to_datetime(time, format="%H:%M:%S", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(time[parsed.isna()], errors="coerce"))
    t = parsed.dt.time
    open_cut = pd.to_datetime("10:00:00").time()
    close_cut = pd.to_datetime("14:30:00").time()
    out = pd.Series("regular", index=time.index, dtype="object")
    out[t < open_cut] = "open"
    out[t >= close_cut] = "close"
    return out


def signed_side(side: pd.Series, buy_values: tuple[str, ...] = ("buy", "B", "1"), sell_values: tuple[str, ...] = ("sell", "S", "-1")) -> pd.Series:
    s = side.astype(str)
    sign = pd.Series(0, index=side.index, dtype="int64")
    sign[s.isin(buy_values)] = 1
    sign[s.isin(sell_values)] = -1
    return sign


def safe_ratio(num: pd.Series | float, den: pd.Series | float) -> pd.Series | float:
    return num / np.where(np.abs(den) <= EPS, np.nan, den)


def _safe_float_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or abs(den) <= EPS:
        return float("nan")
    return float(num / den)


def aggregate_order_flow(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str,
    amount_col: str,
    order_amount_col: str | None = None,
    active_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate normalized trade/order rows into daily money-flow factors."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)
    out["side_sign"] = signed_side(out[side_col])
    out["order_amount"] = out[order_amount_col].astype(float) if order_amount_col else out["amount"]
    out["size_bucket"] = order_size_bucket(out["order_amount"])
    out["is_main"] = out["size_bucket"].isin(["large", "super_large"])
    if active_col:
        out["is_active"] = out[active_col].astype(bool)
    else:
        out["is_active"] = True
    if time_col:
        out["session"] = session_bucket(out[time_col])
    else:
        out["session"] = "regular"

    keys = [date_col, asset_col]
    base = out.groupby(keys)["amount"].sum().rename("total_amount")
    buy = out[out["side_sign"] > 0].groupby(keys)["amount"].sum().rename("buy_amount")
    sell = out[out["side_sign"] < 0].groupby(keys)["amount"].sum().rename("sell_amount")
    main_buy = out[(out["side_sign"] > 0) & out["is_main"]].groupby(keys)["amount"].sum().rename("main_buy_amount")
    main_sell = out[(out["side_sign"] < 0) & out["is_main"]].groupby(keys)["amount"].sum().rename("main_sell_amount")
    active_buy = out[(out["side_sign"] > 0) & out["is_active"]].groupby(keys)["amount"].sum().rename("active_buy_amount")
    active_sell = out[(out["side_sign"] < 0) & out["is_active"]].groupby(keys)["amount"].sum().rename("active_sell_amount")

    features = pd.concat([base, buy, sell, main_buy, main_sell, active_buy, active_sell], axis=1).fillna(0.0)
    features["main_net_inflow"] = features["main_buy_amount"] - features["main_sell_amount"]
    features["main_inflow_rate"] = safe_ratio(
        features["main_net_inflow"],
        features["main_buy_amount"] + features["main_sell_amount"],
    )
    features["active_net_buy"] = features["active_buy_amount"] - features["active_sell_amount"]
    features["active_net_buy_rate"] = safe_ratio(
        features["active_net_buy"],
        features["active_buy_amount"] + features["active_sell_amount"],
    )
    features["big_buy_ratio"] = safe_ratio(features["main_buy_amount"], features["total_amount"])
    features["big_sell_ratio"] = safe_ratio(features["main_sell_amount"], features["total_amount"])

    for sess in ["open", "close"]:
        sess_data = out[out["session"] == sess]
        sess_main_buy = sess_data[(sess_data["side_sign"] > 0) & sess_data["is_main"]].groupby(keys)["amount"].sum()
        sess_main_sell = sess_data[(sess_data["side_sign"] < 0) & sess_data["is_main"]].groupby(keys)["amount"].sum()
        features[f"{sess}_main_buy_amount"] = sess_main_buy
        features[f"{sess}_main_sell_amount"] = sess_main_sell
        features[[f"{sess}_main_buy_amount", f"{sess}_main_sell_amount"]] = features[
            [f"{sess}_main_buy_amount", f"{sess}_main_sell_amount"]
        ].fillna(0.0)
        net = features[f"{sess}_main_buy_amount"] - features[f"{sess}_main_sell_amount"]
        gross = features[f"{sess}_main_buy_amount"] + features[f"{sess}_main_sell_amount"]
        features[f"{sess}_main_inflow_rate"] = safe_ratio(net, gross)

    return features.reset_index()


def reconstruct_orders_from_trades(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    amount_col: str,
    buy_order_id_col: str,
    sell_order_id_col: str,
) -> pd.DataFrame:
    """Reconstruct buy/sell order amounts from tick trades.

    Tick rows are often individual executions. The buy and sell order ids allow
    these executions to be recombined into order-level amounts.
    """
    base_cols = [date_col, asset_col, amount_col, buy_order_id_col, sell_order_id_col]
    missing = [col for col in base_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    clean = df[base_cols].copy()
    clean[date_col] = pd.to_datetime(clean[date_col]).dt.normalize()
    clean[asset_col] = clean[asset_col].astype(str)
    clean[amount_col] = clean[amount_col].astype(float).clip(lower=0.0)

    buy_orders = clean[[date_col, asset_col, buy_order_id_col, amount_col]].rename(
        columns={buy_order_id_col: "order_id", amount_col: "execution_amount"}
    )
    buy_orders["order_side"] = "buy"
    sell_orders = clean[[date_col, asset_col, sell_order_id_col, amount_col]].rename(
        columns={sell_order_id_col: "order_id", amount_col: "execution_amount"}
    )
    sell_orders["order_side"] = "sell"
    orders = pd.concat([buy_orders, sell_orders], ignore_index=True)
    orders["order_id"] = orders["order_id"].astype(str)
    return (
        orders.groupby([date_col, asset_col, "order_side", "order_id"], as_index=False)["execution_amount"]
        .sum()
        .rename(columns={"execution_amount": "order_amount"})
    )


def mark_dynamic_big_orders(
    orders: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str = "order_side",
    amount_col: str = "order_amount",
    n_std: float = 1.0,
) -> pd.DataFrame:
    """Flag big orders using mean + N * std within each stock-day-side."""
    out = orders.copy()
    group_cols = [date_col, asset_col, side_col]
    grouped_amount = out.groupby(group_cols)[amount_col]
    amount_mean = grouped_amount.transform("mean")
    amount_std = grouped_amount.transform("std").fillna(0.0)
    out["big_order_threshold"] = amount_mean + n_std * amount_std
    out["is_big_order"] = out[amount_col] > out["big_order_threshold"]
    return out


def big_order_amount_features(
    orders: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str = "order_side",
    amount_col: str = "order_amount",
    n_std: float = 1.0,
) -> pd.DataFrame:
    """Compute big buy/sell amount ratios from reconstructed order data."""
    marked = mark_dynamic_big_orders(orders, date_col, asset_col, side_col, amount_col, n_std=n_std)
    keys = [date_col, asset_col]
    side_totals = marked.groupby([*keys, side_col])[amount_col].sum().unstack(side_col).fillna(0.0)
    total_amount = side_totals.mean(axis=1).rename("total_amount")
    big = marked[marked["is_big_order"]].groupby([*keys, side_col])[amount_col].sum().unstack(side_col).fillna(0.0)
    for side in ["buy", "sell"]:
        if side not in big.columns:
            big[side] = 0.0
    result = pd.concat([total_amount, big[["buy", "sell"]].rename(columns={"buy": "big_buy_amount", "sell": "big_sell_amount"})], axis=1)
    result["big_buy_amount_ratio"] = safe_ratio(result["big_buy_amount"], result["total_amount"])
    result["big_sell_amount_ratio"] = safe_ratio(result["big_sell_amount"], result["total_amount"])
    result["big_buy_minus_sell_ratio"] = result["big_buy_amount_ratio"] - result["big_sell_amount_ratio"]
    result["big_order_amount_ratio"] = result["big_buy_amount_ratio"] + result["big_sell_amount_ratio"]
    result["big_order_n_std"] = float(n_std)
    return result.reset_index()


def reconstructed_order_active_degrees(
    trades: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str,
    amount_col: str,
    buy_order_id_col: str,
    sell_order_id_col: str,
) -> pd.DataFrame:
    """Reconstruct order-level active trading degrees from tick trades.

    For a buy order, active amount is the execution amount whose tick side is
    active buy. For a sell order, active amount is the execution amount whose
    tick side is active sell. The active degree is active amount divided by the
    reconstructed order amount.
    """
    required = [date_col, asset_col, side_col, amount_col, buy_order_id_col, sell_order_id_col]
    missing = [col for col in required if col not in trades.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = trades[required].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out[amount_col] = out[amount_col].astype(float).clip(lower=0.0)
    out["_side_sign"] = signed_side(out[side_col])

    buy = out[[date_col, asset_col, buy_order_id_col, amount_col, "_side_sign"]].rename(
        columns={buy_order_id_col: "order_id"}
    )
    buy["order_side"] = "buy"
    buy["active_amount"] = np.where(buy["_side_sign"] > 0, buy[amount_col], 0.0)
    sell = out[[date_col, asset_col, sell_order_id_col, amount_col, "_side_sign"]].rename(
        columns={sell_order_id_col: "order_id"}
    )
    sell["order_side"] = "sell"
    sell["active_amount"] = np.where(sell["_side_sign"] < 0, sell[amount_col], 0.0)

    orders = pd.concat([buy, sell], ignore_index=True)
    orders["order_id"] = orders["order_id"].astype(str)
    grouped = (
        orders.groupby([date_col, asset_col, "order_side", "order_id"], as_index=False)
        .agg(order_amount=(amount_col, "sum"), active_amount=("active_amount", "sum"))
    )
    grouped["active_degree"] = safe_ratio(grouped["active_amount"], grouped["order_amount"])
    return grouped


def order_active_degree_features(
    trades: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str,
    amount_col: str,
    buy_order_id_col: str,
    sell_order_id_col: str,
    n_std: float = 1.0,
) -> pd.DataFrame:
    """Compute small/medium/large buy and sell order active degree factors.

    Buckets are computed within each stock-day-side:
    - small: order amount <= mean
    - medium: mean < amount <= mean + n_std * std
    - large: amount > mean + n_std * std
    """
    orders = reconstructed_order_active_degrees(
        trades,
        date_col=date_col,
        asset_col=asset_col,
        side_col=side_col,
        amount_col=amount_col,
        buy_order_id_col=buy_order_id_col,
        sell_order_id_col=sell_order_id_col,
    )
    group_cols = [date_col, asset_col, "order_side"]
    amount_mean = orders.groupby(group_cols)["order_amount"].transform("mean")
    amount_std = orders.groupby(group_cols)["order_amount"].transform("std").fillna(0.0)
    large_cut = amount_mean + n_std * amount_std

    orders["active_degree_bucket"] = "small"
    orders.loc[(orders["order_amount"] > amount_mean) & (orders["order_amount"] <= large_cut), "active_degree_bucket"] = "medium"
    orders.loc[orders["order_amount"] > large_cut, "active_degree_bucket"] = "large"

    keys = [date_col, asset_col]
    bucketed = (
        orders.groupby([*keys, "order_side", "active_degree_bucket"])
        .agg(order_amount=("order_amount", "sum"), active_amount=("active_amount", "sum"))
        .reset_index()
    )
    bucketed["degree"] = safe_ratio(bucketed["active_amount"], bucketed["order_amount"])
    wide = bucketed.pivot_table(
        index=keys,
        columns=["order_side", "active_degree_bucket"],
        values="degree",
        aggfunc="first",
    )
    wide.columns = [f"{side}_{bucket}_active_degree" for side, bucket in wide.columns]
    result = wide.reset_index()
    for side in ["buy", "sell"]:
        small_col = f"{side}_small_active_degree"
        large_col = f"{side}_large_active_degree"
        if small_col in result.columns and large_col in result.columns:
            result[f"{side}_small_minus_large_active_degree"] = result[small_col] - result[large_col]
    result["active_degree_n_std"] = float(n_std)
    return result


def order_concentration_features(
    orders: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str = "order_side",
    amount_col: str = "order_amount",
    log_transform: bool = True,
) -> pd.DataFrame:
    """Compute buy/sell order concentration from reconstructed order data."""
    out = orders.copy()
    keys = [date_col, asset_col]
    side_totals = out.groupby([*keys, side_col])[amount_col].sum().unstack(side_col).fillna(0.0)
    total_amount = side_totals.mean(axis=1)
    squared = out.assign(amount_sq=out[amount_col].astype(float) ** 2)
    side_sq = squared.groupby([*keys, side_col])["amount_sq"].sum().unstack(side_col).fillna(0.0)
    result = pd.DataFrame(index=total_amount.index)
    result["total_amount"] = total_amount
    result["buy_order_concentration"] = safe_ratio(side_sq.get("buy", 0.0), total_amount**2)
    result["sell_order_concentration"] = safe_ratio(side_sq.get("sell", 0.0), total_amount**2)
    result["buy_sell_concentration_diff"] = result["buy_order_concentration"] - result["sell_order_concentration"]
    result["buy_sell_concentration_sum"] = result["buy_order_concentration"] + result["sell_order_concentration"]
    if log_transform:
        for col in [
            "buy_order_concentration",
            "sell_order_concentration",
            "buy_sell_concentration_sum",
        ]:
            result[f"log_{col}"] = np.log(result[col].clip(lower=EPS))
    return result.reset_index()


def active_buy_session_features(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    side_col: str,
    amount_col: str,
    time_col: str,
    limit_flag_col: str | None = None,
    minute_freq: str = "1min",
) -> pd.DataFrame:
    """Build active-buy ratio and strength factors by intraday session.

    BS sign convention: B/1/buy = active buy, S/-1/sell = active sell. Limit-up
    and limit-down minutes can invert intuitive active direction, so provide
    `limit_flag_col` to exclude them when available.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)
    out["side_sign"] = signed_side(out[side_col])
    if limit_flag_col:
        out = out[~out[limit_flag_col].fillna(False).astype(bool)].copy()
    out["session"] = session_bucket(out[time_col])
    timestamps = pd.to_datetime(
        out[date_col].dt.strftime("%Y-%m-%d") + " " + out[time_col].astype(str),
        errors="coerce",
    )
    out["minute"] = timestamps.dt.floor(minute_freq)
    keys = [date_col, asset_col]
    day_total = out.groupby(keys)["amount"].sum()

    rows: list[pd.DataFrame] = []
    for session_name, session_data in [("full", out), ("open", out[out["session"] == "open"]), ("regular", out[out["session"] == "regular"]), ("close", out[out["session"] == "close"])]:
        if session_data.empty:
            continue
        minute = (
            session_data.groupby([date_col, asset_col, "minute", "side_sign"])["amount"]
            .sum()
            .unstack("side_sign")
            .fillna(0.0)
        )
        buy = minute.get(1, pd.Series(0.0, index=minute.index))
        sell = minute.get(-1, pd.Series(0.0, index=minute.index))
        minute_frame = pd.DataFrame({"active_buy_amount": buy, "active_sell_amount": sell})
        minute_frame["active_net_buy_amount"] = minute_frame["active_buy_amount"] - minute_frame["active_sell_amount"]
        session_total = session_data.groupby(keys)["amount"].sum()
        buy_total = session_data[session_data["side_sign"] > 0].groupby(keys)["amount"].sum()
        buy_total = buy_total.reindex(session_total.index).fillna(0.0)

        stat_rows = []
        for idx, group in minute_frame.groupby(level=[0, 1]):
            net_std = group["active_net_buy_amount"].std(ddof=1)
            buy_std = group["active_buy_amount"].std(ddof=1)
            stat_rows.append(
                {
                    date_col: idx[0],
                    asset_col: idx[1],
                    f"{session_name}_active_buy_intensity": _safe_float_ratio(group["active_buy_amount"].mean(), buy_std),
                    f"{session_name}_active_net_buy_intensity": _safe_float_ratio(group["active_net_buy_amount"].mean(), net_std),
                }
            )
        stats = pd.DataFrame(stat_rows).set_index(keys)
        stats[f"{session_name}_active_buy_amount"] = buy_total
        stats[f"{session_name}_total_amount"] = session_total
        stats[f"{session_name}_active_buy_ratio_session"] = safe_ratio(buy_total, session_total)
        stats[f"{session_name}_active_buy_ratio_day"] = safe_ratio(buy_total, day_total.reindex(session_total.index))
        rows.append(stats)

    if not rows:
        return pd.DataFrame(columns=[date_col, asset_col])
    result = pd.concat(rows, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result.reset_index()


def detailed_session_bucket(
    time: pd.Series,
    open_end: str = "10:00:00",
    close_start: str = "14:26:00",
    full_start: str = "09:30:00",
    full_end: str = "14:57:00",
) -> pd.Series:
    """Classify A-share continuous auction time into open/regular/close."""
    parsed = pd.to_datetime(time.astype(str), format="%H:%M:%S", errors="coerce")
    missing_time = parsed.isna()
    if missing_time.any():
        fallback = pd.to_datetime(time[missing_time].astype(str), format="mixed", errors="coerce")
        parsed.loc[missing_time] = fallback
    clock = parsed.dt.time
    valid = parsed.notna()
    open_end_t = pd.to_datetime(open_end).time()
    close_start_t = pd.to_datetime(close_start).time()
    full_start_t = pd.to_datetime(full_start).time()
    full_end_t = pd.to_datetime(full_end).time()

    out = pd.Series("out_of_session", index=time.index, dtype="object")
    in_full = valid & (clock >= full_start_t) & (clock <= full_end_t)
    out[in_full & (clock < open_end_t)] = "open"
    out[in_full & (clock >= open_end_t) & (clock < close_start_t)] = "regular"
    out[in_full & (clock >= close_start_t)] = "close"
    return out


def minute_trade_size_features(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    amount_col: str,
    trade_count_col: str,
    return_col: str,
    top_fraction: float = 0.2,
) -> pd.DataFrame:
    """Build minute-trade features from amount, trade count and minute return.

    The report-style big-trade-minute sample is selected by each stock-day's
    largest average amount per trade minutes.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    out = df[[date_col, asset_col, amount_col, trade_count_col, return_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)
    out["trade_count"] = out[trade_count_col].astype(float).clip(lower=0.0)
    out["ret"] = out[return_col].astype(float)
    out["amount_per_trade"] = safe_ratio(out["amount"], out["trade_count"])
    keys = [date_col, asset_col]

    total_amount = out.groupby(keys)["amount"].sum().rename("total_amount")
    total_trades = out.groupby(keys)["trade_count"].sum().rename("total_trade_count")
    result = pd.concat([total_amount, total_trades], axis=1)
    result["avg_amount_per_trade"] = safe_ratio(result["total_amount"], result["total_trade_count"])

    inflow = out[out["ret"] > 0].groupby(keys).agg({"amount": "sum", "trade_count": "sum"})
    outflow = out[out["ret"] < 0].groupby(keys).agg({"amount": "sum", "trade_count": "sum"})
    result["inflow_avg_amount_per_trade"] = safe_ratio(
        inflow["amount"].reindex(result.index).fillna(0.0),
        inflow["trade_count"].reindex(result.index).fillna(0.0),
    )
    result["outflow_avg_amount_per_trade"] = safe_ratio(
        outflow["amount"].reindex(result.index).fillna(0.0),
        outflow["trade_count"].reindex(result.index).fillna(0.0),
    )
    result["inflow_avg_amount_ratio"] = safe_ratio(result["inflow_avg_amount_per_trade"], result["avg_amount_per_trade"])
    result["outflow_avg_amount_ratio"] = safe_ratio(result["outflow_avg_amount_per_trade"], result["avg_amount_per_trade"])
    result["inflow_outflow_avg_amount_ratio"] = safe_ratio(
        result["inflow_avg_amount_ratio"],
        result["outflow_avg_amount_ratio"],
    )

    rank_desc = out.groupby(keys)["amount_per_trade"].rank(method="first", ascending=False)
    valid_count = out.groupby(keys)["amount_per_trade"].transform(lambda s: s.notna().sum())
    top_n = np.ceil(valid_count * top_fraction).clip(lower=1)
    out["is_big_trade_minute"] = rank_desc <= top_n
    big = out[out["is_big_trade_minute"]].copy()
    big_pos = big[big["ret"] > 0].groupby(keys)["amount"].sum()
    big_neg = big[big["ret"] < 0].groupby(keys)["amount"].sum()
    result["big_minute_net_inflow"] = big_pos.reindex(result.index).fillna(0.0) - big_neg.reindex(result.index).fillna(0.0)
    result["big_minute_net_inflow_ratio"] = safe_ratio(result["big_minute_net_inflow"], result["total_amount"])
    result["big_minute_driven_return"] = big.groupby(keys)["ret"].apply(lambda s: float(np.prod(1.0 + s) - 1.0))
    result["big_minute_top_fraction"] = float(top_fraction)
    return result.reset_index()


def add_net_bid_change_rate(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    bid_volume_cols: list[str],
    ask_volume_cols: list[str],
    float_share_col: str,
    output_col: str = "net_bid_change_rate",
) -> pd.DataFrame:
    """Add net bid change rate from order-book snapshot volume changes."""
    required = [date_col, asset_col, time_col, float_share_col, *bid_volume_cols, *ask_volume_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    timestamp = pd.to_datetime(out[date_col].dt.strftime("%Y-%m-%d") + " " + out[time_col].astype(str), errors="coerce")
    out["_snapshot_time"] = timestamp
    out = out.sort_values([asset_col, date_col, "_snapshot_time"]).copy()
    out["_bid_volume_sum"] = out[bid_volume_cols].astype(float).sum(axis=1)
    out["_ask_volume_sum"] = out[ask_volume_cols].astype(float).sum(axis=1)
    keys = [asset_col, date_col]
    bid_change = out.groupby(keys)["_bid_volume_sum"].diff()
    ask_change = out.groupby(keys)["_ask_volume_sum"].diff()
    out[output_col] = safe_ratio(bid_change - ask_change, out[float_share_col].astype(float))
    return out.drop(columns=["_snapshot_time", "_bid_volume_sum", "_ask_volume_sum"])


def net_bid_change_rate_features(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    bid_volume_cols: list[str],
    ask_volume_cols: list[str],
    float_share_col: str,
    output_col: str = "net_bid_change_rate",
) -> pd.DataFrame:
    """Aggregate net bid change rate into mean, volatility and skew factors."""
    out = add_net_bid_change_rate(
        df,
        date_col=date_col,
        asset_col=asset_col,
        time_col=time_col,
        bid_volume_cols=bid_volume_cols,
        ask_volume_cols=ask_volume_cols,
        float_share_col=float_share_col,
        output_col=output_col,
    )
    out["session_detail"] = detailed_session_bucket(out[time_col])
    keys = [date_col, asset_col]
    frames: list[pd.DataFrame] = []
    for session_name, session_data in [
        ("full", out[out["session_detail"].isin(["open", "regular", "close"])]),
        ("open", out[out["session_detail"] == "open"]),
        ("regular", out[out["session_detail"] == "regular"]),
        ("close", out[out["session_detail"] == "close"]),
    ]:
        stats = session_data.groupby(keys)[output_col].agg(["mean", "std", "skew"])
        stats = stats.rename(
            columns={
                "mean": f"{session_name}_net_bid_change_mean",
                "std": f"{session_name}_net_bid_change_vol",
                "skew": f"{session_name}_net_bid_change_skew",
            }
        )
        frames.append(stats)
    result = pd.concat(frames, axis=1)
    return result.reset_index()


def order_trade_correlation_features(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    return_col: str,
    net_bid_rate_col: str = "net_bid_change_rate",
) -> pd.DataFrame:
    """Compute order-trade correlation between return and net bid changes."""
    required = [date_col, asset_col, time_col, return_col, net_bid_rate_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df[required].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["ret"] = out[return_col].astype(float)
    out["net_bid_rate"] = out[net_bid_rate_col].astype(float)
    out["session_detail"] = detailed_session_bucket(out[time_col])
    keys = [date_col, asset_col]

    frames: list[pd.DataFrame] = []
    for session_name, session_data in [
        ("full", out[out["session_detail"].isin(["open", "regular", "close"])]),
        ("open", out[out["session_detail"] == "open"]),
        ("regular", out[out["session_detail"] == "regular"]),
        ("close", out[out["session_detail"] == "close"]),
    ]:
        corr = session_data.groupby(keys).apply(
            lambda g: g["ret"].corr(g["net_bid_rate"]) if g[["ret", "net_bid_rate"]].dropna().shape[0] >= 3 else np.nan
        )
        mean_net_bid = session_data.groupby(keys)["net_bid_rate"].mean()
        stats = pd.concat(
            [
                corr.rename(f"{session_name}_order_trade_corr"),
                mean_net_bid.rename(f"{session_name}_net_bid_change_mean_for_shape"),
            ],
            axis=1,
        )
        frames.append(stats)
    result = pd.concat(frames, axis=1)
    return result.reset_index()


def classify_order_trade_shapes(
    features: pd.DataFrame,
    date_col: str,
    mean_col: str,
    corr_col: str,
    low_q: float = 0.2,
    high_q: float = 0.8,
    output_col: str = "order_trade_shape",
) -> pd.DataFrame:
    """Classify price/commission shapes from cross-sectional mean and corr."""
    out = features.copy()
    low_mean = out.groupby(date_col)[mean_col].transform(lambda s: s.quantile(low_q))
    high_mean = out.groupby(date_col)[mean_col].transform(lambda s: s.quantile(high_q))
    low_corr = out.groupby(date_col)[corr_col].transform(lambda s: s.quantile(low_q))
    high_corr = out.groupby(date_col)[corr_col].transform(lambda s: s.quantile(high_q))
    out[output_col] = "middle"
    out.loc[(out[mean_col] >= high_mean) & (out[corr_col] <= low_corr), output_col] = "price_down_net_bid_up_support"
    out.loc[(out[mean_col] <= low_mean) & (out[corr_col] >= high_corr), output_col] = "price_down_net_bid_down_weak_support"
    out.loc[(out[mean_col] <= low_mean) & (out[corr_col] <= low_corr), output_col] = "price_up_net_bid_down"
    out.loc[(out[mean_col] >= high_mean) & (out[corr_col] >= high_corr), output_col] = "price_up_net_bid_up"
    return out


def rolling_mean_by_asset(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    value_cols: list[str],
    window: int = 20,
    min_periods: int | None = None,
    include_current: bool = False,
) -> pd.DataFrame:
    """Create rolling mean factor values by asset without look-ahead by default."""
    out = df.sort_values([asset_col, date_col]).copy()
    min_periods = min_periods or window
    for col in value_cols:
        source = out[col] if include_current else out.groupby(asset_col)[col].shift(1)
        out[f"{col}_mean_{window}d"] = source.groupby(out[asset_col]).transform(
            lambda s: s.rolling(window, min_periods=min_periods).mean()
        )
    return out


def rolling_unexpected_minute_returns(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    return_col: str,
    lookback_days: int = 20,
    min_obs: int = 30,
    output_col: str = "unexpected_return",
) -> pd.DataFrame:
    """Estimate minute return residuals using past-day weekday/session/lag controls."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["_model_ret"] = out[return_col].astype(float)
    out["session_detail"] = detailed_session_bucket(out[time_col])
    out["_weekday"] = out[date_col].dt.weekday
    out["_lag_ret"] = out.sort_values([asset_col, date_col, time_col]).groupby([asset_col, date_col])["_model_ret"].shift(1)

    weekday_dummies = pd.get_dummies(out["_weekday"]).reindex(columns=[0, 1, 2, 3], fill_value=0)
    weekday_dummies.columns = [f"weekday_{col}" for col in weekday_dummies.columns]
    session_dummies = pd.get_dummies(out["session_detail"]).reindex(columns=["open", "regular", "close"], fill_value=0)
    session_dummies.columns = [f"session_{col}" for col in session_dummies.columns]
    feature_cols = ["_lag_ret", *weekday_dummies.columns.tolist(), *session_dummies.columns.tolist()]
    model_frame = pd.concat([out[[date_col, asset_col, "_model_ret", "_lag_ret"]], weekday_dummies, session_dummies], axis=1)
    model_frame[feature_cols] = model_frame[feature_cols].fillna(0.0).astype(float)
    out[output_col] = np.nan

    for asset, asset_frame in model_frame.groupby(asset_col, sort=False):
        dates = sorted(asset_frame[date_col].dropna().unique())
        for pos, current_date in enumerate(dates):
            train_dates = dates[max(0, pos - lookback_days) : pos]
            if not train_dates:
                continue
            train = asset_frame[(asset_frame[date_col].isin(train_dates)) & asset_frame["_model_ret"].notna()]
            current = asset_frame[(asset_frame[date_col] == current_date) & asset_frame["_model_ret"].notna()]
            if train.shape[0] < min_obs or current.empty:
                continue
            x_train = np.column_stack([np.ones(train.shape[0]), train[feature_cols].to_numpy(dtype=float)])
            y_train = train["_model_ret"].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
            x_current = np.column_stack([np.ones(current.shape[0]), current[feature_cols].to_numpy(dtype=float)])
            out.loc[current.index, output_col] = current["_model_ret"].to_numpy(dtype=float) - x_current @ beta
    return out.drop(columns=["_model_ret", "session_detail", "_weekday", "_lag_ret"], errors="ignore")


def informed_trade_features(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    unexpected_return_col: str,
    active_buy_amount_col: str,
    active_sell_amount_col: str,
    amount_col: str,
) -> pd.DataFrame:
    """Build informed buy/sell ratios from unexpected returns and active trades."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["unexpected_return"] = out[unexpected_return_col].astype(float)
    out["active_buy_amount"] = out[active_buy_amount_col].astype(float).clip(lower=0.0)
    out["active_sell_amount"] = out[active_sell_amount_col].astype(float).clip(lower=0.0)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)
    out["session_detail"] = detailed_session_bucket(out[time_col])
    out["informed_buy_amount"] = np.where(out["unexpected_return"] < 0, out["active_buy_amount"], 0.0)
    out["informed_sell_amount"] = np.where(out["unexpected_return"] > 0, out["active_sell_amount"], 0.0)
    keys = [date_col, asset_col]
    day_total = out.groupby(keys)["amount"].sum()
    rows: list[pd.DataFrame] = []

    for session_name, session_data in [
        ("full", out[out["session_detail"].isin(["open", "regular", "close"])]),
        ("open", out[out["session_detail"] == "open"]),
        ("regular", out[out["session_detail"] == "regular"]),
        ("close", out[out["session_detail"] == "close"]),
    ]:
        session_total = session_data.groupby(keys)["amount"].sum()
        active_buy_total = session_data.groupby(keys)["active_buy_amount"].sum()
        active_sell_total = session_data.groupby(keys)["active_sell_amount"].sum()
        informed_buy = session_data.groupby(keys)["informed_buy_amount"].sum()
        informed_sell = session_data.groupby(keys)["informed_sell_amount"].sum()
        stats = pd.DataFrame(index=session_total.index)
        stats[f"{session_name}_informed_buy_amount"] = informed_buy.reindex(session_total.index).fillna(0.0)
        stats[f"{session_name}_informed_sell_amount"] = informed_sell.reindex(session_total.index).fillna(0.0)
        stats[f"{session_name}_informed_net_buy_amount"] = (
            stats[f"{session_name}_informed_buy_amount"] - stats[f"{session_name}_informed_sell_amount"]
        )
        stats[f"{session_name}_informed_buy_ratio_day"] = safe_ratio(
            stats[f"{session_name}_informed_buy_amount"],
            day_total.reindex(session_total.index),
        )
        stats[f"{session_name}_informed_sell_ratio_day"] = safe_ratio(
            stats[f"{session_name}_informed_sell_amount"],
            day_total.reindex(session_total.index),
        )
        stats[f"{session_name}_informed_net_buy_ratio_day"] = safe_ratio(
            stats[f"{session_name}_informed_net_buy_amount"],
            day_total.reindex(session_total.index),
        )
        stats[f"{session_name}_informed_buy_ratio_session"] = safe_ratio(
            stats[f"{session_name}_informed_buy_amount"],
            session_total,
        )
        stats[f"{session_name}_informed_sell_ratio_session"] = safe_ratio(
            stats[f"{session_name}_informed_sell_amount"],
            session_total,
        )
        stats[f"{session_name}_informed_buy_ratio_active_buy"] = safe_ratio(
            stats[f"{session_name}_informed_buy_amount"],
            active_buy_total.reindex(session_total.index),
        )
        stats[f"{session_name}_informed_sell_ratio_active_sell"] = safe_ratio(
            stats[f"{session_name}_informed_sell_amount"],
            active_sell_total.reindex(session_total.index),
        )
        rows.append(stats)

    result = pd.concat(rows, axis=1)
    return result.reset_index()


def big_order_participation_features(
    trades: pd.DataFrame,
    date_col: str,
    asset_col: str,
    amount_col: str,
    buy_order_id_col: str,
    sell_order_id_col: str,
    n_std: float = 0.0,
) -> pd.DataFrame:
    """Split trade amount by whether buy and/or sell orders are dynamically big."""
    orders = reconstruct_orders_from_trades(
        trades,
        date_col=date_col,
        asset_col=asset_col,
        amount_col=amount_col,
        buy_order_id_col=buy_order_id_col,
        sell_order_id_col=sell_order_id_col,
    )
    marked = mark_dynamic_big_orders(orders, date_col, asset_col, n_std=n_std)
    out = trades[[date_col, asset_col, amount_col, buy_order_id_col, sell_order_id_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)

    buy_flags = (
        marked[marked["order_side"] == "buy"][[date_col, asset_col, "order_id", "is_big_order"]]
        .rename(columns={"order_id": buy_order_id_col, "is_big_order": "buy_is_big"})
    )
    sell_flags = (
        marked[marked["order_side"] == "sell"][[date_col, asset_col, "order_id", "is_big_order"]]
        .rename(columns={"order_id": sell_order_id_col, "is_big_order": "sell_is_big"})
    )
    out[buy_order_id_col] = out[buy_order_id_col].astype(str)
    out[sell_order_id_col] = out[sell_order_id_col].astype(str)
    out = out.merge(buy_flags, on=[date_col, asset_col, buy_order_id_col], how="left")
    out = out.merge(sell_flags, on=[date_col, asset_col, sell_order_id_col], how="left")
    out[["buy_is_big", "sell_is_big"]] = out[["buy_is_big", "sell_is_big"]].fillna(False).astype(bool)

    keys = [date_col, asset_col]
    total = out.groupby(keys)["amount"].sum().rename("total_amount")
    result = total.to_frame()
    masks = {
        "big_buy_amount": out["buy_is_big"],
        "big_sell_amount": out["sell_is_big"],
        "big_buy_and_big_sell_amount": out["buy_is_big"] & out["sell_is_big"],
        "big_buy_without_big_sell_amount": out["buy_is_big"] & ~out["sell_is_big"],
        "big_sell_without_big_buy_amount": ~out["buy_is_big"] & out["sell_is_big"],
        "any_big_order_amount": out["buy_is_big"] | out["sell_is_big"],
    }
    for name, mask in masks.items():
        amount = out[mask].groupby(keys)["amount"].sum()
        result[name] = amount.reindex(result.index).fillna(0.0)
        result[f"{name}_ratio"] = safe_ratio(result[name], result["total_amount"])
    result["big_order_n_std"] = float(n_std)
    return result.reset_index()


def filtered_minute_bars(
    trades: pd.DataFrame,
    date_col: str,
    asset_col: str,
    time_col: str,
    price_col: str,
    amount_col: str,
    volume_col: str | None = None,
    keep_col: str | None = None,
    minute_freq: str = "1min",
) -> pd.DataFrame:
    """Reconstruct minute bars after optional tick-level filtering."""
    out = trades.copy()
    if keep_col:
        out = out[out[keep_col].fillna(False).astype(bool)].copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[asset_col] = out[asset_col].astype(str)
    out["price"] = out[price_col].astype(float)
    out["amount"] = out[amount_col].astype(float).clip(lower=0.0)
    out["volume"] = out[volume_col].astype(float).clip(lower=0.0) if volume_col else np.nan
    timestamp = pd.to_datetime(out[date_col].dt.strftime("%Y-%m-%d") + " " + out[time_col].astype(str), errors="coerce")
    out["minute"] = timestamp.dt.floor(minute_freq)
    agg = {
        "price": ["first", "max", "min", "last"],
        "amount": "sum",
    }
    if volume_col:
        agg["volume"] = "sum"
    bars = out.groupby([date_col, asset_col, "minute"]).agg(agg)
    bars.columns = ["open", "high", "low", "close", "amount", *(["volume"] if volume_col else [])]
    bars["trade_count"] = out.groupby([date_col, asset_col, "minute"]).size()
    bars["minute_return"] = bars.groupby(level=[0, 1])["close"].pct_change()
    return bars.reset_index()


def remove_recent_abnormal_events(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    event_col: str,
    lookback: int = 10,
) -> pd.DataFrame:
    """Drop rows whose asset had an abnormal-trading event within lookback rows."""
    out = df.sort_values([asset_col, date_col]).copy()
    event = out[event_col].fillna(False).astype(bool)
    recent = event.groupby(out[asset_col]).transform(lambda s: s.shift(1).rolling(lookback, min_periods=1).max())
    return out[~recent.fillna(False).astype(bool)].copy()


def residualize_by_date(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    control_cols: list[str],
    output_col: str | None = None,
) -> pd.DataFrame:
    """Cross-sectionally residualize target against controls by date."""
    out = df.copy()
    output_col = output_col or f"{target_col}_resid"
    out[output_col] = np.nan
    for date, group in out.groupby(date_col):
        clean = group[[target_col, *control_cols]].replace([np.inf, -np.inf], np.nan).dropna()
        if clean.shape[0] <= len(control_cols) + 1:
            continue
        y = clean[target_col].to_numpy(dtype=float)
        x = clean[control_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(x.shape[0]), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        out.loc[clean.index, output_col] = y - x @ beta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--side-col", required=True)
    parser.add_argument("--amount-col", required=True)
    parser.add_argument("--order-amount-col")
    parser.add_argument("--active-col")
    parser.add_argument("--time-col")
    parser.add_argument("--output-dir", default="order_flow_factor_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    features = aggregate_order_flow(
        df,
        date_col=args.date_col,
        asset_col=args.asset_col,
        side_col=args.side_col,
        amount_col=args.amount_col,
        order_amount_col=args.order_amount_col,
        active_col=args.active_col,
        time_col=args.time_col,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "order_flow_factors.csv", index=False, encoding="utf-8-sig")
    print(features.head())
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
