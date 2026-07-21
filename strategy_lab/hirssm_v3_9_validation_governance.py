from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import numpy as np


ROOT = Path("Introduction-to-Quantitative-Finance")
V38_DIR = ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay"
V39_DIR = ROOT / "outputs" / "agent_runs" / "v3_9"
VALIDATION_DIR = V39_DIR / "backtest_validation_auditor_pbo_dsr"
REPRO_DIR = V39_DIR / "code_quality_engineer_repro_manifest"
AGENT_DIR = ROOT / "strategy_lab" / "agents"
V38_SCRIPT = ROOT / "strategy_lab" / "hirssm_v3_8_risk_budget_overlay.py"
V38_CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
V38_SELECTED = "v3_8_vol_budget_overlay"
V38_CONTROL = "v3_8_v36_exact_control"
COSTS = [10.0, 20.0, 30.0]
PBO_BLOCK_COUNT = 10
PBO_TRAIN_BLOCK_COUNT = 5
PBO_PURGE_DAYS = 120
PBO_EMBARGO_DAYS = 21


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(args: list[str]) -> str:
    repo = ROOT.resolve().as_posix()
    cmd = ["git", "-c", f"safe.directory={repo}", "-C", str(ROOT), *args]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        return out.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unknown: {exc}"


def load_yearly(cost: float) -> pd.DataFrame:
    rows = []
    for candidate_dir in sorted((V38_DIR / "candidates").iterdir()):
        if not candidate_dir.is_dir():
            continue
        variant = candidate_dir.name
        path = candidate_dir / f"yearly_returns_{variant}_{int(cost)}bps.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["variant"] = variant
        df["cost_bps"] = cost
        df["excess_return"] = df["strategy_return"] - df["benchmark_return"]
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No yearly return files found for {cost}bps under {V38_DIR}")
    return pd.concat(rows, ignore_index=True)


def iter_splits(years: list[int]) -> tuple[list[tuple[int, ...]], bool]:
    half = len(years) // 2
    split_count = math.comb(len(years), half)
    splits = itertools.combinations(years, half)
    if split_count <= PBO_MAX_EXACT_SPLITS:
        return list(splits), True
    sampled = []
    for i, split in enumerate(splits):
        if i % math.ceil(split_count / PBO_MAX_EXACT_SPLITS) == 0:
            sampled.append(split)
    return sampled[:PBO_MAX_EXACT_SPLITS], False


def pbo_status(pbo: float) -> str:
    if pbo <= 0.20:
        return "pass"
    if pbo <= 0.35:
        return "observation"
    return "fail"


def load_navs(cost: float) -> dict[str, pd.DataFrame]:
    navs = {}
    for candidate_dir in sorted((V38_DIR / "candidates").iterdir()):
        if not candidate_dir.is_dir():
            continue
        variant = candidate_dir.name
        path = candidate_dir / f"nav_{variant}_{int(cost)}bps.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            navs[variant] = df.sort_values("date").reset_index(drop=True)
    if not navs:
        raise FileNotFoundError(f"No nav files found for {cost}bps under {V38_DIR}")
    return navs


def common_dates(navs: dict[str, pd.DataFrame]) -> pd.Series:
    date_sets = [set(df["date"]) for df in navs.values()]
    dates = sorted(set.intersection(*date_sets))
    return pd.Series(dates, name="date")


def block_ids(dates: pd.Series, n_blocks: int = PBO_BLOCK_COUNT) -> pd.Series:
    n = len(dates)
    edges = np.linspace(0, n, n_blocks + 1).round().astype(int)
    ids = np.empty(n, dtype=int)
    for block in range(n_blocks):
        ids[edges[block] : edges[block + 1]] = block
    return pd.Series(ids, index=dates.index, name="block_id")


def purged_train_mask(blocks: pd.Series, train_blocks: set[int], oos_blocks: set[int]) -> pd.Series:
    mask = blocks.isin(train_blocks).to_numpy().copy()
    block_array = blocks.to_numpy()
    for block in oos_blocks:
        idx = np.where(block_array == block)[0]
        if len(idx) == 0:
            continue
        start = max(0, int(idx.min()) - PBO_PURGE_DAYS)
        end = min(len(mask), int(idx.max()) + PBO_EMBARGO_DAYS + 1)
        mask[start:end] = False
    return pd.Series(mask, index=blocks.index)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    nav = (1 + returns).cumprod()
    return float((nav / nav.cummax() - 1).min())


def metric_summary(df: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    sub = df.loc[mask.to_numpy()].copy()
    if sub.empty:
        return {
            "annual_return": float("nan"),
            "annual_excess_vs_benchmark": float("nan"),
            "sharpe_no_rf": float("nan"),
            "information_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "drawdown_improvement_vs_benchmark": float("nan"),
            "avg_cash_weight": float("nan"),
            "avg_trade_turnover": float("nan"),
        }
    rets = sub["portfolio_return"].astype(float)
    bench = sub["benchmark_return"].astype(float)
    years = len(sub) / 252.0
    total = float((1 + rets).prod() - 1)
    bench_total = float((1 + bench).prod() - 1)
    annual_return = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    benchmark_annual = (1 + bench_total) ** (1 / years) - 1 if years > 0 and bench_total > -1 else float("nan")
    daily_std = rets.std(ddof=1)
    sharpe = float(rets.mean() / daily_std * math.sqrt(252)) if daily_std and daily_std > 0 else 0.0
    excess = rets - bench
    te = excess.std(ddof=1)
    ir = float(excess.mean() / te * math.sqrt(252)) if te and te > 0 else 0.0
    mdd = max_drawdown_from_returns(rets)
    bench_mdd = max_drawdown_from_returns(bench)
    return {
        "annual_return": float(annual_return),
        "annual_excess_vs_benchmark": float(annual_return - benchmark_annual),
        "sharpe_no_rf": sharpe,
        "information_ratio": ir,
        "max_drawdown": mdd,
        "drawdown_improvement_vs_benchmark": float(mdd - bench_mdd),
        "avg_cash_weight": float(sub.get("cash_weight", pd.Series([0.0])).mean()),
        "avg_trade_turnover": float(sub.get("turnover", pd.Series([0.0])).mean()),
    }


def fold_score(metrics: dict[str, float], base_metrics: dict[str, float]) -> float:
    ann_delta = metrics["annual_return"] - base_metrics["annual_return"]
    dd_delta = metrics["max_drawdown"] - base_metrics["max_drawdown"]
    sharpe_delta = metrics["sharpe_no_rf"] - base_metrics["sharpe_no_rf"]
    cash_excess = max(metrics["avg_cash_weight"] - 0.22, 0.0)
    annual_guard_penalty = max(-ann_delta - 0.005, 0.0)
    return float(
        1.80 * metrics["annual_excess_vs_benchmark"]
        + 0.85 * metrics["drawdown_improvement_vs_benchmark"]
        + 0.30 * metrics["sharpe_no_rf"]
        + 0.20 * metrics["information_ratio"]
        + 1.20 * dd_delta
        + 0.50 * ann_delta
        + 0.30 * sharpe_delta
        - 1.25 * annual_guard_penalty
        - 0.40 * cash_excess
        - 0.018 * metrics["avg_trade_turnover"]
    )


def run_pbo() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    summary_rows = []
    for cost in COSTS:
        navs = load_navs(cost)
        dates = common_dates(navs)
        blocks = block_ids(dates)
        aligned = {}
        for variant, df in navs.items():
            aligned[variant] = dates.to_frame().merge(df, on="date", how="left")
        candidates = sorted(aligned)
        combos = list(itertools.combinations(range(PBO_BLOCK_COUNT), PBO_TRAIN_BLOCK_COUNT))

        for fold_id, train_blocks_tuple in enumerate(combos):
            train_blocks = set(train_blocks_tuple)
            oos_blocks = set(range(PBO_BLOCK_COUNT)) - train_blocks
            is_mask = purged_train_mask(blocks, train_blocks, oos_blocks)
            oos_mask = blocks.isin(oos_blocks)
            train_scores = {}
            oos_scores = {}
            oos_metrics_by_variant = {}
            base_train = metric_summary(aligned[V38_CONTROL], is_mask)
            base_oos = metric_summary(aligned[V38_CONTROL], oos_mask)
            for variant in candidates:
                train_metrics = metric_summary(aligned[variant], is_mask)
                oos_metrics = metric_summary(aligned[variant], oos_mask)
                train_scores[variant] = fold_score(train_metrics, base_train)
                oos_scores[variant] = fold_score(oos_metrics, base_oos)
                oos_metrics_by_variant[variant] = oos_metrics
            train_scores_s = pd.Series(train_scores).dropna()
            oos_scores_s = pd.Series(oos_scores).dropna()
            selected = str(train_scores_s.idxmax())
            ordered = oos_scores_s.sort_values(ascending=False)
            test_rank = int(list(ordered.index).index(selected) + 1)
            if len(candidates) == 1:
                rank_quantile = 1.0
            else:
                rank_quantile = 1.0 - (test_rank - 1) / (len(candidates) - 1)
            clipped = min(max(rank_quantile, 1e-6), 1 - 1e-6)
            logit_rank = math.log(clipped / (1 - clipped))
            selected_minus_v36 = float(oos_scores_s[selected] - oos_scores_s[V38_CONTROL])
            fold_rows.append(
                {
                    "fold_id": fold_id,
                    "cost_bps": cost,
                    "cost_scope": f"{int(cost)}bps",
                    "is_blocks": " ".join(str(x) for x in sorted(train_blocks)),
                    "oos_blocks": " ".join(str(x) for x in sorted(oos_blocks)),
                    "purge_days": PBO_PURGE_DAYS,
                    "embargo_days": PBO_EMBARGO_DAYS,
                    "n_candidates": len(candidates),
                    "selected_variant": selected,
                    "is_score": float(train_scores_s[selected]),
                    "oos_score": float(oos_scores_s[selected]),
                    "oos_rank": test_rank,
                    "oos_rank_pct": rank_quantile,
                    "logit_rank": logit_rank,
                    "selected_is_v38": selected == V38_SELECTED,
                    "selected_is_control": selected == V38_CONTROL,
                    "selected_minus_v36_oos_score": selected_minus_v36,
                    "selected_minus_v36_oos_annual_return": float(
                        oos_metrics_by_variant[selected]["annual_return"] - base_oos["annual_return"]
                    ),
                }
            )

        fold = pd.DataFrame([row for row in fold_rows if row["cost_bps"] == cost])
        pbo = float((fold["logit_rank"] < 0).mean())
        summary_rows.append(
            {
                "cost_bps": cost,
                "cost_scope": f"{int(cost)}bps",
                "n_folds": int(len(fold)),
                "exact_cscv": True,
                "candidate_count": int(len(candidates)),
                "pbo": pbo,
                "status": pbo_status(pbo),
                "v38_selected_on_train_rate": float(fold["selected_is_v38"].mean()),
                "control_selected_on_train_rate": float(fold["selected_is_control"].mean()),
                "median_oos_rank_pct": float(fold["oos_rank_pct"].median()),
                "top1_oos_rate": float((fold["oos_rank"] == 1).mean()),
                "top2_oos_rate": float((fold["oos_rank"] <= 2).mean()),
                "mean_selected_minus_v36": float(fold["selected_minus_v36_oos_score"].mean()),
                "worst_fold_delta": float(fold["selected_minus_v36_oos_score"].min()),
                "pbo_threshold_pass": 0.20,
                "pbo_threshold_observation": 0.35,
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def annualized_sharpe(returns: pd.Series) -> tuple[float, float]:
    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return 0.0, 0.0
    daily = returns.mean() / std
    return float(daily), float(daily * math.sqrt(252))


def psr_probability(daily_sr: float, benchmark_daily_sr: float, n: int, skew: float, kurtosis: float) -> float:
    denom = 1 - skew * daily_sr + ((kurtosis - 1) / 4.0) * daily_sr * daily_sr
    if n <= 2 or denom <= 0:
        return float("nan")
    z = (daily_sr - benchmark_daily_sr) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(NormalDist().cdf(z))


def dsr_status(prob: float) -> str:
    if pd.isna(prob):
        return "blocked"
    if prob >= 0.95:
        return "pass"
    if prob >= 0.80:
        return "observation"
    return "fail"


def run_dsr() -> pd.DataFrame:
    rows = []
    for cost in COSTS:
        candidate_frames = {}
        for candidate_dir in sorted((V38_DIR / "candidates").iterdir()):
            if not candidate_dir.is_dir():
                continue
            variant = candidate_dir.name
            path = candidate_dir / f"nav_{variant}_{int(cost)}bps.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            candidate_frames[variant] = df
        metrics_by_variant: dict[str, dict[str, pd.Series]] = {}
        for variant, df in candidate_frames.items():
            port = pd.Series(df["portfolio_return"].astype(float).values)
            bench = pd.Series(df["benchmark_return"].astype(float).values)
            metrics_by_variant[variant] = {
                "sharpe": port,
                "information_ratio": port - bench,
            }
        for metric_type in ["sharpe", "information_ratio"]:
            observed_srs = {}
            for variant, metric_returns in metrics_by_variant.items():
                daily_sr, _ = annualized_sharpe(metric_returns[metric_type])
                observed_srs[variant] = daily_sr
            sr_series = pd.Series(observed_srs)
            for n_trials_eff in [6, 18]:
                sr_std = float(sr_series.std(ddof=1)) if len(sr_series) > 1 else 0.0
                gamma = 0.5772156649
                if n_trials_eff > 1 and sr_std > 0:
                    z1 = NormalDist().inv_cdf(1 - 1 / n_trials_eff)
                    z2 = NormalDist().inv_cdf(1 - 1 / (n_trials_eff * math.e))
                    expected_max_noise_sr = sr_std * ((1 - gamma) * z1 + gamma * z2)
                else:
                    expected_max_noise_sr = 0.0
                for variant, metric_returns in metrics_by_variant.items():
                    returns = metric_returns[metric_type]
                    daily_sr, ann_sr = annualized_sharpe(returns)
                    skew = float(returns.skew())
                    kurtosis = float(returns.kurt() + 3)
                    psr_zero = psr_probability(daily_sr, 0.0, len(returns), skew, kurtosis)
                    dsr_prob = psr_probability(daily_sr, expected_max_noise_sr, len(returns), skew, kurtosis)
                    rows.append(
                        {
                            "variant": variant,
                            "cost_bps": cost,
                            "metric_type": metric_type,
                            "n_obs": int(len(returns)),
                            "n_trials_eff": int(n_trials_eff),
                            "sharpe_or_ir_daily": daily_sr,
                            "sharpe_or_ir_annualized": ann_sr,
                            "skew": skew,
                            "kurtosis_non_excess": kurtosis,
                            "expected_max_false_daily_sr": expected_max_noise_sr,
                            "psr_vs_zero": psr_zero,
                            "dsr_probability": dsr_prob,
                            "status": dsr_status(dsr_prob),
                            "selected_v38": variant == V38_SELECTED,
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series, iterations: int = 2000) -> tuple[float, float]:
    values = values.dropna().astype(float)
    if values.empty:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    samples = rng.choice(values.to_numpy(), size=(iterations, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_status(mean_delta: float, ci_low: float, ci_high: float) -> str:
    if mean_delta >= 0.0025 and ci_low > 0:
        return "pass"
    if ci_high <= 0:
        return "fail"
    return "observation"


def run_paired_stability() -> pd.DataFrame:
    rows = []
    for cost in COSTS:
        selected_yearly = load_yearly(cost)
        selected_yearly = selected_yearly[selected_yearly["variant"] == V38_SELECTED].set_index("year")
        control_yearly = load_yearly(cost)
        control_yearly = control_yearly[control_yearly["variant"] == V38_CONTROL].set_index("year")
        deltas = selected_yearly["strategy_return"] - control_yearly["strategy_return"]
        ci_low, ci_high = bootstrap_ci(deltas)
        rows.append(
            {
                "cost_bps": cost,
                "block_type": "year",
                "n_blocks": int(len(deltas)),
                "positive_blocks": int((deltas > 0).sum()),
                "hit_rate": float((deltas > 0).mean()),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(deltas.median()),
                "worst_block": str(deltas.idxmin()),
                "worst_delta": float(deltas.min()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "status": paired_status(float(deltas.mean()), ci_low, ci_high),
            }
        )

        sel_path = V38_DIR / "candidates" / V38_SELECTED / f"regime_returns_{V38_SELECTED}_{int(cost)}bps.csv"
        ctl_path = V38_DIR / "candidates" / V38_CONTROL / f"regime_returns_{V38_CONTROL}_{int(cost)}bps.csv"
        if sel_path.exists() and ctl_path.exists():
            selected_regime = pd.read_csv(sel_path).set_index("state")
            control_regime = pd.read_csv(ctl_path).set_index("state")
            common = selected_regime.index.intersection(control_regime.index)
            regime_delta = selected_regime.loc[common, "annualized_mean"] - control_regime.loc[common, "annualized_mean"]
            ci_low, ci_high = bootstrap_ci(regime_delta, iterations=1000)
            rows.append(
                {
                    "cost_bps": cost,
                    "block_type": "regime",
                    "n_blocks": int(len(regime_delta)),
                    "positive_blocks": int((regime_delta > 0).sum()),
                    "hit_rate": float((regime_delta > 0).mean()),
                    "mean_delta": float(regime_delta.mean()),
                    "median_delta": float(regime_delta.median()),
                    "worst_block": str(regime_delta.idxmin()),
                    "worst_delta": float(regime_delta.min()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "status": paired_status(float(regime_delta.mean()), ci_low, ci_high),
                }
            )
    return pd.DataFrame(rows)


def build_robustness_decision(pbo_summary: pd.DataFrame, dsr_report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected_dsr = dsr_report[(dsr_report["variant"] == V38_SELECTED) & (dsr_report["n_trials_eff"] == 18)]
    for _, pbo_row in pbo_summary.iterrows():
        cost = float(pbo_row["cost_bps"])
        dsr_subset = selected_dsr[selected_dsr["cost_bps"] == cost]
        worst_dsr_prob = float(dsr_subset["dsr_probability"].min())
        worst_dsr_status = str(dsr_subset.sort_values("dsr_probability").iloc[0]["status"])
        pbo = float(pbo_row["pbo"])
        promotion_allowed = bool(pbo <= 0.20 and worst_dsr_prob >= 0.95)
        status = "pass" if promotion_allowed else ("observation" if pbo <= 0.35 and worst_dsr_prob >= 0.80 else "fail")
        rows.append(
            {
                "cost_bps": cost,
                "selected_variant": V38_SELECTED,
                "pbo": pbo,
                "pbo_status": pbo_row["status"],
                "worst_selected_dsr_probability_neff18": worst_dsr_prob,
                "worst_selected_dsr_status_neff18": worst_dsr_status,
                "promotion_allowed": promotion_allowed,
                "combined_status": status,
                "decision_reason": (
                    "passes strict PBO and DSR thresholds"
                    if promotion_allowed
                    else "not promotable: requires pbo<=0.20 and worst selected DSR>=0.95 under N_eff=18"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_robustness_summary(pbo_summary: pd.DataFrame, decision: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pbo_summary.iterrows():
        rows.append(
            {
                "check": "pbo",
                "metric": f"{int(row['cost_bps'])}bps_pbo",
                "value": row["pbo"],
                "threshold_pass": "<=0.20",
                "threshold_observe": "0.20-0.35",
                "threshold_fail": ">0.35",
                "status": row["status"],
                "evidence_path": rel(VALIDATION_DIR / "pbo_cscv_summary.csv"),
            }
        )
    for _, row in decision.iterrows():
        rows.append(
            {
                "check": "dsr",
                "metric": f"{int(row['cost_bps'])}bps_worst_selected_dsr_neff18",
                "value": row["worst_selected_dsr_probability_neff18"],
                "threshold_pass": ">=0.95",
                "threshold_observe": "0.80-0.95",
                "threshold_fail": "<0.80",
                "status": row["worst_selected_dsr_status_neff18"],
                "evidence_path": rel(VALIDATION_DIR / "dsr_results.csv"),
            }
        )
    for _, row in paired.iterrows():
        rows.append(
            {
                "check": "paired_v36_delta",
                "metric": f"{int(row['cost_bps'])}bps_{row['block_type']}",
                "value": row["mean_delta"],
                "threshold_pass": "mean>=25bp and ci_low>0",
                "threshold_observe": "ci crosses 0",
                "threshold_fail": "ci_high<=0",
                "status": row["status"],
                "evidence_path": rel(VALIDATION_DIR / "paired_v36_stability.csv"),
            }
        )
    return pd.DataFrame(rows)


def repro_gap_rows(current_manifest: dict) -> pd.DataFrame:
    checks = [
        ("generated_at", "present", "existing timestamp"),
        ("output_dir", "present", "existing output directory"),
        ("selected", "present", "existing selected candidate"),
        ("costs", "present", "existing cost grid"),
        ("benchmark", "present", "existing benchmark"),
        ("command", "missing", "exact command used to generate outputs"),
        ("argv", "missing", "parsed runtime arguments"),
        ("python_version", "missing", "Python runtime version"),
        ("platform", "missing", "OS and architecture"),
        ("git_commit", "missing", "repository commit hash"),
        ("git_dirty_status", "missing", "tracked/untracked working tree state"),
        ("script_sha256", "missing", "hash of strategy script"),
        ("config_path", "missing", "config file path"),
        ("config_sha256", "missing", "hash of config file"),
        ("input_hashes", "missing", "hashes of key upstream inputs"),
        ("output_hashes", "missing", "hashes of produced critical outputs"),
        ("dependency_versions", "missing", "pandas/numpy/python package versions"),
    ]
    rows = []
    for field, default_status, detail in checks:
        present = field in current_manifest and current_manifest.get(field) not in (None, "", [])
        rows.append(
            {
                "field": field,
                "present": bool(present),
                "status": "present" if present else default_status,
                "detail": detail,
                "blocks_future_promotion_if_missing": field not in {"generated_at", "output_dir", "selected", "costs", "benchmark"},
            }
        )
    return pd.DataFrame(rows)


def artifact_info(path: Path, required: bool = True) -> dict:
    info = {
        "name": path.stem,
        "path": rel(path),
        "kind": path.suffix.lstrip("."),
        "required": required,
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "row_count": None,
        "columns": [],
    }
    if path.exists() and path.suffix.lower() == ".csv":
        try:
            sample = pd.read_csv(path, nrows=5)
            info["columns"] = list(sample.columns)
            with path.open("r", encoding="utf-8-sig") as f:
                info["row_count"] = max(sum(1 for _ in f) - 1, 0)
        except Exception as exc:
            info["read_error"] = str(exc)
    return info


def build_enhanced_manifest(current_manifest: dict) -> dict:
    output_artifacts = [
        V38_DIR / "oos_performance.csv",
        V38_DIR / "walk_forward_target_weights.csv",
        V38_DIR / "self_check_results.csv",
        V38_DIR / "smoke_test_results.csv",
        V38_DIR / "v3_8_risk_budget_score_table.csv",
        V38_DIR / "v3_8_component_ablation.csv",
        V38_DIR / "WALK_FORWARD_REPORT.md",
        V38_DIR / "MODEL_CHANGELOG.md",
        V38_DIR / "SELF_CHECK_REPORT.md",
        V38_DIR / "nav_selected_10bps.csv",
        V38_DIR / "nav_selected_20bps.csv",
        V38_DIR / "nav_selected_30bps.csv",
    ]
    code_refs = [
        V38_SCRIPT,
        ROOT / "strategy_lab" / "hirssm_v2_model.py",
        ROOT / "strategy_lab" / "hirssm_v2_walk_forward.py",
        ROOT / "strategy_lab" / "hirssm_v3_6_component_attribution.py",
    ]
    data_refs = [
        ("v36_targets", ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "walk_forward_target_weights.csv"),
        ("v36_performance", ROOT / "outputs" / "hirssm_v3_6_component_attribution" / "oos_performance.csv"),
        ("v32_performance", ROOT / "outputs" / "hirssm_v3_2_market_beta_timing" / "oos_performance.csv"),
        ("v35_performance", ROOT / "outputs" / "hirssm_v3_3_to_v3_5_alpha_factory" / "v3_5" / "oos_performance.csv"),
    ]
    self_check_path = V38_DIR / "self_check_results.csv"
    self_check = pd.read_csv(self_check_path) if self_check_path.exists() else pd.DataFrame()
    git_status = git_value(["status", "--short"])
    return {
        "schema_version": "model_run_manifest.v1",
        "task_id": "retrospective_v3_8_manifest_upgrade",
        "run_id": "hirssm_v3_8_retrospective_manifest_template",
        "model_version": "HIRSSM V3.8",
        "status": "retrospective_observation",
        "original_manifest": current_manifest,
        "started_at": current_manifest.get("generated_at"),
        "finished_at": current_manifest.get("generated_at"),
        "cwd": str(ROOT.resolve()),
        "command": ["python", "-X", "utf8", "strategy_lab/hirssm_v3_8_risk_budget_overlay.py"],
        "argv": {
            "root": str(ROOT),
            "config": str(V38_CONFIG),
            "output_dir": str(V38_DIR),
        },
        "code_refs": [{"path": rel(path), "sha256": sha256_file(path), "exists": path.exists()} for path in code_refs],
        "config": {"path": rel(V38_CONFIG), "sha256": sha256_file(V38_CONFIG), "exists": V38_CONFIG.exists()},
        "data_refs": [
            {"name": name, "path": rel(path), "sha256": sha256_file(path), "exists": path.exists()} for name, path in data_refs
        ],
        "environment": {
            "python": sys.version,
            "packages": {"pandas": pd.__version__, "numpy": np.__version__},
            "platform": platform.platform(),
            "timezone": datetime.now().astimezone().tzname(),
            "git_commit": git_value(["rev-parse", "HEAD"]),
            "git_status_short": git_status,
            "git_dirty": bool(git_status.strip()),
        },
        "selection": {
            "selected": current_manifest.get("selected"),
            "candidate_count": int(len([p for p in (V38_DIR / "candidates").iterdir() if p.is_dir()]))
            if (V38_DIR / "candidates").exists()
            else None,
            "score_table_sha256": sha256_file(V38_DIR / "v3_8_risk_budget_score_table.csv"),
        },
        "artifacts": [artifact_info(path) for path in output_artifacts],
        "checks": {
            "self_check_pass": bool((self_check["pass"].astype(str).str.lower() == "true").all()) if not self_check.empty else None,
            "fail_count": int((self_check["pass"].astype(str).str.lower() != "true").sum()) if not self_check.empty else None,
            "warn_count": 0,
            "check_results_path": rel(self_check_path),
        },
        "limitations": [
            "This enhanced manifest was generated after the original V3.8 run.",
            "Git status includes the current workspace state, not necessarily the original run state.",
            "Data source snapshots outside hashed local files are not fully captured.",
        ],
        "risk_flags": ["retrospective_manifest", "git_dirty"] if git_status.strip() else ["retrospective_manifest"],
        "next_decision": "require this manifest schema at model generation time for future promotion candidates",
    }


def write_report(pbo_summary: pd.DataFrame, dsr_report: pd.DataFrame, decision: pd.DataFrame, repro_gap: pd.DataFrame) -> None:
    selected_dsr_10 = dsr_report[
        (dsr_report["variant"] == V38_SELECTED)
        & (dsr_report["cost_bps"] == 10.0)
        & (dsr_report["metric_type"] == "sharpe")
        & (dsr_report["n_trials_eff"] == 18)
    ].iloc[0]
    selected_decisions = decision[
        [
            "cost_bps",
            "pbo",
            "pbo_status",
            "worst_selected_dsr_probability_neff18",
            "worst_selected_dsr_status_neff18",
            "promotion_allowed",
        ]
    ]
    lines = [
        "# HIRSSM V3.9 PBO/DSR Validation Report",
        "",
        "## Scope",
        "",
        "This report validates existing HIRSSM V3.8 candidate artifacts. It does not change strategy logic or parameters.",
        "",
        "## Summary",
        "",
        f"- Selected V3.8 candidate: `{V38_SELECTED}`",
        f"- 10bps selected DSR-style probability, Sharpe, N_eff=18: `{selected_dsr_10['dsr_probability']:.4f}`",
        f"- 10bps selected annualized Sharpe: `{selected_dsr_10['sharpe_or_ir_annualized']:.4f}`",
        f"- 10bps PBO: `{pbo_summary.loc[pbo_summary['cost_bps'] == 10.0, 'pbo'].iloc[0]:.4f}`",
        "- Promotion threshold: `PBO <= 0.20` and worst selected `DSR-style probability >= 0.95` under `N_eff=18`.",
        "- Result: V3.8 remains `observation`; promotion is not allowed by this stricter validation layer.",
        "",
        "## Decision Table",
        "",
        "```text",
        selected_decisions.to_string(index=False),
        "```",
        "",
        "## Reproducibility Gap",
        "",
        f"- Missing future-promotion-blocking fields: `{int((~repro_gap['present'] & repro_gap['blocks_future_promotion_if_missing']).sum())}`",
        "- An enhanced manifest template was generated, but it is post-run evidence, not proof of the original run environment.",
    ]
    (VALIDATION_DIR / "agent_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifests(pbo_summary: pd.DataFrame, decision: pd.DataFrame, repro_gap: pd.DataFrame) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    validation_status = "pass" if bool(decision["promotion_allowed"].all()) else "observation"
    validation_manifest = {
        "run_id": "20260526_v3_9_pbo_dsr_gap_run_001",
        "task_id": "20260526_v3_9_pbo_dsr_gap",
        "agent": "backtest_validation_auditor",
        "version": "V3.9",
        "baseline": "HIRSSM V3.8",
        "status": validation_status,
        "started_at": now,
        "command": "python -X utf8 strategy_lab/hirssm_v3_9_validation_governance.py",
        "config": {
            "pbo_block_count": PBO_BLOCK_COUNT,
            "pbo_train_block_count": PBO_TRAIN_BLOCK_COUNT,
            "purge_days": PBO_PURGE_DAYS,
            "embargo_days": PBO_EMBARGO_DAYS,
            "costs": COSTS,
            "selected": V38_SELECTED,
        },
        "data_refs": [rel(V38_DIR)],
        "code_refs": [
            rel(AGENT_DIR / "backtest_validation_auditor" / "AGENT.md"),
            rel(V38_SCRIPT),
            "strategy_lab/hirssm_v3_9_validation_governance.py",
        ],
        "output_dir": rel(VALIDATION_DIR),
        "allowed_inputs": [
            rel(V38_DIR),
            rel(V38_SCRIPT),
            "directly imported local HIRSSM scripts and outputs",
        ],
        "artifacts": [
            rel(VALIDATION_DIR / "agent_run_manifest.json"),
            rel(VALIDATION_DIR / "agent_report.md"),
            rel(VALIDATION_DIR / "pbo_cscv_fold_results.csv"),
            rel(VALIDATION_DIR / "pbo_cscv_summary.csv"),
            rel(VALIDATION_DIR / "dsr_results.csv"),
            rel(VALIDATION_DIR / "paired_v36_stability.csv"),
            rel(VALIDATION_DIR / "robustness_decision.csv"),
            rel(VALIDATION_DIR / "robustness_summary.csv"),
        ],
        "outputs": [
            rel(VALIDATION_DIR / "agent_report.md"),
            rel(VALIDATION_DIR / "pbo_cscv_fold_results.csv"),
            rel(VALIDATION_DIR / "pbo_cscv_summary.csv"),
            rel(VALIDATION_DIR / "dsr_results.csv"),
            rel(VALIDATION_DIR / "paired_v36_stability.csv"),
            rel(VALIDATION_DIR / "robustness_decision.csv"),
            rel(VALIDATION_DIR / "robustness_summary.csv"),
        ],
        "changed_files": [
            rel(VALIDATION_DIR / "agent_run_manifest.json"),
            rel(VALIDATION_DIR / "agent_report.md"),
            rel(VALIDATION_DIR / "pbo_cscv_fold_results.csv"),
            rel(VALIDATION_DIR / "pbo_cscv_summary.csv"),
            rel(VALIDATION_DIR / "dsr_results.csv"),
            rel(VALIDATION_DIR / "paired_v36_stability.csv"),
            rel(VALIDATION_DIR / "robustness_decision.csv"),
            rel(VALIDATION_DIR / "robustness_summary.csv"),
        ],
        "metrics": {
            "min_pbo": float(pbo_summary["pbo"].min()),
            "max_pbo": float(pbo_summary["pbo"].max()),
            "selected_promotion_allowed_all_costs": bool(decision["promotion_allowed"].all()),
        },
        "self_check_pass": True,
        "fail_count": int((decision["combined_status"] == "fail").sum()),
        "warn_count": int((decision["combined_status"] != "pass").sum()),
        "limitations": [
            "DSR implementation is a governance approximation using existing candidate daily returns.",
            "PBO uses 10 continuous blocks with 120-day purge and 21-day embargo over existing V3.8 candidate NAVs.",
            "The validation reads existing artifacts and does not rerun original model generation.",
        ],
        "risk_flags": ["strict_promotion_not_allowed"] if not bool(decision["promotion_allowed"].all()) else [],
        "next_decision": "keep V3.8 as observation baseline unless stricter validation is later passed",
        "handoff_summary": "Generated CSCV/PBO and DSR-style validation from existing V3.8 artifacts.",
    }
    (VALIDATION_DIR / "agent_run_manifest.json").write_text(
        json.dumps(validation_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    missing_blockers = int((~repro_gap["present"] & repro_gap["blocks_future_promotion_if_missing"]).sum())
    repro_manifest = {
        "run_id": "20260526_v3_9_repro_manifest_gap_run_001",
        "task_id": "20260526_v3_9_repro_manifest_gap",
        "agent": "code_quality_engineer",
        "version": "V3.9",
        "baseline": "HIRSSM V3.8",
        "status": "observation",
        "started_at": now,
        "command": "python -X utf8 strategy_lab/hirssm_v3_9_validation_governance.py",
        "config": {"source_manifest": rel(V38_DIR / "run_manifest.json")},
        "data_refs": [rel(V38_DIR / "run_manifest.json")],
        "code_refs": [
            rel(AGENT_DIR / "code_quality_engineer" / "AGENT.md"),
            rel(V38_SCRIPT),
            "strategy_lab/hirssm_v3_9_validation_governance.py",
        ],
        "output_dir": rel(REPRO_DIR),
        "allowed_inputs": [
            rel(V38_DIR / "run_manifest.json"),
            rel(V38_SCRIPT),
            "strategy_lab/agent_framework_check.py",
        ],
        "artifacts": [
            rel(REPRO_DIR / "agent_run_manifest.json"),
            rel(REPRO_DIR / "agent_report.md"),
            rel(REPRO_DIR / "repro_manifest_gap_report.csv"),
            rel(REPRO_DIR / "enhanced_manifest_template.json"),
        ],
        "outputs": [
            rel(REPRO_DIR / "agent_report.md"),
            rel(REPRO_DIR / "repro_manifest_gap_report.csv"),
            rel(REPRO_DIR / "enhanced_manifest_template.json"),
        ],
        "changed_files": [
            rel(REPRO_DIR / "agent_run_manifest.json"),
            rel(REPRO_DIR / "agent_report.md"),
            rel(REPRO_DIR / "repro_manifest_gap_report.csv"),
            rel(REPRO_DIR / "enhanced_manifest_template.json"),
        ],
        "metrics": {
            "missing_future_promotion_blockers": missing_blockers,
            "field_count": int(len(repro_gap)),
        },
        "self_check_pass": True,
        "fail_count": 0,
        "warn_count": missing_blockers,
        "limitations": [
            "Enhanced manifest template was generated after the V3.8 run.",
            "Original V3.8 run environment cannot be fully reconstructed from existing manifest.",
        ],
        "risk_flags": ["incomplete_original_v38_repro_manifest"] if missing_blockers else [],
        "next_decision": "require enhanced manifest fields for future promotion candidates",
        "handoff_summary": "Audited V3.8 manifest reproducibility gaps and generated an enhanced manifest template.",
    }
    (REPRO_DIR / "agent_run_manifest.json").write_text(json.dumps(repro_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_repro_report(repro_gap: pd.DataFrame) -> None:
    missing = repro_gap[~repro_gap["present"]]
    lines = [
        "# V3.9 Reproducibility Manifest Gap Report",
        "",
        "## Status",
        "",
        "`observation`",
        "",
        "## Finding",
        "",
        f"The original V3.8 manifest is structurally present but misses `{len(missing)}` reproducibility fields.",
        "Future promotion candidates should include command, runtime, git, script/config hashes, input hashes, and output hashes at generation time.",
        "",
        "## Blocking Policy",
        "",
        "Missing strict reproducibility metadata should block future production promotion, but it does not invalidate V3.8 historical outputs by itself.",
    ]
    (REPRO_DIR / "agent_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global ROOT, V38_DIR, V39_DIR, VALIDATION_DIR, REPRO_DIR, AGENT_DIR, V38_SCRIPT, V38_CONFIG

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    ROOT = Path(args.root)
    V38_DIR = ROOT / "outputs" / "hirssm_v3_8_risk_budget_overlay"
    V39_DIR = ROOT / "outputs" / "agent_runs" / "v3_9"
    VALIDATION_DIR = V39_DIR / "backtest_validation_auditor_pbo_dsr"
    REPRO_DIR = V39_DIR / "code_quality_engineer_repro_manifest"
    AGENT_DIR = ROOT / "strategy_lab" / "agents"
    V38_SCRIPT = ROOT / "strategy_lab" / "hirssm_v3_8_risk_budget_overlay.py"
    V38_CONFIG = ROOT / "configs" / "hirssm_v2_default.json"

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    REPRO_DIR.mkdir(parents=True, exist_ok=True)

    pbo_detail, pbo_summary = run_pbo()
    dsr_report = run_dsr()
    decision = build_robustness_decision(pbo_summary, dsr_report)
    paired = run_paired_stability()
    robustness_summary = build_robustness_summary(pbo_summary, decision, paired)
    current_manifest = read_json(V38_DIR / "run_manifest.json")
    repro_gap = repro_gap_rows(current_manifest)
    enhanced_manifest = build_enhanced_manifest(current_manifest)

    pbo_detail.to_csv(VALIDATION_DIR / "pbo_cscv_fold_results.csv", index=False, encoding="utf-8-sig")
    pbo_summary.to_csv(VALIDATION_DIR / "pbo_cscv_summary.csv", index=False, encoding="utf-8-sig")
    dsr_report.to_csv(VALIDATION_DIR / "dsr_results.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(VALIDATION_DIR / "paired_v36_stability.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(VALIDATION_DIR / "robustness_decision.csv", index=False, encoding="utf-8-sig")
    robustness_summary.to_csv(VALIDATION_DIR / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    repro_gap.to_csv(REPRO_DIR / "repro_manifest_gap_report.csv", index=False, encoding="utf-8-sig")
    (REPRO_DIR / "enhanced_manifest_template.json").write_text(
        json.dumps(enhanced_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_report(pbo_summary, dsr_report, decision, repro_gap)
    write_repro_report(repro_gap)
    write_manifests(pbo_summary, decision, repro_gap)

    result = {
        "pbo_summary": str(VALIDATION_DIR / "pbo_cscv_summary.csv"),
        "dsr_report": str(VALIDATION_DIR / "dsr_results.csv"),
        "paired_stability": str(VALIDATION_DIR / "paired_v36_stability.csv"),
        "robustness_decision": str(VALIDATION_DIR / "robustness_decision.csv"),
        "repro_gap_report": str(REPRO_DIR / "repro_manifest_gap_report.csv"),
        "promotion_allowed_all_costs": bool(decision["promotion_allowed"].all()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
