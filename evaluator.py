"""
evaluator.py
────────────
Filtering, scoring, ranking, train/test split, and result storage.

Pipeline
  1. filter_metrics()    → reject under-performing strategies
  2. score_strategy()    → composite score
  3. rank_strategies()   → sort by score
  4. train_test_split()  → split df 70/30
  5. validate_on_test()  → re-run top strategies on test set
  6. save_results()      → write CSV
"""

import os
import pandas as pd
from backtester import run_backtest


REGIMES = ("chop", "trendy", "volatile")
REGIME_METRIC_COLUMNS = (
    "trades",
    "trade_share",
    "pnl",
    "return_pct",
    "win_rate",
    "avg_trade_return_pct",
    "total_trade_return_pct",
)


# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_TRADES = 5          # was 10 — relaxed to pass daily/4h data with fewer bars
MIN_TEST_TRADES = 2    # was 3
MAX_DRAWDOWN = -70.0   # was -50
MIN_WIN_RATE = 15.0    # was 20
MIN_RETURN = -15.0     # was -10

# ── 1. Filter ─────────────────────────────────────────────────────────────────

def filter_metrics(metrics: dict, min_trades: int = MIN_TRADES) -> bool:
    if metrics is None:
        return False

    cond1 = metrics["n_trades"]    >= min_trades
    cond2 = metrics["max_drawdown"] >= MAX_DRAWDOWN
    cond3 = metrics["win_rate"]     >= MIN_WIN_RATE
    cond4 = metrics["return_pct"]   >= MIN_RETURN

    return cond1 and cond2 and cond3 and cond4


# ── 2. Score ──────────────────────────────────────────────────────────────────

def score_strategy(metrics: dict) -> float:
    r  = metrics["return_pct"] / 100.0
    dd = abs(metrics["max_drawdown"]) / 100.0
    wr = metrics["win_rate"] / 100.0
    sh = max(min(float(metrics.get("sharpe", 0.0)), 5.0), -5.0) / 5.0
    n  = metrics["n_trades"]

    # Trade count confidence penalty (diminishing returns above 50)
    trade_confidence = min(n / 50.0, 1.0)

    base = 0.9 * r + 0.6 * sh + 0.6 * wr - 0.4 * dd
    base = min(1,base) # now the max value is changed to 1
    return round(base * trade_confidence, 6)


def validation_adjusted_score(
    train_metrics: dict,
    test_metrics: dict | None = None,
    monte_carlo: dict | None = None,
    cross_symbol: dict | None = None,
    sensitivity: dict | None = None,
) -> float:
    """
    Robustness-aware score:
    55% in/out score, 15% train-test consistency, 10% Monte Carlo,
    10% cross-symbol validation, 10% parameter sensitivity stability.
    """
    train_score = score_strategy(train_metrics)
    test_score = score_strategy(test_metrics) if test_metrics else train_score * 0.5
    out_sample_score = 0.35 * train_score + 0.65 * test_score

    train_return = abs(float(train_metrics.get("return_pct", 0.0)))
    test_return = float((test_metrics or {}).get("return_pct", 0.0))
    consistency = 1.0 if train_return <= 1e-9 else max(0.0, min(1.0, test_return / train_return))

    mc = monte_carlo or {}
    mc_loss = float(mc.get("loss_probability_pct", 50.0))
    mc_drawdown = abs(float(mc.get("worst_drawdown_pct", 50.0)))
    mc_component = max(0.0, min(1.0, (100.0 - mc_loss) / 100.0)) * max(0.0, min(1.0, 1.0 - mc_drawdown / 100.0))

    xs = cross_symbol or {}
    cross_component = max(0.0, min(1.0, float(xs.get("pass_rate_pct", 0.0)) / 100.0))

    sens = sensitivity or {}
    return_std = float(sens.get("return_std", 25.0) or 25.0)
    sensitivity_component = max(0.0, min(1.0, 1.0 - return_std / 50.0))

    robust = (
        0.35 * out_sample_score +
        0.25 * consistency +
        0.15 * mc_component +
        0.15 * cross_component +
        0.10 * sensitivity_component
    )
    return round(robust, 6)

# ── 3. Rank ───────────────────────────────────────────────────────────────────

def rank_strategies(results: list[dict], top_n: int = 50) -> list[dict]:
    """
    Sort results by score descending and return top_n.

    Each item in results is expected to have keys:
        strategy, metrics, score
    """
    ranked = sorted(
        results,
        key=lambda x: (x["score"], x["metrics"]["n_trades"]),
        reverse=True,
    )
    return ranked[:top_n]


# ── 4. Train / test split ─────────────────────────────────────────────────────

def train_test_split(df: pd.DataFrame, train_ratio: float = 0.80):
    """Split df into (train_df, test_df) by row index."""
    split = int(len(df) * train_ratio)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── 5. Validate on test set ───────────────────────────────────────────────────

def validate_on_test(
    top_results: list[dict],
    test_df: pd.DataFrame,
    consistency_threshold: float = 0.0,
) -> list[dict]:
    """
    Re-run each top strategy on the test set.

    A strategy passes validation if it meets the test trade-count/quality
    thresholds and its test return is above consistency_threshold.

    Returns list of validated result dicts (with test_metrics added).
    """
    validated = []
    for item in top_results:
        strat  = item["strategy"]
        t_metrics = run_backtest(strat, test_df)

        if t_metrics is None:
            continue

        passes = (
            filter_metrics(t_metrics, min_trades=MIN_TEST_TRADES) and
            t_metrics["return_pct"] > consistency_threshold
        )
        if not passes:
            continue

        validated.append({
            **item,
            "test_metrics": t_metrics,
            "test_score":   score_strategy(t_metrics),
            "robust_score": validation_adjusted_score(item["metrics"], t_metrics),
        })

    # Sort by combined train+test score
    validated.sort(
        key=lambda x: x.get("robust_score", x["score"] + x["test_score"]),
        reverse=True
    )
    return validated


# ── 6. Storage ────────────────────────────────────────────────────────────────

def _flatten_result(item: dict) -> dict:
    """Flatten a result dict to a single CSV row."""
    strat   = item["strategy"]
    metrics = item["metrics"]
    row = {
        "id":           strat["id"],
        "direction":    strat["direction"],
        "signals":      "|".join(strat["signals"]),
        "n_signals":    strat["n_signals"],
        "tp":           strat["tp"],
        "sl":           strat["sl"],
        # train metrics
        "train_return":    metrics["return_pct"],
        "train_sharpe":    metrics["sharpe"],
        "train_sharp":     metrics["sharpe"],
        "train_drawdown":  metrics["max_drawdown"],
        "train_trades":    metrics["n_trades"],
        "train_winrate":   metrics["win_rate"],
        "score":           item["score"],
        "robust_score":    item.get("robust_score", item.get("score", "")),
    }

    row.update(_flatten_regime_metrics(metrics, prefix="train"))

    # test metrics if present
    if "test_metrics" in item:
        tm = item["test_metrics"]
        row.update({
            "test_return":   tm["return_pct"],
            "test_sharpe":   tm["sharpe"],
            "test_sharp":    tm["sharpe"],
            "test_drawdown": tm["max_drawdown"],
            "test_trades":   tm["n_trades"],
            "test_winrate":  tm["win_rate"],
            "test_score":    item.get("test_score", ""),
        })
        row.update(_flatten_regime_metrics(tm, prefix="test"))
    return row


def _flatten_regime_metrics(metrics: dict, prefix: str) -> dict:
    regime_metrics = metrics.get("regime_metrics", {})
    row = {}
    for regime in REGIMES:
        values = regime_metrics.get(regime, {})
        for metric_name in REGIME_METRIC_COLUMNS:
            key = f"{prefix}_{regime}_{metric_name}"
            row[key] = values.get(metric_name, 0)
    return row


def print_regime_breakdown(
    results: list[dict],
    title: str = "Regime Breakdown",
    metric_key: str = "metrics",
    top_n: int = 10,
) -> None:
    """Print compact per-regime trade/return diagnostics for top results."""
    if not results:
        return

    print(f"\n{title}  (showing {min(top_n, len(results))} strategies)")
    header = (
        f"{'ID':<10} {'Regime':<9} {'Trades':>6} {'Share':>7} "
        f"{'Ret%':>8} {'WR%':>7} {'AvgTr%':>8}"
    )
    print(header)
    print("-" * len(header))

    for item in results[:top_n]:
        strategy_id = item["strategy"]["id"]
        metrics = item.get(metric_key, {})
        regime_metrics = metrics.get("regime_metrics", {})
        for regime in REGIMES:
            values = regime_metrics.get(regime, {})
            print(
                f"{strategy_id:<10} {regime:<9} "
                f"{values.get('trades', 0):>6} "
                f"{values.get('trade_share', 0) * 100:>6.1f}% "
                f"{values.get('return_pct', 0):>8.2f} "
                f"{values.get('win_rate', 0):>7.1f} "
                f"{values.get('avg_trade_return_pct', 0):>8.2f}"
            )

def save_results(
    results: list[dict],
    path: str = "strategy_results.csv",
    append: bool = False,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """
    Save results list to CSV and return as DataFrame.

    Parameters
    ----------
    results : list of result dicts
    path    : output file path
    append  : if True and file exists, append without header
    metadata: optional columns to add to every exported row
    """
    rows = [_flatten_result(r) for r in results]
    df   = pd.DataFrame(rows)
    if metadata:
        for key, value in reversed(list(metadata.items())):
            df.insert(0, key, value)

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    if append and file_exists:
        existing = pd.read_csv(path)
        new_columns = [col for col in df.columns if col not in existing.columns]
        missing_columns = [col for col in existing.columns if col not in df.columns]
        if new_columns or missing_columns:
            for col in new_columns:
                existing[col] = ""
            for col in missing_columns:
                df[col] = ""
            columns = list(existing.columns)
            existing.to_csv(path, index=False)
            df = df[columns]

    mode   = "a" if (append and file_exists) else "w"
    header = not (append and file_exists)
    df.to_csv(path, mode=mode, header=header, index=False)

    return df


def print_summary(results: list[dict], title: str = "Results") -> None:
    """Print a readable summary table."""
    print(f"\n{'═'*72}")
    print(f"  {title}  ({len(results)} strategies)")
    print(f"{'═'*72}")
    header = f"{'ID':<10} {'Dir':<5} {'Signals':<30} {'Ret%':>7} {'Sharpe':>7} {'DD%':>7} {'#T':>5} {'WR%':>6} {'Score':>8}"
    print(header)
    print("─"*72)
    for r in results:
        s  = r["strategy"]
        m  = r["metrics"]
        sc = r["score"]
        sigs = "|".join(s["signals"])
        print(
            f"{s['id']:<10} {s['direction']:<5} {sigs:<30} "
            f"{m['return_pct']:>7.2f} {m['sharpe']:>7.3f} "
            f"{m['max_drawdown']:>7.2f} {m['n_trades']:>5} "
            f"{m['win_rate']:>6.1f} {sc:>8.5f}"
        )
    print("─"*72)
