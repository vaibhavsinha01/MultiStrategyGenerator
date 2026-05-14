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

MIN_TRADES    =  0
MAX_DRAWDOWN  = -100.0   # worse than -25% → reject
MIN_WIN_RATE = 0
MIN_RETURN = -100.0

# ── 1. Filter ─────────────────────────────────────────────────────────────────

def filter_metrics(metrics: dict) -> bool:
    if metrics is None:
        return False

    cond1 = metrics["n_trades"]    >= MIN_TRADES
    cond2 = metrics["max_drawdown"] >= MAX_DRAWDOWN
    cond3 = metrics["win_rate"]     >= MIN_WIN_RATE
    cond4 = metrics["return_pct"]   >= MIN_RETURN

    return cond1 and cond2 and cond3 and cond4


# ── 2. Score ──────────────────────────────────────────────────────────────────

def score_strategy(metrics: dict) -> float:
    r  = metrics["return_pct"]   / 100.0
    dd = abs(metrics["max_drawdown"]) / 100.0
    wr = metrics["win_rate"]     / 100.0
    n  = metrics["n_trades"]

    # Trade count confidence penalty (diminishing returns above 50)
    trade_confidence = min(n / 50.0, 1.0)

    base = 0.40*r - 0.30*dd + 0.30*wr # here a lot of weightage is given to winrate
    return round(base * trade_confidence, 6)

# ── 3. Rank ───────────────────────────────────────────────────────────────────

def rank_strategies(results: list[dict], top_n: int = 50) -> list[dict]:
    """
    Sort results by score descending and return top_n.

    Each item in results is expected to have keys:
        strategy, metrics, score
    """
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
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

    A strategy passes validation if:
      • It still meets filter thresholds on test data, OR
      • Its test return is above consistency_threshold

    Returns list of validated result dicts (with test_metrics added).
    """
    validated = []
    for item in top_results:
        strat  = item["strategy"]
        t_metrics = run_backtest(strat, test_df)

        if t_metrics is None:
            continue

        # Consistency check: passes filter OR positive return
        passes = (
            filter_metrics(t_metrics) or
            t_metrics["return_pct"] > consistency_threshold
        )
        if not passes:
            continue

        validated.append({
            **item,
            "test_metrics": t_metrics,
            "test_score":   score_strategy(t_metrics),
        })

    # Sort by combined train+test score
    validated.sort(
        key=lambda x: x["score"] + x["test_score"],
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
        "train_drawdown":  metrics["max_drawdown"],
        "train_trades":    metrics["n_trades"],
        "train_winrate":   metrics["win_rate"],
        "score":           item["score"],
    }

    row.update(_flatten_regime_metrics(metrics, prefix="train"))

    # test metrics if present
    if "test_metrics" in item:
        tm = item["test_metrics"]
        row.update({
            "test_return":   tm["return_pct"],
            "test_sharpe":   tm["sharpe"],
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
) -> pd.DataFrame:
    """
    Save results list to CSV and return as DataFrame.

    Parameters
    ----------
    results : list of result dicts
    path    : output file path
    append  : if True and file exists, append without header
    """
    rows = [_flatten_result(r) for r in results]
    df   = pd.DataFrame(rows)

    mode   = "a" if (append and os.path.exists(path)) else "w"
    header = not (append and os.path.exists(path))
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
