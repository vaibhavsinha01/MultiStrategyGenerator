from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backtester import run_backtest
from evaluator import score_strategy, train_test_split, validation_adjusted_score
from feature_engineering import cached_feature_engineer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
UNIFIED_RESULTS_CSV = RESULTS_DIR / "strategy_results_unified.csv"


def monte_carlo_simulation(returns: Iterable[float], n_runs: int = 1000, horizon: int = 250) -> dict:
    values = pd.Series(list(returns), dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"n_runs": 0, "error": "No returns available"}
    rng = np.random.default_rng(42)
    paths = rng.choice(values.to_numpy(), size=(n_runs, horizon), replace=True)
    equity = (1.0 + paths).cumprod(axis=1)
    terminal = equity[:, -1] - 1.0
    drawdowns = equity / np.maximum.accumulate(equity, axis=1) - 1.0
    result = {
        "result_type": "monte_carlo",
        "n_runs": int(n_runs),
        "horizon": int(horizon),
        "median_return_pct": round(float(np.median(terminal) * 100), 4),
        "p05_return_pct": round(float(np.percentile(terminal, 5) * 100), 4),
        "p95_return_pct": round(float(np.percentile(terminal, 95) * 100), 4),
        "worst_drawdown_pct": round(float(drawdowns.min() * 100), 4),
        "loss_probability_pct": round(float((terminal < 0).mean() * 100), 4),
    }
    append_unified_result(result)
    return result


def validate_in_out_sample(strategy: dict, df: pd.DataFrame, train_ratio: float = 0.8) -> dict:
    train_df, test_df = train_test_split(df, train_ratio)
    train = run_backtest(strategy, train_df)
    test = run_backtest(strategy, test_df)
    result = {
        "result_type": "in_out_validation",
        "strategy_id": strategy.get("id"),
        "train_return_pct": (train or {}).get("return_pct"),
        "train_sharpe": (train or {}).get("sharpe"),
        "train_drawdown": (train or {}).get("max_drawdown"),
        "train_trades": (train or {}).get("n_trades"),
        "train_win_rate": (train or {}).get("win_rate"),
        "test_return_pct": (test or {}).get("return_pct"),
        "test_sharpe": (test or {}).get("sharpe"),
        "test_drawdown": (test or {}).get("max_drawdown"),
        "test_trades": (test or {}).get("n_trades"),
        "test_win_rate": (test or {}).get("win_rate"),
        "train_score": score_strategy(train) if train else None,
        "test_score": score_strategy(test) if test else None,
        "robust_score": validation_adjusted_score(train, test) if train else None,
        "passes": bool(train and test and test["n_trades"] >= 10 and test["return_pct"] > -10),
    }
    append_unified_result(result)
    return result


def cross_symbol_validation(strategy: dict, symbols: Iterable[str], timeframe: str = "15m") -> dict:
    rows = []
    for symbol in symbols:
        path = DATA_DIR / f"{symbol.lower()}_{timeframe.lower()}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        df = cached_feature_engineer(raw, symbol.lower(), timeframe.lower())
        metrics = run_backtest(strategy, df)
        rows.append({
            "result_type": "cross_symbol_validation",
            "strategy_id": strategy.get("id"),
            "symbol": symbol.lower(),
            "timeframe": timeframe.lower(),
            "return_pct": (metrics or {}).get("return_pct"),
            "sharpe": (metrics or {}).get("sharpe"),
            "max_drawdown": (metrics or {}).get("max_drawdown"),
            "n_trades": (metrics or {}).get("n_trades"),
            "win_rate": (metrics or {}).get("win_rate"),
            "passes": bool(metrics and metrics["n_trades"] >= 10 and metrics["return_pct"] > -10),
        })
    pass_rate = round(float(pd.Series([r["passes"] for r in rows]).mean() * 100), 4) if rows else 0.0
    for row in rows:
        row["symbols_tested"] = len(rows)
        row["pass_rate_pct"] = pass_rate
    append_unified_rows(rows)
    result = {"strategy_id": strategy.get("id"), "symbols_tested": len(rows), "pass_rate_pct": pass_rate, "rows": rows}
    return result


def parameter_sensitivity(strategy: dict, df: pd.DataFrame, pct_steps: Iterable[float] = (-0.2, -0.1, 0, 0.1, 0.2)) -> dict:
    rows = []
    for step in pct_steps:
        candidate = dict(strategy)
        candidate["tp"] = max(0.001, float(strategy.get("tp", 0.02)) * (1 + step))
        candidate["sl"] = max(0.001, float(strategy.get("sl", 0.01)) * (1 - step))
        metrics = run_backtest(candidate, df)
        rows.append({
            "result_type": "parameter_sensitivity",
            "strategy_id": strategy.get("id"),
            "step": step,
            "tp": candidate["tp"],
            "sl": candidate["sl"],
            "return_pct": (metrics or {}).get("return_pct"),
            "sharpe": (metrics or {}).get("sharpe"),
            "max_drawdown": (metrics or {}).get("max_drawdown"),
            "n_trades": (metrics or {}).get("n_trades"),
        })
    valid_returns = [r["return_pct"] for r in rows if r["return_pct"] is not None]
    result = {
        "result_type": "parameter_sensitivity_summary",
        "strategy_id": strategy.get("id"),
        "return_std": round(float(np.std(valid_returns)), 4) if valid_returns else None,
        "robust": bool(valid_returns and np.std(valid_returns) < 20),
    }
    for row in rows:
        row["return_std"] = result["return_std"]
        row["robust"] = result["robust"]
    append_unified_rows(rows)
    return result


def append_unified_result(row: dict) -> None:
    append_unified_rows([row])


def append_unified_rows(rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    file_exists = UNIFIED_RESULTS_CSV.exists() and UNIFIED_RESULTS_CSV.stat().st_size > 0
    if file_exists:
        existing = pd.read_csv(UNIFIED_RESULTS_CSV, nrows=0)
        for col in existing.columns:
            if col not in df.columns:
                df[col] = ""
        new_columns = [col for col in df.columns if col not in existing.columns]
        if new_columns:
            full_existing = pd.read_csv(UNIFIED_RESULTS_CSV)
            for col in new_columns:
                full_existing[col] = ""
            full_existing.to_csv(UNIFIED_RESULTS_CSV, index=False)
            df = df[list(full_existing.columns)]
        else:
            df = df[list(existing.columns)]
    df.to_csv(UNIFIED_RESULTS_CSV, mode="a", header=not file_exists, index=False)


def latest_validation_summary() -> dict:
    if not UNIFIED_RESULTS_CSV.exists():
        return {"available": False}
    df = pd.read_csv(UNIFIED_RESULTS_CSV)
    if "result_type" not in df.columns:
        return {"available": False}
    summary = {"available": True}
    for result_type in (
        "monte_carlo",
        "in_out_validation",
        "cross_symbol_validation",
        "parameter_sensitivity",
    ):
        subset = df[df["result_type"] == result_type]
        summary[result_type] = subset.tail(1).to_dict("records")[0] if not subset.empty else None
    return summary
