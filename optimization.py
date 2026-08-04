"""Manual, single-flight walk-forward optimization for saved strategies."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd
from backtesting import Backtest

from backtester import COMMISSION, INITIAL_CASH, _build_strategy_class, _prep_df
from feature_engineering import feature_engineer
from signals import SIGNALS

_lock = threading.Lock()


def try_acquire() -> bool:
    return _lock.acquire(blocking=False)


def release() -> None:
    if _lock.locked():
        _lock.release()


def _data_path(symbol: str, timeframe: str, root: Path) -> Path:
    tf = "D" if timeframe.upper() == "1D" else timeframe
    path = root / "data" / f"{symbol.lower()}_{tf}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name}")
    return path


def optimize_walk_forward(strategy: dict, symbol: str, timeframe: str, root: Path) -> dict[str, Any]:
    """Optimize TP/SL with Backtesting.py's SAMBO, evaluating the held-out tail."""
    raw = pd.read_csv(_data_path(symbol, timeframe, root))
    df = feature_engineer(raw)
    for key in strategy["signals"]:
        if key not in SIGNALS:
            raise ValueError(f"Unknown signal: {key}")
        df[f"signal_{key}"] = SIGNALS[key]["fn"](df).fillna(False).astype(bool)
    split = max(int(len(df) * 0.7), 1)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    if len(test) < 2:
        raise ValueError("Dataset is too short for walk-forward optimization")
    cls = _build_strategy_class(strategy)
    bt = Backtest(_prep_df(train), cls, cash=INITIAL_CASH, commission=COMMISSION, exclusive_orders=True, trade_on_close=False)
    stats = bt.optimize(
        tp_pct=[0.005, 0.01, 0.02, 0.03, 0.05],
        sl_pct=[0.003, 0.005, 0.01, 0.02, 0.03],
        constraint=lambda p: p.tp_pct > p.sl_pct,
        maximize="Sharpe Ratio",
        method="sambo",
    )
    params = {"tp": float(stats._strategy.tp_pct), "sl": float(stats._strategy.sl_pct)}
    optimized = {**strategy, **params}
    test_bt = Backtest(_prep_df(test), _build_strategy_class(optimized), cash=INITIAL_CASH, commission=COMMISSION, exclusive_orders=True, trade_on_close=False)
    test_stats = test_bt.run()
    return {
        "method": "sambo", "split": "70/30 walk-forward", "parameters": params,
        "train": {"return_pct": float(stats.get("Return [%]", 0)), "sharpe": float(stats.get("Sharpe Ratio", 0) or 0), "trades": int(stats.get("# Trades", 0))},
        "test": {"return_pct": float(test_stats.get("Return [%]", 0)), "sharpe": float(test_stats.get("Sharpe Ratio", 0) or 0), "trades": int(test_stats.get("# Trades", 0))},
    }
