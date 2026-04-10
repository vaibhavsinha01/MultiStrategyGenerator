"""
backtester.py
─────────────
Clean, fast, production-ready backtester.

Key improvements:
  • Minimal logging (debug-controlled)
  • Robust NaN handling
  • Faster execution
  • Clean failure handling
"""

import warnings
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from signals import SIGNALS
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

COMMISSION   = 0.002 # changed from 0.002
INITIAL_CASH = 1_000_000

# ─────────────────────────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────────────────────────
def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

# ─────────────────────────────────────────────────────────────
# STRATEGY BUILDER
# ─────────────────────────────────────────────────────────────
def _build_strategy_class(strat_dict: dict) -> type:
    signal_keys = strat_dict["signals"]
    direction   = strat_dict["direction"]
    tp_pct      = strat_dict["tp"]
    sl_pct      = strat_dict["sl"]

    def init(self):
        self._signals = {
            k: self.data[f"signal_{k}"]
            for k in signal_keys
        }

    def next(self):
        i = len(self.data) - 1
        lookback = 4

        if i < lookback:
            return

        # Count how many signals fired within the lookback window
        n_active = sum(
            any(self._signals[k][j] for j in range(i - lookback + 1, i + 1))
            for k in signal_keys
        )

        threshold = max(2, int(len(signal_keys) * 0.67))
        entry = n_active >= threshold

        if not entry:
            return

        price = self.data.Close[-1]

        if direction == "bull":
            if not self.position.is_long:
                self.buy(
                    sl=price * (1 - sl_pct),
                    tp=price * (1 + tp_pct),
                )
        else:
            if not self.position.is_short:
                self.sell(
                    sl=price * (1 + sl_pct),
                    tp=price * (1 - tp_pct),
                )

    return type(
        f"Strat_{strat_dict['id']}",
        (Strategy,),
        {"init": init, "next": next},
    )


# ─────────────────────────────────────────────────────────────
# MAIN BACKTEST FUNCTION
# ─────────────────────────────────────────────────────────────
def run_backtest(
    strat_dict: dict,
    df: pd.DataFrame,
    debug: bool = False
) -> dict | None:

    try:
        df = df.copy()

        # ── 1. COMPUTE SIGNALS ─────────────────────────────
        for k in strat_dict["signals"]:
            sig = SIGNALS[k]["fn"](df)

            if not isinstance(sig, pd.Series):
                raise ValueError(f"{k} did not return Series")

            df[f"signal_{k}"] = sig.fillna(False).astype(bool)

        if debug:
            print(f"\n🔍 Strategy {strat_dict['id']}")
            for k in strat_dict["signals"]:
                print(f"{k}: {df[f'signal_{k}'].sum()} signals")

        # ── 2. PREP DATA ──────────────────────────────────
        ohlcv = _prep_df(df)

        # ── 3. RUN BACKTEST ───────────────────────────────
        StratClass = _build_strategy_class(strat_dict)

        bt = Backtest(
            ohlcv,
            StratClass,
            cash=INITIAL_CASH,
            commission=COMMISSION,
            exclusive_orders=True,
            trade_on_close=False,
        )

        stats = bt.run()

        # ── 4. EXTRACT METRICS ────────────────────────────
        n_trades = int(stats.get("# Trades", 0))
        if n_trades == 0:
            return None

        return_pct = float(stats.get("Return [%]", 0.0))
        max_dd     = float(stats.get("Max. Drawdown [%]", 0.0))
        win_rate   = float(stats.get("Win Rate [%]", 0.0))

        # Sharpe FIX (important)
        sharpe = stats.get("Sharpe Ratio", 0.0)
        if sharpe is None or not np.isfinite(sharpe):
            sharpe = 0.0

        # sanity
        if not np.isfinite(return_pct) or not np.isfinite(max_dd):
            return None

        if debug:
            print(f"Trades: {n_trades}")
            print(f"Return: {return_pct}")
            print(f"DD: {max_dd}")
            print(f"WinRate: {win_rate}")

        return {
            "return_pct":   round(return_pct, 4),
            "sharpe":       round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "n_trades":     n_trades,
            "win_rate":     round(win_rate, 4),
        }

    except Exception as e:
        logging.error(f"Strategy failed: {strat_dict.get('id')}")
        if debug:
            print(f"❌ Error in {strat_dict.get('id')}: {e}")
        return None