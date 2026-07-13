"""
Auto-generated Simurgh Capital trading bot.
Strategy id: 622937c8

Setup:
  cd simurgh_trading_bot && pip install -e .
  set BINANCE_API_KEY / BINANCE_API_SECRET
  python run_strategy.py
"""
import os
import time

from simurgh_trading_bot.strategy import SimpleStrategy

from dotenv import  load_dotenv
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

load_dotenv()


STRATEGY_PARAMS = {'strategy_id': '622937c8',
 'direction': 'bear',
 'symbol_binance': 'BTCUSDT',
 'interval': '15m',
 'take_profit_pct': 0.05,
 'stop_loss_pct': 0.05,
 'quantity': 0.1,
 'leverage': 5,
 'testnet': True,
 'trade_side': 'SELL',
 'scan_interval_seconds': 60,
 'signal_codes': ['s7']}


def _resolve_quantity(symbol: str, desired: float, testnet: bool) -> float:
    tmp = SimpleStrategy(api_key="", api_secret="", symbol=symbol, testnet=testnet)
    try:
        min_qty, step = tmp.get_symbol_lot_size(symbol)
        q = max(float(desired), float(min_qty))
        if step > 0:
            q = tmp.quantize_to_step(q, step)
        return q
    except Exception:
        return float(desired)


class GeneratedStrategy(SimpleStrategy):
    """Uses signal_codes from STRATEGY_PARAMS with the simurgh_trading_bot signal registry."""
    pass


if __name__ == "__main__":
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        raise SystemExit("Set BINANCE_API_KEY and BINANCE_API_SECRET in your environment.")

    qty = _resolve_quantity(
        STRATEGY_PARAMS["symbol_binance"],
        float(STRATEGY_PARAMS.get("quantity", 0.01)),
        bool(STRATEGY_PARAMS.get("testnet", True)),
    )
    bot = GeneratedStrategy(
        api_key=api_key,
        api_secret=api_secret,
        symbol=STRATEGY_PARAMS["symbol_binance"],
        interval=STRATEGY_PARAMS["interval"],
        quantity=qty,
        testnet=bool(STRATEGY_PARAMS.get("testnet", True)),
        take_profit_pct=float(STRATEGY_PARAMS["take_profit_pct"]),
        stop_loss_pct=float(STRATEGY_PARAMS["stop_loss_pct"]),
        trade_side=str(STRATEGY_PARAMS.get("trade_side", "BUY")),
        signal_codes=list(STRATEGY_PARAMS.get("signal_codes", [])),
    )
    try:
        bot.set_leverage(bot.symbol, int(STRATEGY_PARAMS.get("leverage", 5)))
    except Exception as exc:
        print("Leverage set skipped:", exc)

    print("Bot started — Ctrl+C to stop.")
    while True:
        try:
            bot.execute_strategy()
            time.sleep(int(STRATEGY_PARAMS.get("scan_interval_seconds", 60)))
        except Exception as exc:
            print("ERROR:", exc)
            time.sleep(10)
