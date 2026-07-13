"""
Simurgh Capital — runnable strategy template.

1. pip install -e .   (from simurgh_trading_bot folder)
2. Set BINANCE_API_KEY and BINANCE_API_SECRET
3. Fill STRATEGY_PARAMS below (or paste from dashboard export)
4. python run_strategy.py
"""
import os
import time

from dotenv import load_dotenv

from simurgh_trading_bot.strategy import SimpleStrategy

load_dotenv()

# ── Edit these values (dashboard export fills this block) ─────────────────────
STRATEGY_PARAMS = {
    "strategy_id": "YOUR_STRATEGY_ID",
    "direction": "bull",
    "symbol_binance": "BTCUSDT",
    "interval": "1h",
    "take_profit_pct": 2.0,
    "stop_loss_pct": 2.0,
    "quantity": 0.01,
    "leverage": 5,
    "testnet": True,
    "trade_side": "BUY",
    "scan_interval_seconds": 60,
    "signal_codes": ["s13", "s27"],
}


class GeneratedStrategy(SimpleStrategy):
    """Override only if you need custom indicator/signal logic."""

    pass


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

    print(f"Bot started — {STRATEGY_PARAMS['symbol_binance']} {STRATEGY_PARAMS['interval']}")
    while True:
        try:
            bot.execute_strategy()
            time.sleep(int(STRATEGY_PARAMS.get("scan_interval_seconds", 60)))
        except Exception as exc:
            print("ERROR:", exc)
            time.sleep(10)
