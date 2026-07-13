# Simurgh Trading Bot

Modular pip package for running Simurgh Capital generated strategies on Binance USDT-M futures.

## Install

```bash
cd simurgh_trading_bot
pip install -e .
```

## Quick start

1. Copy `examples/run_strategy.py` to your working folder (or download from the dashboard).
2. Set environment variables:

```bash
set BINANCE_API_KEY=your_key
set BINANCE_API_SECRET=your_secret
```

3. Edit `STRATEGY_PARAMS` in the file (symbol, interval, TP/SL, signal codes).
4. Run:

```bash
python run_strategy.py
```

## Package contents

- `simurgh_trading_bot.broker` — Binance futures wrapper
- `simurgh_trading_bot.strategy` — `SimpleStrategy` execution loop
- `simurgh_trading_bot.signals` — 130+ signal functions + registry
- `simurgh_trading_bot.feature_engineering` — OHLCV feature pipeline
