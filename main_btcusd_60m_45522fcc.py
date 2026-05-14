import os
import time
import pandas as pd
from tools_llm_strategy import SimpleStrategy

# Define STRATEGY_PARAMS as a Python dict literal
STRATEGY_PARAMS = {
    "strategy_id": "45522fcc",
    "direction": "bear",
    "symbol_binance": "BTCUSDT",
    "interval": "60m",
    "take_profit_pct": 3.21299454,
    "stop_loss_pct": 1.96032373,
    "quantity": 0.01,
    "leverage": 5,
    "testnet": True,
    "trade_side": "SELL",
    "scan_interval_seconds": 60
}

class GeneratedStrategy(SimpleStrategy):
    def compute_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators based on the provided OHLCV data.
        """
        if ohlcv.empty:
            return ohlcv

        # Ensure OHLCV columns are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in ohlcv.columns:
                ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
        
        # Drop rows with NaN values introduced by to_numeric, especially at the start
        ohlcv.dropna(subset=['close', 'high', 'low'], inplace=True)
        if ohlcv.empty:
            return ohlcv

        # s50: "gainzy bearish regime"
        # Placeholder: A bearish regime can be indicated by price trading below a short-term moving average.
        # Using a 20-period Simple Moving Average (SMA) as a proxy.
        ohlcv['SMA_20'] = ohlcv['close'].rolling(window=20).mean()
        ohlcv['s50_bearish'] = ohlcv['close'] < ohlcv['SMA_20']

        # s99: "smc bearish regime"
        # Placeholder: A more significant bearish regime can be indicated by price trading below a longer-term moving average.
        # Using a 50-period Simple Moving Average (SMA) as a proxy.
        ohlcv['SMA_50'] = ohlcv['close'].rolling(window=50).mean()
        ohlcv['s99_bearish'] = ohlcv['close'] < ohlcv['SMA_50']

        # s23: "ultimate oscillator below 30"
        # Ultimate Oscillator (UO) calculation with periods 7, 14, 28
        period1, period2, period3 = 7, 14, 28

        # Calculate True Range (TR) and Buying Pressure (BP)
        # Need previous close for TR and BP calculation
        ohlcv['prev_close'] = ohlcv['close'].shift(1)

        # TR = max(high, prev_close) - min(low, prev_close)
        # Handle the first row where prev_close is NaN
        ohlcv['TR'] = (ohlcv[['high', 'prev_close']].max(axis=1) -
                       ohlcv[['low', 'prev_close']].min(axis=1)).fillna(ohlcv['high'] - ohlcv['low'])

        # BP = close - min(low, prev_close)
        # Handle the first row where prev_close is NaN
        ohlcv['BP'] = (ohlcv['close'] - ohlcv[['low', 'prev_close']].min(axis=1)).fillna(0) # BP is 0 if prev_close is NaN

        # Calculate Sum of BP and TR for each period
        sum_bp1 = ohlcv['BP'].rolling(window=period1).sum()
        sum_tr1 = ohlcv['TR'].rolling(window=period1).sum()
        sum_bp2 = ohlcv['BP'].rolling(window=period2).sum()
        sum_tr2 = ohlcv['TR'].rolling(window=period2).sum()
        sum_bp3 = ohlcv['BP'].rolling(window=period3).sum()
        sum_tr3 = ohlcv['TR'].rolling(window=period3).sum()

        # Calculate Average Ratios, handling potential division by zero
        # Replace 0 in sum_tr with NaN to avoid division by zero, then fillna(0) for ratios
        avg_ratio1 = (sum_bp1 / sum_tr1.replace(0, pd.NA)).fillna(0)
        avg_ratio2 = (sum_bp2 / sum_tr2.replace(0, pd.NA)).fillna(0)
        avg_ratio3 = (sum_bp3 / sum_tr3.replace(0, pd.NA)).fillna(0)

        # Calculate Ultimate Oscillator
        ohlcv['UO'] = 100 * ((4 * avg_ratio1) + (2 * avg_ratio2) + avg_ratio3) / (4 + 2 + 1)
        ohlcv['s23_uo_below_30'] = ohlcv['UO'] < 30

        # Drop intermediate columns to keep the DataFrame clean
        ohlcv.drop(columns=['SMA_20', 'SMA_50', 'prev_close', 'TR', 'BP'], errors='ignore', inplace=True)

        return ohlcv

    def generate_signals(self, ohlcv: pd.DataFrame) -> str | None:
        """
        Generates trading signals based on the computed indicators.
        The signalsPipe "s50|s99|s23" implies an OR logic for bearish signals.
        """
        if ohlcv.empty or len(ohlcv) < 1:
            return None

        # Get the latest row of indicators
        latest_indicators = ohlcv.iloc[-1]

        # Check for bearish conditions based on the signal hints
        signal_s50_bearish = latest_indicators.get('s50_bearish', False)
        signal_s99_bearish = latest_indicators.get('s99_bearish', False)
        signal_s23_uo_below_30 = latest_indicators.get('s23_uo_below_30', False)

        # If any of the bearish conditions are met, generate a SELL signal
        # as per STRATEGY_PARAMS['trade_side'] = "SELL"
        if signal_s50_bearish or signal_s99_bearish or signal_s23_uo_below_30:
            return "SELL"
        
        return None

if __name__ == '__main__':
    # Read API keys from environment variables
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY and BINANCE_API_SECRET environment variables must be set.")
        exit(1)

    # Instantiate the strategy with parameters from STRATEGY_PARAMS
    strategy = GeneratedStrategy(
        api_key=api_key,
        api_secret=api_secret,
        symbol=STRATEGY_PARAMS['symbol_binance'],
        interval=STRATEGY_PARAMS['interval'],
        quantity=STRATEGY_PARAMS['quantity'],
        testnet=STRATEGY_PARAMS['testnet'],
        take_profit_pct=STRATEGY_PARAMS['take_profit_pct'],
        stop_loss_pct=STRATEGY_PARAMS['stop_loss_pct'],
        trade_side=STRATEGY_PARAMS['trade_side']
    )

    # Optionally set leverage for the trading symbol
    try:
        leverage = STRATEGY_PARAMS.get('leverage')
        if leverage:
            print(f"Attempting to set leverage for {STRATEGY_PARAMS['symbol_binance']} to {leverage}x...")
            strategy.set_leverage(STRATEGY_PARAMS['symbol_binance'], leverage)
            print(f"Leverage set to {leverage}x for {STRATEGY_PARAMS['symbol_binance']}.")
    except Exception as e:
        print(f"Could not set leverage for {STRATEGY_PARAMS['symbol_binance']}: {e}")

    # Main trading loop
    print(f"Starting trading loop for {STRATEGY_PARAMS['symbol_binance']} with interval {STRATEGY_PARAMS['interval']}...")
    while True:
        try:
            strategy.execute_strategy()
        except Exception as e:
            print(f"An error occurred during strategy execution: {e}")
            # In a production environment, consider more robust error handling,
            # such as logging, notifications, or specific retry logic.
        finally:
            scan_interval = STRATEGY_PARAMS['scan_interval_seconds']
            print(f"Sleeping for {scan_interval} seconds before next scan...")
            time.sleep(scan_interval)