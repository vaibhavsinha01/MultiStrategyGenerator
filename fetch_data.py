import os
import time
import requests
import pandas as pd

BASE_URL = "https://api.bybit.com"

DATA_DIR = r"C:\Users\vaibh\OneDrive\Desktop\Workstation\MultiStrategyGenerator\data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# TIMEFRAMES = {
#     "1m": "1",
#     "5m": "5",
#     "15m": "15",
#     "30m": "30",
#     "1h": "60",
#     "4h": "240",
#     "1d": "D"
# }

TIMEFRAMES = {
    "4h": "240"
}

TOTAL_CANDLES = 1000
CATEGORY = "linear"

def get_klines(symbol, interval, limit=1000):

    endpoint = f"{BASE_URL}/v5/market/kline"

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(endpoint, params=params)
    r.raise_for_status()

    result = r.json()

    if result["retCode"] != 0:
        raise Exception(result["retMsg"])

    data = result["result"]["list"]

    df = pd.DataFrame(
        data,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover"
        ]
    )

    df = df[["time", "open", "high", "low", "close", "volume"]]

    df["time"] = df["time"].astype("int64")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("time").reset_index(drop=True)

    return df

def main():

    os.makedirs(DATA_DIR, exist_ok=True)

    for symbol in SYMBOLS:

        for tf_name, tf_value in TIMEFRAMES.items():

            print(f"Fetching {symbol} {tf_name}")

            df = get_klines(
                symbol=symbol,
                interval=tf_value,
                limit=TOTAL_CANDLES
            )

            filename = f"{symbol.lower()}_{tf_name}.csv"

            save_path = os.path.join(DATA_DIR, filename)

            df.to_csv(save_path, index=False)

            print(f"Saved -> {filename}")

            time.sleep(0.2)

    print("ALL DATA FETCHED SUCCESSFULLY")

if __name__ == "__main__":
    main()