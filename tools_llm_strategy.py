import json
import math
import pprint
import re
import time
import hmac
import hashlib
from dataclasses import asdict, dataclass, field
from urllib.parse import urlencode
from typing import Any, Dict, Optional

import requests


class BinanceBroker:
    """
    Lightweight Binance USDT-M Futures Wrapper

    Features
    --------
    - Authentication
    - Wallet balance
    - OHLCV data
    - Mark price
    - Set leverage
    - Place market/limit orders
    - Close positions
    - Software-managed bracket orders
    - Testnet + Live support
    """

    LIVE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        timeout: int = 10
    ):

        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout

        self.base_url = (
            self.TESTNET_URL if testnet
            else self.LIVE_URL
        )

        self.session = requests.Session()

        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key
        })

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    @staticmethod
    def _timestamp() -> int:
        return int(time.time() * 1000)

    def _generate_signature(self, params: Dict[str, Any]) -> str:

        query_string = urlencode(params, doseq=True)

        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ):

        url = f"{self.base_url}{endpoint}"

        params = params or {}

        if signed:
            params["timestamp"] = self._timestamp()
            params["recvWindow"] = 5000
            params["signature"] = self._generate_signature(params)

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            timeout=self.timeout
        )

        try:
            data = response.json()
        except Exception:
            data = response.text

        if response.status_code != 200:
            raise Exception(
                f"""
Binance API Error
-----------------
Status Code : {response.status_code}
Endpoint    : {endpoint}
Response    : {data}
"""
            )

        return data

    # =========================================================
    # AUTH
    # =========================================================

    def authenticate(self):
        """
        Verify API credentials.
        """

        return self._request(
            method="GET",
            endpoint="/fapi/v2/account",
            signed=True
        )

    # =========================================================
    # ACCOUNT
    # =========================================================

    def get_wallet_balance(self, asset: str = "USDT"):
        """
        Get wallet balance.
        """

        balances = self._request(
            method="GET",
            endpoint="/fapi/v2/balance",
            signed=True
        )

        for item in balances:

            if item["asset"] == asset:

                return {
                    "asset": item["asset"],
                    "balance": float(item["balance"]),
                    "availableBalance": float(item["availableBalance"])
                }

        return None

    # =========================================================
    # MARKET DATA
    # =========================================================

    def get_exchange_info(self) -> dict:
        """
        Public endpoint: exchange metadata incl. filters (minQty/stepSize).
        """
        return self._request(
            method="GET",
            endpoint="/fapi/v1/exchangeInfo",
            signed=False,
        )

    def get_symbol_lot_size(self, symbol: str) -> tuple[float, float]:
        """
        Returns (min_qty, step_size) for the given futures symbol.
        """
        info = self.get_exchange_info()
        symbols = info.get("symbols", []) if isinstance(info, dict) else []
        target = symbol.upper()
        for s in symbols:
            if s.get("symbol") != target:
                continue
            for f in s.get("filters", []) or []:
                if f.get("filterType") == "LOT_SIZE":
                    min_qty = float(f.get("minQty", 0) or 0)
                    step = float(f.get("stepSize", 0) or 0)
                    return min_qty, step
        raise ValueError(f"LOT_SIZE filter not found for symbol {target}")

    @staticmethod
    def quantize_to_step(quantity: float, step_size: float) -> float:
        """
        Floors quantity down to the nearest multiple of step_size.
        """
        q = float(quantity)
        step = float(step_size)
        if step <= 0:
            return q
        return math.floor(q / step) * step

    def resolve_min_quantity(self, symbol: str, fallback: float = 0.0) -> float:
        """
        Best-effort min quantity:
        - Uses exchangeInfo LOT_SIZE minQty if reachable
        - Falls back to provided fallback
        """
        try:
            min_qty, _step = self.get_symbol_lot_size(symbol)
            return float(min_qty)
        except Exception:
            return float(fallback)

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100
    ):
        """
        Fetch OHLCV candles.
        """

        data = self._request(
            method="GET",
            endpoint="/fapi/v1/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": limit
            }
        )

        candles = []

        for candle in data:

            candles.append({
                "open_time": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "close_time": candle[6]
            })

        return candles

    def get_mark_price(self, symbol: str):
        """
        Get current mark price.
        """

        data = self._request(
            method="GET",
            endpoint="/fapi/v1/premiumIndex",
            params={
                "symbol": symbol.upper()
            }
        )

        return float(data["markPrice"])

    # =========================================================
    # LEVERAGE
    # =========================================================

    def set_leverage(self, symbol: str, leverage: int):
        """
        Set leverage.
        """

        return self._request(
            method="POST",
            endpoint="/fapi/v1/leverage",
            params={
                "symbol": symbol.upper(),
                "leverage": leverage
            },
            signed=True
        )

    # =========================================================
    # ORDERS
    # =========================================================

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        reduce_only: bool = False
    ):
        """
        Place MARKET or LIMIT order.
        """

        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "reduceOnly": "true" if reduce_only else "false"
        }

        if order_type.upper() == "LIMIT":

            if price is None:
                raise ValueError("LIMIT order requires price")

            params["price"] = price
            params["timeInForce"] = "GTC"

        return self._request(
            method="POST",
            endpoint="/fapi/v1/order",
            params=params,
            signed=True
        )

    def close_position(
        self,
        symbol: str,
        original_side: str,
        quantity: float
    ):
        """
        Close existing position.
        """

        closing_side = (
            "SELL"
            if original_side.upper() == "BUY"
            else "BUY"
        )

        return self.place_order(
            symbol=symbol,
            side=closing_side,
            quantity=quantity,
            order_type="MARKET",
            reduce_only=True
        )

    # =========================================================
    # SOFTWARE BRACKET ORDER
    # =========================================================

    def place_simple_bracket(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        poll_interval: int = 1
    ):
        """
        Software-managed bracket order.

        This avoids Binance Futures Testnet TP/SL issues.
        """

        entry = self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET"
        )

        print("\nEntry Order Placed")
        print(entry)

        while True:

            current_price = self.get_mark_price(symbol)

            print(f"Current Price: {current_price}")

            # LONG POSITION
            if side.upper() == "BUY":

                if current_price >= take_profit_price:

                    print("Take Profit Hit")

                    close = self.close_position(
                        symbol=symbol,
                        original_side="BUY",
                        quantity=quantity
                    )

                    return {
                        "result": "TAKE_PROFIT",
                        "close_order": close
                    }

                if current_price <= stop_loss_price:

                    print("Stop Loss Hit")

                    close = self.close_position(
                        symbol=symbol,
                        original_side="BUY",
                        quantity=quantity
                    )

                    return {
                        "result": "STOP_LOSS",
                        "close_order": close
                    }

            # SHORT POSITION
            else:

                if current_price <= take_profit_price:

                    print("Take Profit Hit")

                    close = self.close_position(
                        symbol=symbol,
                        original_side="SELL",
                        quantity=quantity
                    )

                    return {
                        "result": "TAKE_PROFIT",
                        "close_order": close
                    }

                if current_price >= stop_loss_price:

                    print("Stop Loss Hit")

                    close = self.close_position(
                        symbol=symbol,
                        original_side="SELL",
                        quantity=quantity
                    )

                    return {
                        "result": "STOP_LOSS",
                        "close_order": close
                    }

            time.sleep(poll_interval)

# # =============================================================
# # EXAMPLE USAGE
# # =============================================================

# if __name__ == "__main__":

#     API_KEY = "mc7Hi9vZgN4vHgRpysUhpFCGq69UMBM6IMQwxcI9BNISV9okFRWUpiIlkXtlnmGt"
#     API_SECRET = "456tm0qlmnsEV6w1CWf1kK1PROW1khGPj9eBsaBGsOqlbkctc2QZSnTdnVqtyxpl"

#     broker = BinanceBroker(
#         api_key=API_KEY,
#         api_secret=API_SECRET,
#         testnet=True
#     )

#     # AUTHENTICATION
#     account = broker.authenticate()
#     print("Authenticated")

#     # WALLET
#     wallet = broker.get_wallet_balance()
#     print("Wallet:", wallet)

#     # LEVERAGE
#     leverage = broker.set_leverage("BTCUSDT", 10)
#     print("Leverage:", leverage)

#     # OHLCV
#     candles = broker.get_ohlcv(
#         symbol="BTCUSDT",
#         interval="1m",
#         limit=5
#     )

#     print("\nCandles:")
#     for candle in candles:
#         print(candle)

#     # CURRENT PRICE
#     current_price = broker.get_mark_price("BTCUSDT")
#     print(f"\nCurrent BTC Price: {current_price}")

#     # SIMPLE SOFTWARE BRACKET
#     result = broker.place_simple_bracket(
#         symbol="BTCUSDT",
#         side="BUY",
#         quantity=0.01,
#         take_profit_price=current_price + 10,
#         stop_loss_price=current_price - 10
#     )
#     # result = broker.close_position(symbol="BTCUSDT",original_side="BUY",quantity=0.01)

#     print("\nBracket Result:")
#     print(result)

class Signals:
    def __init__(self):
        self.signal_data = {
            "s1": "ema9 above ema21",
            "s2": "ema21 above ema50",
            "s3": "ema50 above ema200",
            "s4": "price above vwap",
            "s5": "macd above signal",
            "s6": "macd histogram rising",
            "s7": "supertrend bullish",
            "s8": "price above hullma20",
            "s9": "wavetrend wt1 above wt2",
            "s10": "ut bot bullish",
            "s11": "range filter bullish",
            "s12": "adx bullish trend",
            "s13": "rsi14 below 35",
            "s14": "rsi14 above 50",
            "s15": "rsi14 between 55 and 75",
            "s16": "stochastic k below 25",
            "s17": "stochastic bullish crossover",
            "s18": "stoch rsi below 20",
            "s19": "cci below minus 100",
            "s20": "williams r below minus 80",
            "s21": "wavetrend bullish crossover",
            "s22": "wavetrend oversold",
            "s23": "ultimate oscillator below 30",
            "s24": "tsi positive",
            "s25": "bollinger band oversold",
            "s26": "bollinger squeeze active",
            "s27": "atr volatility expansion",
            "s28": "price above keltner upper band",
            "s29": "wvf spike detected",
            "s30": "high volume confirmation",
            "s31": "mfi below 25",
            "s32": "cmf positive",
            "s33": "obv rising",
            "s34": "delta normalized positive",
            "s35": "price above donchian middle",
            "s36": "price above fibonacci 0.618",
            "s37": "inside bar detected",
            "s38": "inside bar bullish breakout",
            "s39": "range filter buy signal",
            "s40": "ut bot buy signal",
            "s41": "heikin ashi bullish candle",
            "s42": "two bullish heikin ashi candles",
            "s43": "price above tema21",
            "s44": "vwma20 above sma20",
            "s45": "gainzy strong bullish",
            "s46": "gainzy bullish regime",
            "s47": "gainzy bullish flip",
            "s48": "gainzy momentum zone",
            "s49": "gainzy strong bearish",
            "s50": "gainzy bearish regime",
            "s51": "gainzy bearish flip",
            "s52": "gainzy not bearish",
            "s53": "alphatrend bullish regime",
            "s54": "alphatrend buy crossover",
            "s55": "alphatrend confirmed buy",
            "s56": "alphatrend bearish regime",
            "s57": "alphatrend sell crossover",
            "s58": "chandelier bullish regime",
            "s59": "chandelier buy signal",
            "s60": "price above chandelier stop",
            "s61": "chandelier bearish regime",
            "s62": "chandelier sell signal",
            "s63": "qqe long signal",
            "s64": "qqe bullish regime",
            "s65": "qqe rsi above 50",
            "s66": "qqe short signal",
            "s67": "qqe bearish regime",
            "s68": "halftrend buy signal",
            "s69": "halftrend bullish regime",
            "s70": "price above halftrend line",
            "s71": "halftrend sell signal",
            "s72": "halftrend bearish regime",
            "s73": "rmi buy signal",
            "s74": "rmi bullish regime",
            "s75": "rmi momentum bullish",
            "s76": "rmi sell signal",
            "s77": "rmi bearish regime",
            "s78": "bullish fair value gap",
            "s79": "price inside bullish fvg",
            "s80": "bearish fair value gap",
            "s81": "price inside bearish fvg",
            "s82": "no bearish fvg overhead",
            "s83": "liquidity bullish sweep",
            "s84": "liquidity bullish breakout",
            "s85": "liquidity bearish sweep",
            "s86": "liquidity bearish breakout",
            "s87": "bullish order block created",
            "s88": "price inside bullish order block",
            "s89": "bearish order block created",
            "s90": "price inside bearish order block",
            "s91": "no bearish order block overhead",
            "s92": "smc bullish bos",
            "s93": "smc bullish choch",
            "s94": "smc bullish regime",
            "s95": "smc internal bullish choch",
            "s96": "smc internal bullish regime",
            "s97": "smc bearish bos",
            "s98": "smc bearish choch",
            "s99": "smc bearish regime",
            "s100": "smc internal bearish choch",
            "s101": "smc internal bearish regime"
        }

# if __name__ == "__main__":
#     signals = Signals()

#     print(signals.signal_data["s1"])
#     print(signals.signal_data["s54"])
#     print(signals.signal_data["s101"])

# from broker import *
import pandas as pd
import time


class SimpleStrategy(BinanceBroker):

    def __init__(
        self,
        api_key,
        api_secret,
        symbol="BTCUSDT",
        interval="1m",
        quantity=0.01,
        testnet=True,
        take_profit_pct: float = 2.0,
        stop_loss_pct: float = 2.0,
        trade_side: str = "BUY",
    ):

        super().__init__(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )

        self.symbol = symbol
        self.interval = interval
        self.quantity = quantity
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trade_side = trade_side.upper() if trade_side else "BUY"

        # 0 = no position
        # 1 = active position
        self.h_pos = 0

    # =========================================================
    # FETCH DATA
    # =========================================================

    def fetch_data(self):

        candles = self.get_ohlcv(
            symbol=self.symbol,
            interval=self.interval,
            limit=100
        )

        df = pd.DataFrame(candles)

        return df

    # =========================================================
    # INDICATORS
    # =========================================================

    def compute_indicators(self, df):

        # RSI-14
        delta = df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        df["rsi_14"] = 100 - (100 / (1 + rs))

        # ATR-14
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat(
            [high_low, high_close, low_close],
            axis=1
        ).max(axis=1)

        df["atr_14"] = tr.rolling(14).mean()

        return df

    # =========================================================
    # SIGNALS
    # =========================================================

    def generate_signals(self, df):

        latest = df.iloc[-1]

        # s13 → RSI < 35
        s13 = latest["rsi_14"] < 35

        # s27 → ATR > ATR SMA20
        s27 = (
            latest["atr_14"] >
            df["atr_14"].rolling(20).mean().iloc[-1]
        )

        return s13 and s27

    # =========================================================
    # EXECUTION
    # =========================================================

    def execute_strategy(self):

        df = self.fetch_data()

        df = self.compute_indicators(df)

        entry_signal = self.generate_signals(df)

        # ENTER TRADE
        if self.h_pos == 0 and entry_signal:

            current_price = float(df.iloc[-1]["close"])
            tp_pct = self.take_profit_pct / 100.0
            sl_pct = self.stop_loss_pct / 100.0
            side = self.trade_side

            if side == "BUY":
                tp = current_price * (1.0 + tp_pct)
                sl = current_price * (1.0 - sl_pct)
            else:
                tp = current_price * (1.0 - tp_pct)
                sl = current_price * (1.0 + sl_pct)

            # Print only trade direction signal: 1=long, -1=short
            print(1 if side == "BUY" else -1)

            self.h_pos = 1

            result = self._place_simple_bracket_quiet(
                symbol=self.symbol,
                side=side,
                quantity=self.quantity,
                take_profit_price=tp,
                stop_loss_price=sl,
                poll_interval=1,
            )

            self.h_pos = 0

    def _place_simple_bracket_quiet(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        poll_interval: int = 1,
    ) -> dict:
        """
        Software bracket without continuous price printing.
        Mirrors BinanceBroker.place_simple_bracket behaviour but stays quiet.
        """
        self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type="MARKET",
        )

        while True:
            current_price = self.get_mark_price(symbol)

            if side.upper() == "BUY":
                if current_price >= take_profit_price:
                    close = self.close_position(
                        symbol=symbol,
                        original_side="BUY",
                        quantity=quantity,
                    )
                    return {"result": "TAKE_PROFIT", "close_order": close}
                if current_price <= stop_loss_price:
                    close = self.close_position(
                        symbol=symbol,
                        original_side="BUY",
                        quantity=quantity,
                    )
                    return {"result": "STOP_LOSS", "close_order": close}
            else:
                if current_price <= take_profit_price:
                    close = self.close_position(
                        symbol=symbol,
                        original_side="SELL",
                        quantity=quantity,
                    )
                    return {"result": "TAKE_PROFIT", "close_order": close}
                if current_price >= stop_loss_price:
                    close = self.close_position(
                        symbol=symbol,
                        original_side="SELL",
                        quantity=quantity,
                    )
                    return {"result": "STOP_LOSS", "close_order": close}

            time.sleep(poll_interval)


# if __name__ == "__main__":

#     strategy = SimpleStrategy(
#         api_key="YOUR_API_KEY",
#         api_secret="YOUR_API_SECRET",
#         symbol="BTCUSDT",
#         interval="1m",
#         quantity=0.01,
#         testnet=True
#     )
#     print(f"Trading bot has started.")
#     while True:

#         try:
#             strategy.execute_strategy()

#             time.sleep(60)

#         except Exception as e:

#             print(f"\nERROR: {e}")

#             time.sleep(10)


# ══════════════════════════════════════════════════════════════════════════════
#  Dashboard export: runtime params + LLM-generated runnable main.py
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyRuntimeParams:
    """
    Execution-facing parameters for a Binance futures bot derived from a
    dashboard/backtest strategy row. Percent fields are whole percents (e.g. 2.0 = 2%).
    """

    strategy_id: str
    direction: str
    symbol_binance: str
    interval: str
    take_profit_pct: float
    stop_loss_pct: float
    quantity: float = 0.01
    leverage: int = 5
    testnet: bool = True
    trade_side: str = "BUY"
    scan_interval_seconds: int = 60
    signal_codes: list[str] = field(default_factory=list)

    def to_code_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["signal_codes"] = data.get("signal_codes") or []
        return data


def binance_symbol_from_dashboard_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        return s.replace("USD", "") + "USDT"
    if not s.endswith("USDT"):
        return f"{s}USDT"
    return s


def binance_kline_interval(timeframe_display: str) -> str:
    t = timeframe_display.strip().upper()
    if t in ("D", "1D"):
        return "1d"
    # Binance futures does not accept "60m"/"240m" style intervals; convert minutes>=60 to hours.
    m = re.fullmatch(r"(\d+)([MH])", t)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "M":
            if n >= 60 and n % 60 == 0:
                return f"{n // 60}h"
            return f"{n}m"
        return f"{n}h"
    return timeframe_display.strip().lower()


def trade_side_from_direction(direction: str) -> str:
    d = (direction or "bull").strip().lower()
    return "BUY" if d == "bull" else "SELL"


def strategy_runtime_params_from_document_payload(payload: dict) -> StrategyRuntimeParams:
    strat = payload.get("strategy") or {}
    sym = binance_symbol_from_dashboard_symbol(str(payload.get("symbol", "ethusd")))
    interval = binance_kline_interval(str(payload.get("timeframe", "15m")))
    tp_dec = float(strat.get("tp", 0) or 0)
    sl_dec = float(strat.get("sl", 0) or 0)
    tp_pct = tp_dec * 100.0 if tp_dec > 0 else 2.0
    sl_pct = sl_dec * 100.0 if sl_dec > 0 else 2.0
    direction = str(strat.get("direction", "bull"))
    signal_codes = [p.strip() for p in str(strat.get("signals", "") or "").split("|") if p.strip()]
    return StrategyRuntimeParams(
        strategy_id=str(strat.get("id", "unknown")),
        direction=direction,
        symbol_binance=sym,
        interval=interval,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        trade_side=trade_side_from_direction(direction),
        signal_codes=signal_codes,
    )


def signals_text_map_for_codes(signals_pipe: str, catalog: Optional[dict[str, str]] = None) -> dict[str, str]:
    catalog = catalog or Signals().signal_data
    out: dict[str, str] = {}
    for key in [p.strip() for p in signals_pipe.split("|") if p.strip()]:
        out[key] = catalog.get(key, "")
    return out


class StrategyCodegenContext:
    """
    Collects runtime parameters and LLM context for exporting a user `main.py`
    that subclasses SimpleStrategy without touching BinanceBroker.
    """

    def __init__(self, document_payload: dict):
        self.payload = document_payload

    def runtime_params(self) -> StrategyRuntimeParams:
        return strategy_runtime_params_from_document_payload(self.payload)

    def llm_context(self) -> dict[str, Any]:
        return build_strategy_codegen_prompt_payload(self.payload)

    def main_py_source(self, *, use_llm: bool = True) -> str:
        return generate_strategy_main_py_source(self.payload, use_llm=use_llm)


def build_strategy_codegen_prompt_payload(payload: dict) -> dict[str, Any]:
    params = strategy_runtime_params_from_document_payload(payload)
    strat = payload.get("strategy") or {}
    sigs = str(strat.get("signals", "") or "")
    return {
        "runtime": params.to_code_dict(),
        "strategyId": strat.get("id"),
        "direction": strat.get("direction"),
        "signalsPipe": sigs,
        "signalHints": signals_text_map_for_codes(sigs),
        "train": strat.get("train"),
        "test": strat.get("test"),
        "score": strat.get("score"),
    }


def _strip_markdown_python_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)\n```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _build_strategy_code_prompt(ctx: dict[str, Any]) -> str:
    return (
        "You are a Python quant engineer. Output ONE complete Python source file for a small Binance USDT-M "
        "futures trading script.\n\n"
        "Hard rules:\n"
        "- Import ONLY `SimpleStrategy` from `simurgh_trading_bot`. Do NOT copy or redefine BinanceBroker.\n"
        "- Define `STRATEGY_PARAMS` as a Python dict literal whose keys and values match the JSON object "
        "`runtime` in the context below (include signal_codes list).\n"
        "- The script MUST be runnable after `pip install -e simurgh_trading_bot`.\n"
        "- Define `class GeneratedStrategy(SimpleStrategy):` only if custom logic is needed; otherwise pass "
        "signal_codes from STRATEGY_PARAMS to SimpleStrategy.\n"
        "- In `if __name__ == '__main__':`, read `BINANCE_API_KEY` and `BINANCE_API_SECRET` from `os.environ`.\n"
        "- Resolve minQty/stepSize via exchangeInfo before starting the loop.\n"
        "- Loop: `while True:` call `execute_strategy()`, then sleep scan_interval_seconds.\n\n"
        "Return ONLY the Python file inside one markdown code fence: ```python ... ```.\n\n"
        f"Context JSON:\n{json.dumps(ctx, indent=2)}"
    )


def _python_literal(value: Any) -> str:
    """Render Python dict/list with True/False/None (not JSON true/false/null)."""
    return pprint.pformat(value, width=100, sort_dicts=False)


def _fallback_main_py_source(ctx: dict[str, Any]) -> str:
    r = ctx["runtime"]
    blob = _python_literal(r)
    return f'''"""
Auto-generated Simurgh Capital trading bot.
Strategy id: {r["strategy_id"]}

Setup:
  cd simurgh_trading_bot && pip install -e .
  set BINANCE_API_KEY / BINANCE_API_SECRET
  python run_strategy.py
"""
import os
import time

from simurgh_trading_bot.strategy import SimpleStrategy


STRATEGY_PARAMS = {blob}


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
'''


def generate_strategy_main_py_source(payload: dict, *, use_llm: bool = True) -> str:
    """
    Produce a downloadable `main.py`-style script: LLM-authored subclass of SimpleStrategy,
    or a deterministic fallback if the model fails validation.
    """
    ctx = build_strategy_codegen_prompt_payload(payload)
    if not use_llm:
        return _fallback_main_py_source(ctx)
    try:
        from llm_generator import _call_gemini

        raw = _call_gemini(_build_strategy_code_prompt(ctx), max_output_tokens=8192)
        code = _strip_markdown_python_fence(raw)
        if "SimpleStrategy" not in code or "STRATEGY_PARAMS" not in code:
            return _fallback_main_py_source(ctx)
        return code
    except Exception:
        return _fallback_main_py_source(ctx)
