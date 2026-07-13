# """Base strategy class for Simurgh trading bots."""

# from __future__ import annotations

# import time

# import pandas as pd

# from simurgh_trading_bot.broker import BinanceBroker
# from simurgh_trading_bot.feature_engineering import feature_engineer
# from simurgh_trading_bot.signals import SIGNAL_REGISTRY

# class SimpleStrategy(BinanceBroker):
#     def __init__(
#         self,
#         api_key: str,
#         api_secret: str,
#         symbol: str = "BTCUSDT",
#         interval: str = "1m",
#         quantity: float = 0.01,
#         testnet: bool = True,
#         take_profit_pct: float = 2.0,
#         stop_loss_pct: float = 2.0,
#         trade_side: str = "BUY",
#         signal_codes: list[str] | None = None,
#     ):
#         super().__init__(api_key=api_key, api_secret=api_secret, testnet=testnet)
#         self.symbol = symbol
#         self.interval = interval
#         self.quantity = quantity
#         self.take_profit_pct = float(take_profit_pct)
#         self.stop_loss_pct = float(stop_loss_pct)
#         self.trade_side = trade_side.upper() if trade_side else "BUY"
#         self.signal_codes = signal_codes or []
#         self.h_pos = 0

#     def fetch_data(self) -> pd.DataFrame:
#         candles = self.get_ohlcv(symbol=self.symbol, interval=self.interval, limit=300)
#         return pd.DataFrame(candles)

#     def compute_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
#         if ohlcv.empty:
#             return ohlcv
#         return feature_engineer(ohlcv.copy())

#     def generate_signals(self, ohlcv: pd.DataFrame) -> bool:
#         if ohlcv.empty or not self.signal_codes:
#             return False
#         for code in self.signal_codes:
#             meta = SIGNAL_REGISTRY.get(code)
#             if not meta:
#                 continue
#             try:
#                 active = bool(meta["fn"](ohlcv).iloc[-1])
#             except Exception:
#                 return False
#             if not active:
#                 return False
#         return True

#     def execute_strategy(self):
#         df = self.fetch_data()
#         df = self.compute_indicators(df)
#         entry_signal = self.generate_signals(df)
#         if self.h_pos == 0 and entry_signal:
#             current_price = float(df.iloc[-1]["close"])
#             tp_pct = self.take_profit_pct / 100.0
#             sl_pct = self.stop_loss_pct / 100.0
#             side = self.trade_side
#             if side == "BUY":
#                 tp = current_price * (1.0 + tp_pct)
#                 sl = current_price * (1.0 - sl_pct)
#             else:
#                 tp = current_price * (1.0 - tp_pct)
#                 sl = current_price * (1.0 + sl_pct)
#             print(1 if side == "BUY" else -1)
#             self.h_pos = 1
#             print(f"The current price is {current_price} tp is {tp} and the sl is {sl} the current position is {self.h_pos}")
#             self._place_simple_bracket_quiet(
#                 symbol=self.symbol,
#                 side=side,
#                 quantity=self.quantity,
#                 take_profit_price=tp,
#                 stop_loss_price=sl,
#             )
#             self.h_pos = 0
#             print(f"current position variable is {self.h_pos} hence the current position is closed")

#     def _place_simple_bracket_quiet(
#         self,
#         symbol: str,
#         side: str,
#         quantity: float,
#         take_profit_price: float,
#         stop_loss_price: float,
#         poll_interval: int = 1,
#     ) -> dict:
#         self.place_order(symbol=symbol, side=side, quantity=quantity, order_type="MARKET")
#         while True:
#             current_price = self.get_mark_price(symbol)
#             if side.upper() == "BUY":
#                 if current_price >= take_profit_price:
#                     print(f"the tp has hit take profit order confirmed current position is {self.h_pos}")
#                     return {
#                         "result": "TAKE_PROFIT",
#                         "close_order": self.close_position(symbol, "BUY", quantity),
#                     }
#                 if current_price <= stop_loss_price:
#                     print(f"the sl has hit stop loss order confirmed current position is {self.h_pos}")
#                     return {
#                         "result": "STOP_LOSS",
#                         "close_order": self.close_position(symbol, "BUY", quantity),
#                     }
#             else:
#                 if current_price <= take_profit_price:
#                     print(f"the tp has hit take profit order confirmed current position is {self.h_pos}")
#                     return {
#                         "result": "TAKE_PROFIT",
#                         "close_order": self.close_position(symbol, "SELL", quantity),
#                     }
#                 if current_price >= stop_loss_price:
#                     print(f"the sl has hit stop loss order confirmed current position is {self.h_pos}")
#                     return {
#                         "result": "STOP_LOSS",
#                         "close_order": self.close_position(symbol, "SELL", quantity),
#                     }
#             time.sleep(poll_interval)

"""Base strategy class for Simurgh trading bots."""

from __future__ import annotations

import time

import pandas as pd

from simurgh_trading_bot.broker import BinanceBroker
from simurgh_trading_bot.feature_engineering import feature_engineer
from simurgh_trading_bot.signals import SIGNAL_REGISTRY


class SimpleStrategy(BinanceBroker):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        quantity: float = 0.01,
        testnet: bool = True,
        take_profit_pct: float = 2.0,
        stop_loss_pct: float = 2.0,
        trade_side: str = "BUY",
        signal_codes: list[str] | None = None,
    ):
        super().__init__(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.symbol = symbol
        self.interval = interval
        self.quantity = quantity
        self.take_profit_pct = float(take_profit_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trade_side = trade_side.upper() if trade_side else "BUY"
        self.signal_codes = signal_codes or []
        self.h_pos = 0

        # Keeps track of every candle's open_time (ms since epoch) that
        # we've already traded on, so we never fire more than one trade
        # per candle.
        self.traded_candles: list[int] = []

    def fetch_data(self) -> pd.DataFrame:
        candles = self.get_ohlcv(symbol=self.symbol, interval=self.interval, limit=300)
        return pd.DataFrame(candles)

    def compute_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return ohlcv
        return feature_engineer(ohlcv.copy())

    def generate_signals(self, ohlcv: pd.DataFrame) -> bool:
        if ohlcv.empty or not self.signal_codes:
            return False
        for code in self.signal_codes:
            meta = SIGNAL_REGISTRY.get(code)
            if not meta:
                continue
            try:
                active = bool(meta["fn"](ohlcv).iloc[-1])
            except Exception:
                return False
            if not active:
                return False
        return True

    def _get_candle_timestamp(self, df: pd.DataFrame) -> int:
        """Return the open_time (ms since epoch, as an int) of the most
        recent candle in df. BinanceBroker.get_ohlcv() always includes
        'open_time', so this is used directly as the unique candle key.
        """
        return int(df.iloc[-1]["open_time"])

    def _already_traded_this_candle(self, candle_ts: int) -> bool:
        return candle_ts in self.traded_candles

    def _mark_candle_traded(self, candle_ts: int) -> None:
        self.traded_candles.append(candle_ts)
        # Optional: keep the list from growing unbounded over a long-running bot.
        # Comment this out if you want the full history kept instead.
        if len(self.traded_candles) > 1000:
            self.traded_candles = self.traded_candles[-1000:]

    def execute_strategy(self):
        raw_df = self.fetch_data()
        if raw_df.empty:
            return
        # Capture the candle's identity from the raw OHLCV data, before
        # compute_indicators() runs -- feature_engineer() may drop rows
        # (e.g. for rolling-window warmup) or columns, so we don't rely
        # on open_time still being present/aligned afterwards.
        candle_ts = self._get_candle_timestamp(raw_df)

        df = self.compute_indicators(raw_df)
        if df.empty:
            return

        entry_signal = self.generate_signals(df)

        if self.h_pos == 0 and entry_signal and not self._already_traded_this_candle(candle_ts):
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

            print(1 if side == "BUY" else -1)
            self.h_pos = 1
            # Record this candle as traded BEFORE the (blocking) bracket call,
            # so a re-entrant call during the same candle can't slip through.
            self._mark_candle_traded(candle_ts)
            print(f"The current price is {current_price} tp is {tp} and the sl is {sl} the current position is {self.h_pos}")
            self._place_simple_bracket_quiet(
                symbol=self.symbol,
                side=side,
                quantity=self.quantity,
                take_profit_price=tp,
                stop_loss_price=sl,
            )
            self.h_pos = 0
            print(f"current position variable is {self.h_pos} hence the current position is closed")

    def _place_simple_bracket_quiet(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        poll_interval: int = 1,
    ) -> dict:
        self.place_order(symbol=symbol, side=side, quantity=quantity, order_type="MARKET")
        while True:
            current_price = self.get_mark_price(symbol)
            if side.upper() == "BUY":
                if current_price >= take_profit_price:
                    print(f"the tp has hit take profit order confirmed current position is {self.h_pos}")
                    return {
                        "result": "TAKE_PROFIT",
                        "close_order": self.close_position(symbol, "BUY", quantity),
                    }
                if current_price <= stop_loss_price:
                    print(f"the sl has hit stop loss order confirmed current position is {self.h_pos}")
                    return {
                        "result": "STOP_LOSS",
                        "close_order": self.close_position(symbol, "BUY", quantity),
                    }
            else:
                if current_price <= take_profit_price:
                    print(f"the tp has hit take profit order confirmed current position is {self.h_pos}")
                    return {
                        "result": "TAKE_PROFIT",
                        "close_order": self.close_position(symbol, "SELL", quantity),
                    }
                if current_price >= stop_loss_price:
                    print(f"the sl has hit stop loss order confirmed current position is {self.h_pos}")
                    return {
                        "result": "STOP_LOSS",
                        "close_order": self.close_position(symbol, "SELL", quantity),
                    }
            time.sleep(poll_interval)