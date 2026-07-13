"""Binance USDT-M futures broker wrapper."""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests


class BinanceBroker:
    LIVE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, timeout: int = 10):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.base_url = self.TESTNET_URL if testnet else self.LIVE_URL
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    @staticmethod
    def _timestamp() -> int:
        return int(time.time() * 1000)

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = urlencode(params, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ):
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        if signed:
            params["timestamp"] = self._timestamp()
            params["recvWindow"] = 5000
            params["signature"] = self._generate_signature(params)
        response = self.session.request(method=method, url=url, params=params, timeout=self.timeout)
        try:
            data = response.json()
        except Exception:
            data = response.text
        if response.status_code != 200:
            raise Exception(
                f"Binance API Error status={response.status_code} endpoint={endpoint} response={data}"
            )
        return data

    def authenticate(self):
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_wallet_balance(self, asset: str = "USDT"):
        balances = self._request("GET", "/fapi/v2/balance", signed=True)
        for item in balances:
            if item["asset"] == asset:
                return {
                    "asset": item["asset"],
                    "balance": float(item["balance"]),
                    "availableBalance": float(item["availableBalance"]),
                }
        return None

    def get_exchange_info(self) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo", signed=False)

    def get_symbol_lot_size(self, symbol: str) -> tuple[float, float]:
        info = self.get_exchange_info()
        target = symbol.upper()
        for s in info.get("symbols", []):
            if s.get("symbol") != target:
                continue
            for f in s.get("filters", []) or []:
                if f.get("filterType") == "LOT_SIZE":
                    return float(f.get("minQty", 0) or 0), float(f.get("stepSize", 0) or 0)
        raise ValueError(f"LOT_SIZE filter not found for symbol {target}")

    @staticmethod
    def quantize_to_step(quantity: float, step_size: float) -> float:
        q, step = float(quantity), float(step_size)
        if step <= 0:
            return q
        return math.floor(q / step) * step

    def get_ohlcv(self, symbol: str, interval: str = "1m", limit: int = 100):
        data = self._request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        )
        return [
            {
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": c[6],
            }
            for c in data
        ]

    def get_mark_price(self, symbol: str) -> float:
        data = self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()})
        return float(data["markPrice"])

    def set_leverage(self, symbol: str, leverage: int):
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": leverage},
            signed=True,
        )

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ):
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "reduceOnly": "true" if reduce_only else "false",
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price")
            params["price"] = price
            params["timeInForce"] = "GTC"
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def close_position(self, symbol: str, original_side: str, quantity: float):
        closing_side = "SELL" if original_side.upper() == "BUY" else "BUY"
        return self.place_order(
            symbol=symbol,
            side=closing_side,
            quantity=quantity,
            order_type="MARKET",
            reduce_only=True,
        )

