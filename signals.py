"""
signals.py
──────────
Each signal is a pure function:
    f(df: pd.DataFrame) -> pd.Series[bool]

Columns expected in df come from the feature-engineering pipeline.
Signal names are kept short (s1..sN) so they compose cleanly
in strategy dictionaries.
"""

import pandas as pd
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crosses above b."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crosses below b."""
    return (a < b) & (a.shift(1) >= b.shift(1))


# ── trend / ma signals ────────────────────────────────────────────────────────

def s1(df: pd.DataFrame) -> pd.Series:
    """EMA-9 above EMA-21  →  short-term uptrend."""
    return df["ema_9"] > df["ema_21"]


def s2(df: pd.DataFrame) -> pd.Series:
    """EMA-21 above EMA-50  →  medium-term uptrend."""
    return df["ema_21"] > df["ema_50"]


def s3(df: pd.DataFrame) -> pd.Series:
    """EMA-50 above EMA-200  →  long-term uptrend (golden cross regime)."""
    return df["ema_50"] > df["ema_200"]


def s4(df: pd.DataFrame) -> pd.Series:
    """Price above VWAP  →  bullish intraday bias."""
    return df["close"] > df["vwap"]


def s5(df: pd.DataFrame) -> pd.Series:
    """MACD line above signal line  →  bullish momentum."""
    return df["macd"] > df["macd_signal"]


def s6(df: pd.DataFrame) -> pd.Series:
    """MACD histogram positive and rising  →  accelerating bullish momentum."""
    return (df["macd_hist"] > 0) & (df["macd_hist"] > df["macd_hist"].shift(1))


def s7(df: pd.DataFrame) -> pd.Series:
    """Supertrend direction bullish (+1)."""
    return df["supertrend_direction"] == 1


def s8(df: pd.DataFrame) -> pd.Series:
    """Price above Hull-MA-20  →  dynamic trend filter."""
    return df["close"] > df["hull_ma_20"]


def s9(df: pd.DataFrame) -> pd.Series:
    """WaveTrend wt1 above wt2  →  WT bullish cross."""
    return df["wt1"] > df["wt2"]


def s10(df: pd.DataFrame) -> pd.Series:
    """UT Bot bar_buy flag  →  ATR trailing stop bullish."""
    return df["ut_bar_buy"].astype(bool)


def s11(df: pd.DataFrame) -> pd.Series:
    """Range Filter trend bullish (+1)."""
    return df["RF_Trend"] == 1


def s12(df: pd.DataFrame) -> pd.Series:
    """ADX > 20 AND +DI > -DI  →  strong bullish trend."""
    return (df["adx"] > 20) & (df["adx_pos"] > df["adx_neg"])


# ── momentum / oscillator signals ────────────────────────────────────────────

def s13(df: pd.DataFrame) -> pd.Series:
    """RSI-14 below 35  →  oversold, potential reversal long."""
    return df["rsi_14"] < 35


def s14(df: pd.DataFrame) -> pd.Series:
    """RSI-14 above 50  →  bullish momentum regime."""
    return df["rsi_14"] > 50


def s15(df: pd.DataFrame) -> pd.Series:
    """RSI-14 above 55 and below 75  →  trend continuation zone."""
    return (df["rsi_14"] > 55) & (df["rsi_14"] < 75)


def s16(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K below 25  →  oversold."""
    return df["stoch_k"] < 25


def s17(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K crosses above %D from below 30  →  stoch buy."""
    return _cross_above(df["stoch_k"], df["stoch_d"]) & (df["stoch_d"] < 30)


def s18(df: pd.DataFrame) -> pd.Series:
    """Stochastic RSI %K below 20  →  deeply oversold."""
    return df["stoch_rsi_k"] < 20


def s19(df: pd.DataFrame) -> pd.Series:
    """CCI below -100  →  oversold on CCI."""
    return df["cci"] < -100


def s20(df: pd.DataFrame) -> pd.Series:
    """Williams %R below -80  →  oversold."""
    return df["williams_r"] < -80


def s21(df: pd.DataFrame) -> pd.Series:
    """WaveTrend crosses up (wt_cross_up)  →  WT buy signal."""
    return df["wt_cross_up"].astype(bool)


def s22(df: pd.DataFrame) -> pd.Series:
    """WaveTrend oversold zone (wt_oversold)."""
    return df["wt_oversold"].astype(bool)


def s23(df: pd.DataFrame) -> pd.Series:
    """Ultimate Oscillator below 30  →  oversold pressure."""
    return df["ultimate_osc"] < 30


def s24(df: pd.DataFrame) -> pd.Series:
    """TSI positive  →  true strength bullish."""
    return df["tsi"] > 0


# ── volatility signals ────────────────────────────────────────────────────────

def s25(df: pd.DataFrame) -> pd.Series:
    """Price near lower Bollinger Band (BB %B < 0.15)  →  potential mean-reversion long."""
    return df["bb_pct"] < 0.15


def s26(df: pd.DataFrame) -> pd.Series:
    """BB squeeze active  →  consolidation before breakout."""
    return df["squeeze"].astype(bool)


def s27(df: pd.DataFrame) -> pd.Series:
    """ATR-14 above its 20-bar SMA  →  expanding volatility."""
    return df["atr_14"] > df["atr_14"].rolling(20).mean()


def s28(df: pd.DataFrame) -> pd.Series:
    """Price above upper Keltner Channel  →  strong breakout."""
    return df["close"] > df["kc_upper"]


def s29(df: pd.DataFrame) -> pd.Series:
    """Williams Vix Fix spike (wvf_alert1)  →  fear spike / reversal setup."""
    return df["wvf_alert1"].astype(bool)


# ── volume / flow signals ─────────────────────────────────────────────────────

def s30(df: pd.DataFrame) -> pd.Series:
    """Volume above 1.5x its 20-bar average  →  volume confirmation."""
    return df["volume_ratio"] > 1.5


def s31(df: pd.DataFrame) -> pd.Series:
    """MFI below 25  →  money flow oversold."""
    return df["mfi"] < 25


def s32(df: pd.DataFrame) -> pd.Series:
    """CMF positive  →  Chaikin Money Flow bullish."""
    return df["cmf"] > 0


def s33(df: pd.DataFrame) -> pd.Series:
    """OBV rising (above its 10-bar SMA)  →  volume supporting price."""
    return df["obv"] > df["obv"].rolling(10).mean()


def s34(df: pd.DataFrame) -> pd.Series:
    """Delta normalized positive  →  buying pressure bar."""
    return df["delta_normalized"] > 0


# ── structure / price-action signals ─────────────────────────────────────────

def s35(df: pd.DataFrame) -> pd.Series:
    """Price above 20-bar Donchian mid  →  upper half of range."""
    return df["close"] > df["dc_middle"]


def s36(df: pd.DataFrame) -> pd.Series:
    """Price above Fibonacci 0.618 level (rolling 50-bar)  →  above retracement."""
    return df["close"] > df["fib_618"]


def s37(df: pd.DataFrame) -> pd.Series:
    """Inside bar detected  →  consolidation / coil."""
    return df["IB_IsIB"].astype(bool)


def s38(df: pd.DataFrame) -> pd.Series:
    """Inside bar Green Arrow breakout  →  bullish IB breakout."""
    return df["IB_GreenArrow"].astype(bool)


def s39(df: pd.DataFrame) -> pd.Series:
    """Range Filter buy signal."""
    return df["RF_BuySignal"].astype(bool)


def s40(df: pd.DataFrame) -> pd.Series:
    """UT Bot buy signal."""
    return df["ut_buy"].astype(bool)


def s41(df: pd.DataFrame) -> pd.Series:
    """Heikin Ashi bullish bar (ha_close > ha_open)."""
    return df["ha_close"] > df["ha_open"]


def s42(df: pd.DataFrame) -> pd.Series:
    """Two consecutive bullish HA bars  →  sustained HA trend."""
    bull = (df["ha_close"] > df["ha_open"]).astype(int)
    return (bull == 1) & (bull.shift(1) == 1)


def s43(df: pd.DataFrame) -> pd.Series:
    """Price above TEMA-21  →  dynamic trend."""
    return df["close"] > df["tema_21"]


def s44(df: pd.DataFrame) -> pd.Series:
    """VWMA-20 above SMA-20  →  volume-weighted bias bullish."""
    return df["vwma_20"] > df["sma_20"]

def validate_signals(df: pd.DataFrame):
    missing = {}
    for name, meta in SIGNALS.items():
        try:
            out = meta["fn"](df)
        except Exception as e:
            missing[name] = str(e)
    return missing


# ── registry ──────────────────────────────────────────────────────────────────
# Maps signal name → (function, short description, conflict_group)
# Signals in the same conflict_group cannot both appear in one strategy
# (they would be mutually exclusive or redundant).

SIGNAL_REGISTRY: dict[str, dict] = {
    "s1":  {"fn": s1,  "desc": "EMA9 > EMA21",              "group": "ema_short"},
    "s2":  {"fn": s2,  "desc": "EMA21 > EMA50",             "group": "ema_mid"},
    "s3":  {"fn": s3,  "desc": "EMA50 > EMA200",            "group": "ema_long"},
    "s4":  {"fn": s4,  "desc": "Price > VWAP",              "group": "price_level"},
    "s5":  {"fn": s5,  "desc": "MACD > Signal",             "group": "macd"},
    "s6":  {"fn": s6,  "desc": "MACD hist rising",          "group": "macd"},
    "s7":  {"fn": s7,  "desc": "Supertrend bullish",        "group": "supertrend"},
    "s8":  {"fn": s8,  "desc": "Price > HullMA20",          "group": "hull"},
    "s9":  {"fn": s9,  "desc": "WT wt1 > wt2",              "group": "wt_trend"},
    "s10": {"fn": s10, "desc": "UT bar_buy",                "group": "ut"},
    "s11": {"fn": s11, "desc": "RF trend bullish",          "group": "rf"},
    "s12": {"fn": s12, "desc": "ADX>20 & +DI>-DI",          "group": "adx"},
    "s13": {"fn": s13, "desc": "RSI14 < 35",                "group": "rsi"},
    "s14": {"fn": s14, "desc": "RSI14 > 50",                "group": "rsi"},
    "s15": {"fn": s15, "desc": "RSI 55-75",                 "group": "rsi"},
    "s16": {"fn": s16, "desc": "Stoch %K < 25",             "group": "stoch"},
    "s17": {"fn": s17, "desc": "Stoch cross up <30",        "group": "stoch"},
    "s18": {"fn": s18, "desc": "StochRSI %K < 20",          "group": "stochrsi"},
    "s19": {"fn": s19, "desc": "CCI < -100",                "group": "cci"},
    "s20": {"fn": s20, "desc": "Williams%R < -80",          "group": "williams"},
    "s21": {"fn": s21, "desc": "WT cross up",               "group": "wt_signal"},
    "s22": {"fn": s22, "desc": "WT oversold",               "group": "wt_signal"},
    "s23": {"fn": s23, "desc": "UltOsc < 30",               "group": "ultosc"},
    "s24": {"fn": s24, "desc": "TSI > 0",                   "group": "tsi"},
    "s25": {"fn": s25, "desc": "BB%B < 0.15",               "group": "bb"},
    "s26": {"fn": s26, "desc": "BB squeeze",                "group": "squeeze"},
    "s27": {"fn": s27, "desc": "ATR expanding",             "group": "atr"},
    "s28": {"fn": s28, "desc": "Price > KC upper",          "group": "kc"},
    "s29": {"fn": s29, "desc": "WVF spike",                 "group": "wvf"},
    "s30": {"fn": s30, "desc": "Volume > 1.5x avg",         "group": "volume"},
    "s31": {"fn": s31, "desc": "MFI < 25",                  "group": "mfi"},
    "s32": {"fn": s32, "desc": "CMF > 0",                   "group": "cmf"},
    "s33": {"fn": s33, "desc": "OBV rising",                "group": "obv"},
    "s34": {"fn": s34, "desc": "Delta norm > 0",            "group": "delta"},
    "s35": {"fn": s35, "desc": "Price > DC mid",            "group": "dc"},
    "s36": {"fn": s36, "desc": "Price > Fib0.618",          "group": "fib"},
    "s37": {"fn": s37, "desc": "Inside bar",                "group": "ib"},
    "s38": {"fn": s38, "desc": "IB green breakout",         "group": "ib"},
    "s39": {"fn": s39, "desc": "RF buy signal",             "group": "rf"},
    "s40": {"fn": s40, "desc": "UT buy signal",             "group": "ut"},
    "s41": {"fn": s41, "desc": "HA bullish bar",            "group": "ha"},
    "s42": {"fn": s42, "desc": "2 consec HA bulls",         "group": "ha"},
    "s43": {"fn": s43, "desc": "Price > TEMA21",            "group": "tema"},
    "s44": {"fn": s44, "desc": "VWMA20 > SMA20",            "group": "vwma"},
}

# ── derived signal groups ─────────────────────────────────────────────

SIGNALS = SIGNAL_REGISTRY

# You NEED to define direction — do it manually (important for quality)

BULL_SIGNALS = [
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12",
    "s14","s15","s21","s24","s30","s32","s33","s34","s35","s36",
    "s38","s39","s40","s41","s42","s43","s44"
]

BEAR_SIGNALS = [
    # inverse / mean reversion / overbought-type signals
    "s13","s16","s17","s18","s19","s20","s22","s23","s25","s28",
    "s29","s31","s37"
]

NEUTRAL_SIGNALS = [
    "s26","s27"  # squeeze, volatility expansion
]