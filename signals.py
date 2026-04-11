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


# ══════════════════════════════════════════════════════════════════════════════
#  RSI GAINZY SIGNALS  (s45 – s52)
#  Depend on columns added by RSIGainzy in feature_engineer():
#    gainzy_trend        : int  {-3, -1, 0, 1, 2, 3}
#    gainzy_strong_bull  : int  {0, 1}   trend == 3
#    gainzy_any_bull     : int  {0, 1}   trend > 0
#    gainzy_momentum     : int  {0, 1}   trend == 2  (blue zone)
#    gainzy_strong_bear  : int  {0, 1}   trend == -3
#    gainzy_any_bear     : int  {0, 1}   trend < 0
#    gainzy_bull_flip    : int  {0, 1}   0→positive transition
#    gainzy_bear_flip    : int  {0, 1}   0→negative transition
# ══════════════════════════════════════════════════════════════════════════════

def s45(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy strong bull (light_green, trend=+3).
    RSI broke above BOTH trend lines while they were rising — the highest-
    conviction bullish state in the Gainzy model.
    Direction: BULL
    """
    return df["gainzy_strong_bull"].astype(bool)


def s46(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy any-bull (green or light_green, trend>0).
    RSI is above at least one ascending trend line — a broad bullish filter
    that stays on for the duration of the bullish phase.
    Direction: BULL
    """
    return df["gainzy_any_bull"].astype(bool)


def s47(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy bull flip — fresh bullish turn (trend flipped from ≤0 to >0).
    Fires on the single bar where Gainzy switches from neutral/bearish into
    bullish territory → good entry trigger rather than a regime filter.
    Direction: BULL
    """
    return df["gainzy_bull_flip"].astype(bool)


def s48(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy momentum zone (blue, trend=+2).
    RSI sits between the two trend lines while the lower line is above the
    upper — an ambiguous but often transitional bullish setup used as a
    confirmation layer alongside other bull signals.
    Direction: BULL  (used as confirmation layer)
    """
    return df["gainzy_momentum"].astype(bool)


def s49(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy strong bear (pink, trend=-3).
    RSI broke below BOTH trend lines while they were falling — the highest-
    conviction bearish state in the Gainzy model.
    Direction: BEAR
    """
    return df["gainzy_strong_bear"].astype(bool)


def s50(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy any-bear (red or pink, trend<0).
    RSI is below at least one descending trend line — broad bearish regime
    filter that persists for the length of the bearish phase.
    Direction: BEAR
    """
    return df["gainzy_any_bear"].astype(bool)


def s51(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy bear flip — fresh bearish turn (trend flipped from ≥0 to <0).
    Fires on the single bar where Gainzy switches from neutral/bullish into
    bearish territory → entry trigger for short strategies.
    Direction: BEAR
    """
    return df["gainzy_bear_flip"].astype(bool)


def s52(df: pd.DataFrame) -> pd.Series:
    """
    Gainzy NOT bearish (trend >= 0).
    Used as a bearish-regime exclusion filter — avoids shorting into a
    neutral or bullish Gainzy state when combined with other bear signals.
    Direction: BEAR  (permissive filter / guard)
    """
    return df["gainzy_trend"] >= 0

# ══════════════════════════════════════════════════════════════════════════════
#  ALPHATREND SIGNALS  (s53 – s57)
#  Columns produced by alpha_trend() in feature_engineer():
#    at_line           – the AT value
#    at_trend_up       – AT > AT[2]  (bullish colour regime)
#    at_buy_signal     – crossover(AT, AT[2])   ← Pine BUY signal bar
#    at_sell_signal    – crossunder(AT, AT[2])  ← Pine SELL signal bar
#    at_confirmed_buy  – buy signal shifted 1 bar (confirmed after close)
#    at_confirmed_sell – sell signal shifted 1 bar
#    at_price_above    – close > AT
#    at_price_below    – close < AT
# ══════════════════════════════════════════════════════════════════════════════

def s53(df: pd.DataFrame) -> pd.Series:
    """AlphaTrend bullish regime — AT > AT[2] (green fill active).
    Broad trend filter; stays True for the whole upswing. Direction: BULL"""
    return df["at_trend_up"].astype(bool)

def s54(df: pd.DataFrame) -> pd.Series:
    """AlphaTrend buy signal — AT crosses above AT[2].
    Fires on the single crossover bar; use as entry trigger. Direction: BULL"""
    return df["at_buy_signal"].astype(bool)

def s55(df: pd.DataFrame) -> pd.Series:
    """AlphaTrend confirmed buy — buy signal confirmed after bar close.
    One bar later than s54; lower noise, slightly later entry. Direction: BULL"""
    return df["at_confirmed_buy"].astype(bool)

def s56(df: pd.DataFrame) -> pd.Series:
    """AlphaTrend bearish regime — AT < AT[2] (red fill active).
    Broad trend filter for shorts; stays True for whole downswing. Direction: BEAR"""
    return df["at_trend_down"].astype(bool)

def s57(df: pd.DataFrame) -> pd.Series:
    """AlphaTrend sell signal — AT crosses below AT[2].
    Fires on the single crossunder bar; use as short entry trigger. Direction: BEAR"""
    return df["at_sell_signal"].astype(bool)

# ══════════════════════════════════════════════════════════════════════════════
#  CHANDELIER EXIT SIGNALS  (s58 – s62)
#  Columns produced by chandelier_exit() in feature_engineer():
#    ce_direction       – 1 (bull) or -1 (bear)
#    ce_buy_signal      – dir flipped -1 → 1  (Pine Buy label bar)
#    ce_sell_signal     – dir flipped  1 → -1 (Pine Sell label bar)
#    ce_is_long         – entire bullish regime (dir == 1)
#    ce_is_short        – entire bearish regime (dir == -1)
#    ce_price_above_stop– close above long stop while in bull regime
#    ce_price_below_stop– close below short stop while in bear regime
# ══════════════════════════════════════════════════════════════════════════════

def s58(df: pd.DataFrame) -> pd.Series:
    """CE bullish regime — dir == 1 (long stop active, green fill).
    Broad trend filter; stays True for the whole upswing. Direction: BULL"""
    return df["ce_is_long"].astype(bool)

def s59(df: pd.DataFrame) -> pd.Series:
    """CE buy signal — direction flipped from -1 to 1.
    Fires on the single bar Pine plots the Buy label; sharp entry trigger. Direction: BULL"""
    return df["ce_buy_signal"].astype(bool)

def s60(df: pd.DataFrame) -> pd.Series:
    """CE price holding above long stop in bull regime.
    Confirms price has not pulled back through the trailing stop. Direction: BULL"""
    return df["ce_price_above_stop"].astype(bool)

def s61(df: pd.DataFrame) -> pd.Series:
    """CE bearish regime — dir == -1 (short stop active, red fill).
    Broad trend filter for shorts; stays True for whole downswing. Direction: BEAR"""
    return df["ce_is_short"].astype(bool)

def s62(df: pd.DataFrame) -> pd.Series:
    """CE sell signal — direction flipped from 1 to -1.
    Fires on the single bar Pine plots the Sell label; short entry trigger. Direction: BEAR"""
    return df["ce_sell_signal"].astype(bool)

# ══════════════════════════════════════════════════════════════════════════════
#  QQE SIGNALS  (s63 – s67)
#  Columns: qqe_long, qqe_short, qqe_is_bull, qqe_is_bear, qqe_rsi_above50
# ══════════════════════════════════════════════════════════════════════════════

def s63(df: pd.DataFrame) -> pd.Series:
    """QQE long signal — TL crosses below RsiMa for the first bar.
    Equivalent to Pine qqeLong plotshape. Sharp entry trigger. Direction: BULL"""
    return df["qqe_long"].astype(bool)

def s64(df: pd.DataFrame) -> pd.Series:
    """QQE bull regime — FastAtrRsiTL trend == 1 (full upswing).
    Broad regime filter; stays True for whole bull phase. Direction: BULL"""
    return df["qqe_is_bull"].astype(bool)

def s65(df: pd.DataFrame) -> pd.Series:
    """QQE RSI-MA above 50 — smoothed RSI in bullish momentum zone.
    Use as confirmation alongside trend signals. Direction: BULL"""
    return df["qqe_rsi_above50"].astype(bool)

def s66(df: pd.DataFrame) -> pd.Series:
    """QQE short signal — TL crosses above RsiMa for the first bar.
    Equivalent to Pine qqeShort plotshape. Short entry trigger. Direction: BEAR"""
    return df["qqe_short"].astype(bool)

def s67(df: pd.DataFrame) -> pd.Series:
    """QQE bear regime — FastAtrRsiTL trend == -1 (full downswing).
    Broad regime filter for shorts. Direction: BEAR"""
    return df["qqe_is_bear"].astype(bool)

# ══════════════════════════════════════════════════════════════════════════════
#  HALFTREND SIGNALS  (s68 – s72)
#  Columns: ht_buy_signal, ht_sell_signal, ht_is_bull, ht_is_bear, ht_line
# ══════════════════════════════════════════════════════════════════════════════

def s68(df: pd.DataFrame) -> pd.Series:
    """HalfTrend buy signal — trend flipped from bear to bull.
    Fires on the single transition bar Pine plots the up arrow. Direction: BULL"""
    return df["ht_buy_signal"].astype(bool)

def s69(df: pd.DataFrame) -> pd.Series:
    """HalfTrend bull regime — ht_trend == 0 (full bull phase).
    Broad regime filter; stays True for the whole upswing. Direction: BULL"""
    return df["ht_is_bull"].astype(bool)

def s70(df: pd.DataFrame) -> pd.Series:
    """Price above HalfTrend line — close on bullish side of HT.
    Confirms price hasn't broken back below the dynamic support. Direction: BULL"""
    return (df["close"] > df["ht_line"]).astype(bool)

def s71(df: pd.DataFrame) -> pd.Series:
    """HalfTrend sell signal — trend flipped from bull to bear.
    Fires on the single transition bar Pine plots the down arrow. Direction: BEAR"""
    return df["ht_sell_signal"].astype(bool)

def s72(df: pd.DataFrame) -> pd.Series:
    """HalfTrend bear regime — ht_trend == 1 (full bear phase).
    Broad regime filter for shorts. Direction: BEAR"""
    return df["ht_is_bear"].astype(bool)

# ══════════════════════════════════════════════════════════════════════════════
#  RMI TREND SNIPER SIGNALS  (s73 – s77)
#  Columns: rmi_buy, rmi_sell, rmi_positive, rmi_negative, rmi_above_pmom
# ══════════════════════════════════════════════════════════════════════════════

def s73(df: pd.DataFrame) -> pd.Series:
    """RMI buy signal — positive state latched on for the first bar.
    Equivalent to Pine alertcondition BUY. Sharp entry trigger. Direction: BULL"""
    return df["rmi_buy"].astype(bool)

def s74(df: pd.DataFrame) -> pd.Series:
    """RMI positive regime — RSI_MFI latched above pmom threshold.
    Broad bull regime filter; stays True for whole positive phase. Direction: BULL"""
    return df["rmi_positive"].astype(bool)

def s75(df: pd.DataFrame) -> pd.Series:
    """RMI RSI_MFI above pmom (66) — momentum in strong bull zone.
    Use as confirmation that momentum has not faded. Direction: BULL"""
    return df["rmi_above_pmom"].astype(bool)

def s76(df: pd.DataFrame) -> pd.Series:
    """RMI sell signal — negative state latched on for the first bar.
    Equivalent to Pine alertcondition SELL. Short entry trigger. Direction: BEAR"""
    return df["rmi_sell"].astype(bool)

def s77(df: pd.DataFrame) -> pd.Series:
    """RMI negative regime — RSI_MFI latched below nmom threshold.
    Broad bear regime filter; stays True for whole negative phase. Direction: BEAR"""
    return df["rmi_negative"].astype(bool)


# ── validation helper ──────────────────────────────────────────────────────────

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
    # ── RSI Gainzy signals ────────────────────────────────────────────────────
    # group "gainzy" ensures only ONE Gainzy signal per strategy (no redundancy)
    "s45": {"fn": s45, "desc": "Gainzy strong bull (+3)",   "group": "gainzy"},
    "s46": {"fn": s46, "desc": "Gainzy any bull (>0)",      "group": "gainzy"},
    "s47": {"fn": s47, "desc": "Gainzy bull flip",          "group": "gainzy"},
    "s48": {"fn": s48, "desc": "Gainzy momentum (blue)",    "group": "gainzy"},
    "s49": {"fn": s49, "desc": "Gainzy strong bear (-3)",   "group": "gainzy"},
    "s50": {"fn": s50, "desc": "Gainzy any bear (<0)",      "group": "gainzy"},
    "s51": {"fn": s51, "desc": "Gainzy bear flip",          "group": "gainzy"},
    "s52": {"fn": s52, "desc": "Gainzy not bearish (≥0)",   "group": "gainzy"},
    "s53": {"fn": s53, "desc": "AT trend up (regime)",     "group": "alphatrend"},
    "s54": {"fn": s54, "desc": "AT buy signal (cross)",    "group": "alphatrend"},
    "s55": {"fn": s55, "desc": "AT confirmed buy",         "group": "alphatrend"},
    "s56": {"fn": s56, "desc": "AT trend down (regime)",   "group": "alphatrend"},
    "s57": {"fn": s57, "desc": "AT sell signal (cross)",   "group": "alphatrend"},
    "s58": {"fn": s58, "desc": "CE bull regime (dir=1)",      "group": "chandelier"},
    "s59": {"fn": s59, "desc": "CE buy signal (flip)",        "group": "chandelier"},
    "s60": {"fn": s60, "desc": "CE price above long stop",    "group": "chandelier"},
    "s61": {"fn": s61, "desc": "CE bear regime (dir=-1)",     "group": "chandelier"},
    "s62": {"fn": s62, "desc": "CE sell signal (flip)",       "group": "chandelier"},
    "s63": {"fn": s63, "desc": "QQE long signal",        "group": "qqe"},
    "s64": {"fn": s64, "desc": "QQE bull regime",         "group": "qqe"},
    "s65": {"fn": s65, "desc": "QQE RSI-MA > 50",         "group": "qqe"},
    "s66": {"fn": s66, "desc": "QQE short signal",        "group": "qqe"},
    "s67": {"fn": s67, "desc": "QQE bear regime",         "group": "qqe"},
    "s68": {"fn": s68, "desc": "HT buy signal",           "group": "halftrend"},
    "s69": {"fn": s69, "desc": "HT bull regime",           "group": "halftrend"},
    "s70": {"fn": s70, "desc": "Price > HT line",          "group": "halftrend"},
    "s71": {"fn": s71, "desc": "HT sell signal",           "group": "halftrend"},
    "s72": {"fn": s72, "desc": "HT bear regime",           "group": "halftrend"},
    "s73": {"fn": s73, "desc": "RMI buy signal",          "group": "rmi"},
    "s74": {"fn": s74, "desc": "RMI positive regime",     "group": "rmi"},
    "s75": {"fn": s75, "desc": "RMI RSI_MFI > pmom",      "group": "rmi"},
    "s76": {"fn": s76, "desc": "RMI sell signal",         "group": "rmi"},
    "s77": {"fn": s77, "desc": "RMI negative regime",     "group": "rmi"},
}

# ── derived signal groups ─────────────────────────────────────────────

SIGNALS = SIGNAL_REGISTRY

# You NEED to define direction — do it manually (important for quality)

BULL_SIGNALS = [
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12",
    "s14","s15","s21","s24","s30","s32","s33","s34","s35","s36",
    "s38","s39","s40","s41","s42","s43","s44",
    # RSI Gainzy — bullish
    "s45",  # strong bull (light_green)
    "s46",  # any bull regime
    "s47",  # bull flip (entry trigger)
    "s48",  # momentum / blue zone (confirmation)
    "s53","s54","s55","s58","s59","s60","s63","s64","s65","s68","s69","s70","s73","s74","s75"
]

BEAR_SIGNALS = [
    # inverse / mean reversion / overbought-type signals
    "s13","s16","s17","s18","s19","s20","s22","s23","s25","s28",
    "s29","s31","s37",
    # RSI Gainzy — bearish
    "s49",  # strong bear (pink)
    "s50",  # any bear regime
    "s51",  # bear flip (entry trigger)
    "s52",  # not-bearish guard (permissive bear filter)
    "s56","s57","s61","s62","s66","s67","s71","s72","s76","s77"
]

NEUTRAL_SIGNALS = [
    "s26","s27"  # squeeze, volatility expansion
]