"""
main.py
───────
End-to-end Strategy Factory orchestrator.

Flow
  1. Load & feature-engineer data
  2. Split train / test
  3. Generate N strategies
  4. Evaluate on TRAIN in parallel (multiprocessing)
  5. Filter → Score → Rank top strategies
  6. Validate top strategies on TEST
  7. Save results to CSV + print summary

Usage
  python main.py --csv path/to/ethusd_15m.csv --n 750 --top 30 --workers 4
"""

import argparse
import sys
import time
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange

# ── local imports ─────────────────────────────────────────────────────────────
from generator import generate_strategies
from backtester import run_backtest
from evaluator  import (
    filter_metrics, score_strategy, rank_strategies,
    train_test_split, validate_on_test, save_results, print_summary
)

def _evaluate_one(args):
    strat, df = args
    metrics = run_backtest(strat, df)

    if metrics is None:
        return None
    
    print("\nDEBUG METRICS CHECK")
    print(metrics)

    if metrics is not None:
        print("Trades:", metrics["n_trades"], ">= 10 ?", metrics["n_trades"] >= 10)
        print("DD:", metrics["max_drawdown"], ">= -25 ?", metrics["max_drawdown"] >= -25)

    if not filter_metrics(metrics):
        return None

    return {
        "strategy": strat,
        "metrics": metrics,
        "score": score_strategy(metrics),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def alpha_trend(df: pd.DataFrame, coeff: float = 1.0, ap: int = 14, no_volume: bool = False) -> pd.DataFrame:
    """
    AlphaTrend — faithful Python port of KivancOzbilgic's Pine Script.

    Pine logic:
      ATR   = sma(tr, AP)                        ← simple MA, not RMA
      upT   = low  - ATR * coeff                 ← bull-mode floor
      downT = high + ATR * coeff                 ← bear-mode ceiling
      cond  = mfi(hlc3, AP) >= 50                ← or rsi>=50 if no volume
      AT[i] = cond ? max(upT[i], AT[i-1])        ← ratchet up
                   : min(downT[i], AT[i-1])       ← ratchet down
    """
    high  = df["high"];  low = df["low"];  close = df["close"]
    hlc3  = (high + low + close) / 3

    # True Range → simple ATR (Pine uses ta.sma not ta.rma here)
    prev_close = close.shift(1)
    tr  = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ap).mean()

    upT   = low  - atr * coeff
    downT = high + atr * coeff

    # Oscillator condition
    if no_volume:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=ap - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=ap - 1, adjust=False).mean()
        cond  = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) >= 50
    else:
        raw_mf  = hlc3 * df["volume"]
        pos_mf  = raw_mf.where(hlc3 > hlc3.shift(1), 0.0)
        neg_mf  = raw_mf.where(hlc3 < hlc3.shift(1), 0.0)
        mfi     = 100 - 100 / (1 + pos_mf.rolling(ap).sum() / neg_mf.rolling(ap).sum().replace(0, np.nan))
        cond    = mfi >= 50

    # Stateful ratchet loop — cannot be vectorised
    upT_v = upT.values;  downT_v = downT.values;  cond_v = cond.values
    at    = np.full(len(close), np.nan)
    start = 0
    for i in range(len(close)):
        if not (np.isnan(upT_v[i]) or np.isnan(downT_v[i])):
            at[i] = upT_v[i] if cond_v[i] else downT_v[i];  start = i + 1;  break

    for i in range(start, len(close)):
        if np.isnan(upT_v[i]) or np.isnan(downT_v[i]):
            at[i] = at[i - 1];  continue
        at[i] = max(upT_v[i], at[i-1]) if cond_v[i] else min(downT_v[i], at[i-1])

    at_s = pd.Series(at, index=df.index)
    at_2 = at_s.shift(2)          # AT[2] in Pine
    at_prev   = at_s.shift(1)
    at_2prev  = at_2.shift(1)

    df["at_line"]          = at_s
    df["at_trend_up"]      = (at_s > at_2).astype(int)               # green fill regime
    df["at_trend_down"]    = (at_s < at_2).astype(int)               # red fill regime
    df["at_buy_signal"]    = ((at_s > at_2) & (at_prev <= at_2prev)).astype(int)   # crossover
    df["at_sell_signal"]   = ((at_s < at_2) & (at_prev >= at_2prev)).astype(int)  # crossunder
    df["at_confirmed_buy"] = df["at_buy_signal"].shift(1).fillna(0).astype(int)   # 1-bar confirmed
    df["at_confirmed_sell"]= df["at_sell_signal"].shift(1).fillna(0).astype(int)
    df["at_price_above"]   = (close > at_s).astype(int)              # price on bull side
    df["at_price_below"]   = (close < at_s).astype(int)

    return df

def chandelier_exit(df: pd.DataFrame, length: int = 22, mult: float = 3.0, use_close: bool = True) -> pd.DataFrame:
    """
    Chandelier Exit — faithful Python port of Alex Orekhov's Pine Script.

    Pine logic:
      atr       = mult * ta.atr(length)

      longStop  = highest(close, length) - atr          # if useClose
                  ratchets UP:  close[1] > longStopPrev → max(longStop, longStopPrev)

      shortStop = lowest(close, length) + atr           # if useClose
                  ratchets DOWN: close[1] < shortStopPrev → min(shortStop, shortStopPrev)

      dir = 1  if close > shortStopPrev
           -1  if close < longStopPrev
            dir (unchanged) otherwise

      buySignal  = dir == 1 and dir[1] == -1
      sellSignal = dir == -1 and dir[1] == 1
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # ATR using Wilder's RMA (ta.atr in Pine uses RMA, not SMA)
    prev_close = close.shift(1)
    tr  = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean() * mult

    # Raw stops before ratcheting
    if use_close:
        long_stop_raw  = close.rolling(length).max() - atr
        short_stop_raw = close.rolling(length).min() + atr
    else:
        long_stop_raw  = high.rolling(length).max() - atr
        short_stop_raw = low.rolling(length).min()  + atr

    # Stateful ratchet — must loop bar by bar
    n              = len(close)
    long_stop_v    = long_stop_raw.values
    short_stop_v   = short_stop_raw.values
    close_v        = close.values

    long_stop_out  = np.full(n, np.nan)
    short_stop_out = np.full(n, np.nan)
    direction      = np.ones(n, dtype=int)   # default dir = 1

    for i in range(n):
        if np.isnan(long_stop_v[i]) or np.isnan(short_stop_v[i]):
            continue

        # ── Long stop ratchet ─────────────────────────────────────────────
        if i == 0:
            long_stop_out[i]  = long_stop_v[i]
            short_stop_out[i] = short_stop_v[i]
        else:
            prev_ls = long_stop_out[i - 1] if not np.isnan(long_stop_out[i - 1]) else long_stop_v[i]
            prev_ss = short_stop_out[i - 1] if not np.isnan(short_stop_out[i - 1]) else short_stop_v[i]

            # Pine: close[1] > longStopPrev → max(longStop, longStopPrev)
            if close_v[i - 1] > prev_ls:
                long_stop_out[i] = max(long_stop_v[i], prev_ls)
            else:
                long_stop_out[i] = long_stop_v[i]

            # Pine: close[1] < shortStopPrev → min(shortStop, shortStopPrev)
            if close_v[i - 1] < prev_ss:
                short_stop_out[i] = min(short_stop_v[i], prev_ss)
            else:
                short_stop_out[i] = short_stop_v[i]

        # ── Direction ─────────────────────────────────────────────────────
        prev_dir  = direction[i - 1] if i > 0 else 1
        prev_ss_v = short_stop_out[i - 1] if i > 0 and not np.isnan(short_stop_out[i - 1]) else short_stop_out[i]
        prev_ls_v = long_stop_out[i - 1]  if i > 0 and not np.isnan(long_stop_out[i - 1])  else long_stop_out[i]

        if close_v[i] > prev_ss_v:
            direction[i] = 1
        elif close_v[i] < prev_ls_v:
            direction[i] = -1
        else:
            direction[i] = prev_dir

    ls_s  = pd.Series(long_stop_out,  index=df.index)
    ss_s  = pd.Series(short_stop_out, index=df.index)
    dir_s = pd.Series(direction,       index=df.index)

    df["ce_long_stop"]    = ls_s
    df["ce_short_stop"]   = ss_s
    df["ce_direction"]    = dir_s                                        # 1 = bull, -1 = bear

    df["ce_buy_signal"]   = ((dir_s == 1)  & (dir_s.shift(1) == -1)).astype(int)   # dir flipped to bull
    df["ce_sell_signal"]  = ((dir_s == -1) & (dir_s.shift(1) == 1)).astype(int)    # dir flipped to bear

    df["ce_is_long"]      = (dir_s == 1).astype(int)    # entire bull regime
    df["ce_is_short"]     = (dir_s == -1).astype(int)   # entire bear regime

    # Price relative to active stop
    df["ce_price_above_stop"] = (
        (dir_s == 1)  & (close > ls_s)
    ).astype(int)
    df["ce_price_below_stop"] = (
        (dir_s == -1) & (close < ss_s)
    ).astype(int)

    return df

def qqe_signals(df: pd.DataFrame, rsi_period: int = 14, sf: int = 5, qqe_factor: float = 4.238, threshold: int = 10) -> pd.DataFrame:
    """
    QQE (Quantitative Qualitative Estimation) — port of colinmck's Pine v4 script.

    Pine logic:
      Wilders_Period = RSI_Period * 2 - 1
      RsiMa          = ema(rsi(close, RSI_Period), SF)
      AtrRsi         = abs(RsiMa[1] - RsiMa)
      MaAtrRsi       = ema(AtrRsi, Wilders_Period)
      dar            = ema(MaAtrRsi, Wilders_Period) * QQE_Factor

      newlongband  = RsiMa - dar
      newshortband = RsiMa + dar

      longband  ratchets UP   when RsiMa stays above it
      shortband ratchets DOWN when RsiMa stays below it

      trend = 1  if RsiMa crosses above shortband[1]
             -1  if longband[1] crosses above RsiMa
              (unchanged otherwise)

      FastAtrRsiTL = trend==1 ? longband : shortband

      qqe_long  fires when FastAtrRsiTL < RsiMa for exactly 1 bar (QQExlong==1)
      qqe_short fires when FastAtrRsiTL > RsiMa for exactly 1 bar (QQExshort==1)
    """
    close = df["close"]
    wp    = rsi_period * 2 - 1

    # RSI → smoothed with EMA(SF)
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi_ma = rsi.ewm(span=sf, adjust=False).mean()

    # ATR of RSI
    atr_rsi    = rsi_ma.diff().abs()
    ma_atr_rsi = atr_rsi.ewm(span=wp, adjust=False).mean()
    dar        = ma_atr_rsi.ewm(span=wp, adjust=False).mean() * qqe_factor

    rsi_ma_v  = rsi_ma.values
    dar_v     = dar.values
    n         = len(close)

    long_band  = np.full(n, np.nan)
    short_band = np.full(n, np.nan)
    trend      = np.ones(n, dtype=int)

    for i in range(1, n):
        if np.isnan(dar_v[i]) or np.isnan(rsi_ma_v[i]):
            continue

        new_lb = rsi_ma_v[i] - dar_v[i]
        new_sb = rsi_ma_v[i] + dar_v[i]

        prev_lb = long_band[i-1]  if not np.isnan(long_band[i-1])  else new_lb
        prev_sb = short_band[i-1] if not np.isnan(short_band[i-1]) else new_sb

        # longband ratchets UP when RSI stays above it
        if rsi_ma_v[i-1] > prev_lb and rsi_ma_v[i] > prev_lb:
            long_band[i] = max(prev_lb, new_lb)
        else:
            long_band[i] = new_lb

        # shortband ratchets DOWN when RSI stays below it
        if rsi_ma_v[i-1] < prev_sb and rsi_ma_v[i] < prev_sb:
            short_band[i] = min(prev_sb, new_sb)
        else:
            short_band[i] = new_sb

        # trend: cross(RSIndex, shortband[1]) → 1, cross(longband[1], RSIndex) → -1
        prev_trend = trend[i-1]
        cross_up   = (rsi_ma_v[i-1] <= prev_sb) and (rsi_ma_v[i] > prev_sb)   # RSI crosses above shortband
        cross_dn   = (prev_lb >= rsi_ma_v[i-1]) and (prev_lb < rsi_ma_v[i])    # longband[1] crosses above RSI — NOTE: Pine cross(longband[1], RSIndex) means longband was below and is now above
        # Re-check Pine: cross(longband[1], RSIndex) = longband[1] crosses RSIndex → longband goes above RSI
        cross_dn   = (long_band[i-1] <= rsi_ma_v[i-1]) and (long_band[i] > rsi_ma_v[i]) if not np.isnan(long_band[i]) else False

        if cross_up:
            trend[i] = 1
        elif cross_dn:
            trend[i] = -1
        else:
            trend[i] = prev_trend

    lb_s  = pd.Series(long_band,  index=df.index)
    sb_s  = pd.Series(short_band, index=df.index)
    dir_s = pd.Series(trend,       index=df.index)

    fast_tl = lb_s.where(dir_s == 1, sb_s)   # FastAtrRsiTL

    # QQExlong / QQExshort — consecutive bar counts
    above = (fast_tl < rsi_ma).astype(int)    # TL below RSI → bull side
    below = (fast_tl > rsi_ma).astype(int)    # TL above RSI → bear side

    # Count resets to 0 when condition breaks, else increments
    qqe_long_count  = above * (above.groupby((above != above.shift()).cumsum()).cumcount() + 1)
    qqe_short_count = below * (below.groupby((below != below.shift()).cumsum()).cumcount() + 1)

    df["qqe_rsi_ma"]      = rsi_ma
    df["qqe_fast_tl"]     = fast_tl
    df["qqe_trend"]       = dir_s                                          # 1 bull / -1 bear
    df["qqe_long"]        = (qqe_long_count  == 1).astype(int)            # exact crossover bar
    df["qqe_short"]       = (qqe_short_count == 1).astype(int)            # exact crossunder bar
    df["qqe_is_bull"]     = (dir_s == 1).astype(int)                      # full bull regime
    df["qqe_is_bear"]     = (dir_s == -1).astype(int)                     # full bear regime
    df["qqe_rsi_above50"] = (rsi_ma > 50).astype(int)                     # RSI-MA momentum

    return df

def half_trend(df: pd.DataFrame, amplitude: int = 2, channel_deviation: int = 2) -> pd.DataFrame:
    """
    HalfTrend — port of Alex Orekhov's Pine v6 script.

    Pine logic:
      atr2 = ta.atr(100) / 2
      dev  = channelDeviation * atr2

      nextTrend toggles the "candidate" direction.
      trend     is the confirmed direction (0=bull, 1=bear in Pine).

      In bull mode (trend==0):
        up = max(maxLowPrice, up[1])      — ratchets up with lowest low of amplitude bars
        arrowUp fires on transition from trend==1 → trend==0

      In bear mode (trend==1):
        down = min(minHighPrice, down[1]) — ratchets down with highest high of amplitude bars
        arrowDown fires on transition from trend==0 → trend==1

      buySignal  = arrowUp fired   (trend flipped to 0)
      sellSignal = arrowDown fired (trend flipped to 1)
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(close)

    # ATR(100) using Wilder RMA
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr     = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    atr100 = pd.Series(tr).ewm(alpha=1/100, adjust=False).mean().values
    atr2   = atr100 / 2
    dev    = channel_deviation * atr2

    # Output arrays
    trend_out     = np.zeros(n, dtype=int)
    up_out        = np.full(n, np.nan)
    down_out      = np.full(n, np.nan)
    atr_high_out  = np.full(n, np.nan)
    atr_low_out   = np.full(n, np.nan)
    buy_signal    = np.zeros(n, dtype=int)
    sell_signal   = np.zeros(n, dtype=int)

    # State variables
    trend_     = 0
    next_trend = 0
    max_low    = low[0]   if len(low) > 0 else 0.0
    min_high   = high[0]  if len(high) > 0 else 0.0
    up_        = 0.0
    down_      = 0.0

    for i in range(1, n):
        # highest/lowest over amplitude bars
        lo_slice = low[max(0, i - amplitude + 1): i + 1]
        hi_slice = high[max(0, i - amplitude + 1): i + 1]
        low_price  = lo_slice.min()
        high_price = hi_slice.max()

        high_ma = np.mean(hi_slice)
        low_ma  = np.mean(lo_slice)

        prev_low  = low[i-1]
        prev_high = high[i-1]

        if next_trend == 1:
            max_low = max(low_price, max_low)
            if high_ma < max_low and close[i] < prev_low:
                trend_     = 1
                next_trend = 0
                min_high   = high_price
        else:
            min_high = min(high_price, min_high)
            if low_ma > min_high and close[i] > prev_high:
                trend_     = 0
                next_trend = 1
                max_low    = low_price

        prev_trend = trend_out[i-1]

        if trend_ == 0:
            if prev_trend != 0:
                # Transition: bear → bull
                up_         = down_out[i-1] if not np.isnan(down_out[i-1]) else down_
                buy_signal[i] = 1
            else:
                up_ = max(max_low, up_out[i-1]) if not np.isnan(up_out[i-1]) else max_low
            up_out[i]       = up_
            atr_high_out[i] = up_ + dev[i]
            atr_low_out[i]  = up_ - dev[i]
        else:
            if prev_trend != 1:
                # Transition: bull → bear
                down_          = up_out[i-1] if not np.isnan(up_out[i-1]) else up_
                sell_signal[i] = 1
            else:
                down_ = min(min_high, down_out[i-1]) if not np.isnan(down_out[i-1]) else min_high
            down_out[i]     = down_
            atr_high_out[i] = down_ + dev[i]
            atr_low_out[i]  = down_ - dev[i]

        trend_out[i] = trend_

    idx = df.index
    ht  = np.where(trend_out == 0, up_out, down_out)   # the HalfTrend line value

    df["ht_line"]       = pd.Series(ht,           index=idx)
    df["ht_trend"]      = pd.Series(trend_out,    index=idx)   # 0=bull, 1=bear (Pine convention)
    df["ht_atr_high"]   = pd.Series(atr_high_out, index=idx)
    df["ht_atr_low"]    = pd.Series(atr_low_out,  index=idx)
    df["ht_buy_signal"] = pd.Series(buy_signal,   index=idx)   # trend flipped to bull
    df["ht_sell_signal"]= pd.Series(sell_signal,  index=idx)   # trend flipped to bear
    df["ht_is_bull"]    = (pd.Series(trend_out, index=idx) == 0).astype(int)
    df["ht_is_bear"]    = (pd.Series(trend_out, index=idx) == 1).astype(int)

    return df

def rmi_trend_sniper(df: pd.DataFrame, length: int = 14, pmom: int = 66, nmom: int = 30, rma_period: int = 20) -> pd.DataFrame:
    """
    RMI Trend Sniper — port of TZack88's Pine v5 script.

    Pine logic:
      up      = rma(max(change(close), 0), length)
      down    = rma(-min(change(close), 0), length)
      rsi     = 100 - 100/(1 + up/down)           ← RMI-style RSI using RMA
      mf      = mfi(hlc3, length)
      rsi_mfi = avg(rsi, mf)                       ← blend of RSI and MFI

      p_mom = rsi_mfi[1] < pmom and rsi_mfi > pmom   ← crossed above pmom
              and rsi_mfi > nmom
              and change(ema(close,5)) > 0            ← EMA-5 rising

      n_mom = rsi_mfi < nmom                          ← dropped below nmom
              and change(ema(close,5)) < 0

      positive: latches True on p_mom, resets on n_mom
      negative: latches True on n_mom, resets on p_mom

      RWMA = range-weighted MA of close over rma_period
      Band = clamp(atr(30)*0.3, close*0.003) [20-bar lag] / 2 * 8
      RWMA plotted as rwma-Band (bull) or rwma+Band (bear)

      buy  fires on positive and not positive[1]   ← first bar of bull latch
      sell fires on negative and not negative[1]   ← first bar of bear latch
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    hlc3  = (high + low + close) / 3

    # RMA = ewm with alpha=1/length (Wilder smoothing)
    chg   = close.diff()
    up    = chg.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
    down  = (-chg.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rsi   = 100 - 100 / (1 + up / down.replace(0, np.nan))

    # MFI
    raw_mf  = hlc3 * df["volume"]
    pos_mf  = raw_mf.where(hlc3 > hlc3.shift(1), 0.0)
    neg_mf  = raw_mf.where(hlc3 < hlc3.shift(1), 0.0)
    mfi     = 100 - 100 / (1 + pos_mf.rolling(length).sum() / neg_mf.rolling(length).sum().replace(0, np.nan))

    rsi_mfi = (rsi + mfi) / 2
    ema5    = close.ewm(span=5, adjust=False).mean()

    # p_mom / n_mom conditions
    p_mom = (
        (rsi_mfi.shift(1) < pmom) &
        (rsi_mfi > pmom) &
        (rsi_mfi > nmom) &
        (ema5.diff() > 0)
    )
    n_mom = (
        (rsi_mfi < nmom) &
        (ema5.diff() < 0)
    )

    # Stateful latch for positive / negative
    n          = len(close)
    positive   = np.zeros(n, dtype=int)
    negative   = np.zeros(n, dtype=int)
    pos_state  = False
    neg_state  = False

    for i in range(n):
        if p_mom.iloc[i]:
            pos_state = True
            neg_state = False
        if n_mom.iloc[i]:
            pos_state = False
            neg_state = True
        positive[i] = int(pos_state)
        negative[i] = int(neg_state)

    pos_s = pd.Series(positive, index=df.index)
    neg_s = pd.Series(negative, index=df.index)

    # ATR Band (Pine: _Band(30) = min(atr(30)*0.3, close*0.003)[20] / 2 * 8)
    prev_c  = close.shift(1)
    tr      = pd.concat([(high-low), (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    atr30   = tr.ewm(alpha=1/30, adjust=False).mean()
    raw_band= pd.concat([atr30 * 0.3, close * 0.003], axis=1).min(axis=1)
    band    = raw_band.shift(20) / 2 * 8   # [20] lag then scale

    # Range-Weighted MA (RWMA)
    bar_range = (high - low).replace(0, np.nan)
    weight    = bar_range / bar_range.rolling(rma_period).sum()
    rwma      = (close * weight).rolling(rma_period).sum() / weight.rolling(rma_period).sum()

    df["rmi_rsi_mfi"]    = rsi_mfi
    df["rmi_positive"]   = pos_s                                              # latched bull state
    df["rmi_negative"]   = neg_s                                              # latched bear state
    df["rmi_buy"]        = ((pos_s == 1) & (pos_s.shift(1) == 0)).astype(int)   # first bull bar
    df["rmi_sell"]       = ((neg_s == 1) & (neg_s.shift(1) == 0)).astype(int)   # first bear bar
    df["rmi_rwma"]       = rwma
    df["rmi_band"]       = band
    df["rmi_above_pmom"] = (rsi_mfi > pmom).astype(int)                      # RSI_MFI in strong zone
    df["rmi_below_nmom"] = (rsi_mfi < nmom).astype(int)                      # RSI_MFI in weak zone

    return df

def _sma(src, n): return SMAIndicator(close=src, window=n).sma_indicator()
def _ema(src, n): return EMAIndicator(close=src, window=n).ema_indicator()
def _wma(src, n):
    return src.rolling(n).apply(
        lambda x: np.dot(x, np.arange(1,n+1)) / np.arange(1,n+1).sum(), raw=True)
def _rma(src, n): return src.ewm(alpha=1/n, adjust=False).mean()
def _vwma(src, vol, n):
    return (src*vol).rolling(n).sum() / vol.rolling(n).sum()
def _hull_ma(src, n):
    h = max(1, int(n/2)); sq = max(1, round(np.sqrt(n)))
    return _wma(2*_wma(src,h) - _wma(src,n), sq)
def _tema(src, n):
    e1=_ema(src,n); e2=_ema(e1,n); e3=_ema(e2,n)
    return 3*e1 - 3*e2 + e3
def _gd(src, n, f):
    e1=_ema(src,n); e2=_ema(e1,n)
    return e1*(1+f) - e2*f
def _t3(src, n, f): return _gd(_gd(_gd(src,n,f),n,f),n,f)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def tema(series: pd.Series, window: int) -> pd.Series:
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return 3 * ema1 - 3 * ema2 + ema3


def heikin_ashi(df):
    ha_close = (df['open']+df['high']+df['low']+df['close'])/4
    ha_open  = ha_close.copy()
    for i in range(1,len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1]+ha_close.iloc[i-1])/2
    ha_high = pd.concat([ha_open, ha_close, df['high']], axis=1).max(axis=1)
    ha_low  = pd.concat([ha_open, ha_close, df['low']], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close


def supertrend(high, low, close, period=10, mult=3.0):
    atr   = AverageTrueRange(high,low,close,window=period).average_true_range()
    hl2   = (high+low)/2
    upper = (hl2+mult*atr).values; lower = (hl2-mult*atr).values
    close_v = close.values
    st = np.zeros(len(close)); trend = np.zeros(len(close), dtype=int)
    for i in range(1,len(close)):
        pu = upper[i-1] if not np.isnan(upper[i-1]) else upper[i]
        pl = lower[i-1] if not np.isnan(lower[i-1]) else lower[i]
        fu = upper[i] if upper[i]<pu or close_v[i-1]>pu else pu
        fl = lower[i] if lower[i]>pl or close_v[i-1]<pl else pl
        if st[i-1]==pu:
            st[i]=fl if close_v[i]>fu else fu; trend[i]=1 if close_v[i]>fu else -1
        else:
            st[i]=fu if close_v[i]<fl else fl; trend[i]=-1 if close_v[i]<fl else 1
    return pd.Series(st,index=close.index), pd.Series(trend,index=close.index)


def williams_vix_fix(close, low, pd_len=22, bbl=20, mult=2.0, lb=50, ph=0.85, pl=1.01):
    hc  = close.rolling(pd_len).max()
    wvf = (hc-low)/hc*100
    mid = wvf.rolling(bbl).mean(); std = wvf.rolling(bbl).std()
    upper = mid+mult*std; lower = mid-mult*std
    rh = wvf.rolling(lb).max()*ph; rl = wvf.rolling(lb).min()*pl
    return wvf, upper, lower, rh, rl, wvf>=upper, wvf>=rh


def wavetrend(high, low, close, n1=10, n2=21):
    hlc3 = (high+low+close)/3
    esa  = _ema(hlc3,n1); d = _ema((hlc3-esa).abs(),n1)
    ci   = (hlc3-esa)/(0.015*d)
    wt1  = _ema(ci,n2); wt2 = _sma(wt1,4)
    wt_diff = wt1-wt2
    wt1_min = wt1.rolling(100).min(); wt1_max = wt1.rolling(100).max()
    wt1_norm = 2*(wt1-wt1_min)/(wt1_max-wt1_min+1e-9)-1
    p1=wt1.shift(1); p2=wt2.shift(1)
    cross_up   = (p1<=p2)&(wt1>wt2)
    cross_down = (p1>=p2)&(wt1<wt2)
    return wt1,wt2,wt_diff,wt1_norm,wt1>=60,wt1<=-60,wt1>=53,wt1<=-53,cross_up,cross_down

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    cmf_val = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf_val

def tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    diff = series.diff()
    ema1 = diff.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()
    abs_diff = diff.abs()
    abs_ema1 = abs_diff.ewm(span=r, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=s, adjust=False).mean()
    tsi_val = 100 * (ema2 / abs_ema2)
    return tsi_val

def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )

def hull_ma(series: pd.Series, period: int) -> pd.Series:
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    raw = 2 * wma_half - wma_full
    hma = wma(raw, sqrt_n)
    return hma

def ut_bot_fn(close, high, low, kv=1.0, atr_p=10):
    pc  = close.shift(1)
    tr  = np.maximum(high-low, np.maximum((high-pc).abs(),(low-pc).abs()))
    atr = pd.Series(tr).ewm(alpha=1/atr_p,adjust=False).mean(); atr.index=close.index
    nl  = kv*atr
    src = close.values; nl_v = nl.values; n=len(src)
    xts = np.zeros(n)
    for i in range(1,n):
        if np.isnan(nl_v[i]): continue
        prev=xts[i-1]; s=src[i]; s1=src[i-1]
        if s>prev and s1>prev:   xts[i]=max(prev,s-nl_v[i])
        elif s<prev and s1<prev: xts[i]=min(prev,s+nl_v[i])
        elif s>prev:             xts[i]=s-nl_v[i]
        else:                    xts[i]=s+nl_v[i]
    pos=np.zeros(n,dtype=int)
    for i in range(1,n):
        ps=xts[i-1]; s=src[i]; s1=src[i-1]
        if s1<ps and s>ps:    pos[i]=1
        elif s1>ps and s<ps:  pos[i]=-1
        else:                 pos[i]=pos[i-1]
    xts_s=pd.Series(xts,index=close.index); ps=xts_s.shift(1); psrc=close.shift(1)
    above=(psrc<=ps)&(close>xts_s); below=(ps<=psrc)&(xts_s>close)
    return xts_s,pd.Series(pos,index=close.index),atr,nl,(close>xts_s)&above,(close<xts_s)&below,close>xts_s,close<xts_s


class RangeFilter:
    def cema(self,v,c,p):
        ev=np.nan; r=np.full_like(v,np.nan,dtype=float)
        for i in range(len(v)):
            if c[i] and not np.isnan(v[i]):
                ev=v[i] if np.isnan(ev) else (v[i]-ev)*2/(p+1)+ev
            r[i]=ev
        return r
    def csma(self,v,c,p):
        r=np.full_like(v,np.nan,dtype=float); buf=[]
        for i in range(len(v)):
            if c[i] and not np.isnan(v[i]):
                buf.append(v[i]);
                if len(buf)>p: buf.pop(0)
            if buf: r[i]=np.mean(buf)
        return r
    def true_range(self,h,l,c):
        pc=np.roll(c,1); pc[0]=c[0]
        return np.maximum(h-l,np.maximum(np.abs(h-pc),np.abs(l-pc)))
    def rsize(self,v,h,l,c,scale,qty,per):
        cond=np.ones_like(v,dtype=bool)
        if scale=="ATR": return qty*self.cema(self.true_range(h,l,c),cond,per)
        elif scale=="Average Change":
            ch=np.abs(v-np.roll(v,1)); ch[0]=0
            return qty*self.cema(ch,cond,per)
        return np.full_like(v,qty)
    def run(self,df):
        h=df['high'].values.astype(float); l=df['low'].values.astype(float)
        c=df['close'].values.astype(float)
        avg=(h+l)/2
        rs=self.rsize(avg,h,l,c,"Average Change",2.618,14)
        cond=np.ones_like(rs,dtype=bool)
        r=self.cema(rs,cond,27)
        rf=np.full(len(c),np.nan); rf[0]=(h[0]+l[0])/2
        for i in range(1,len(c)):
            rf[i]=rf[i-1]
            if c[i]-r[i]>rf[i-1]: rf[i]=c[i]-r[i]
            elif l[i]+r[i]<rf[i-1]: rf[i]=l[i]+r[i]
        hi=rf+r; lo=rf-r
        fc=np.zeros(len(rf),dtype=bool); fc[0]=True; fc[1:]=rf[1:]!=rf[:-1]
        rf_f=self.cema(rf,fc,2); hi_f=self.cema(hi,fc,2); lo_f=self.cema(lo,fc,2)
        fdir=np.zeros_like(rf_f); fv=0
        for i in range(1,len(rf_f)):
            if rf_f[i]>rf_f[i-1]: fv=1
            elif rf_f[i]<rf_f[i-1]: fv=-1
            fdir[i]=fv
        return rf_f, hi_f, lo_f, fdir


def ib_box(df_in):
    d=df_in.copy()
    d['IB_IsIB']=False; d['IB_BoxHigh']=np.nan; d['IB_BoxLow']=np.nan
    d['IB_GreenArrow']=False; d['IB_RedArrow']=False
    bh=bl=np.nan; bi=1; ff=False
    def isib(idx,lb):
        if idx<lb: return False
        ref=d.iloc[idx-lb]; cur=d.iloc[idx]
        return (cur['close']<=ref['high'] and cur['close']>=ref['low'] and
                cur['open']<=ref['high']  and cur['open']>=ref['low'])
    for i in range(1,len(d)):
        is_b=isib(i,bi); prev_b=d.iloc[i-1]['IB_IsIB']
        d.iloc[i,d.columns.get_loc('IB_IsIB')]=is_b
        if is_b and not prev_b:
            pb=d.iloc[i-1]; bh=pb['high']; bl=pb['low']; ff=True
            d.iloc[i,d.columns.get_loc('IB_BoxHigh')]=bh
            d.iloc[i,d.columns.get_loc('IB_BoxLow')]=bl; bi+=1
        elif is_b and prev_b:
            if not np.isnan(bh):
                d.iloc[i,d.columns.get_loc('IB_BoxHigh')]=bh
                d.iloc[i,d.columns.get_loc('IB_BoxLow')]=bl
            bi+=1
        elif prev_b and not is_b: bi=1
        else: ff=False
        if ff and not np.isnan(bh):
            pc=d.iloc[i-1]['close']; cc=d.iloc[i]['close']
            if pc<=bh and cc>bh: d.iloc[i,d.columns.get_loc('IB_GreenArrow')]=True
            elif pc>=bl and cc<bl: d.iloc[i,d.columns.get_loc('IB_RedArrow')]=True
    return d[['IB_IsIB','IB_BoxHigh','IB_BoxLow','IB_GreenArrow','IB_RedArrow']]


def fibonacci_levels(high, low, window=50):
    rh=high.rolling(window).max(); rl=low.rolling(window).min(); diff=rh-rl
    return {f'fib_{str(p).replace("0.","")}':(rh-p*diff)
            for p in [0.236,0.382,0.5,0.618,0.786]} | {'fib_high':rh,'fib_low':rl}


# ══════════════════════════════════════════════════════════════════════════════
#  RSI GAINZY — integrated feature
# ══════════════════════════════════════════════════════════════════════════════

class RSIGainzy:
    """
    RSI Gainzy Strategy — adds gainzy_color + numeric gainzy_trend to df.
    Colors: light_green (+3), green (+1), blue (+2), black (0), red (-1), pink (-3)
    """

    def calculate_rsi(self, close_prices: np.ndarray, period: int = 14) -> np.ndarray:
        close_prices = np.array(close_prices, dtype=float)
        deltas = np.diff(close_prices)
        deltas = np.concatenate([[0], deltas])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        rsi    = np.full(len(close_prices), np.nan)
        if len(close_prices) < period + 1:
            return rsi
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])
        rsi[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        for i in range(period + 1, len(close_prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i])  / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rsi[i]   = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        return rsi

    def _find_pivots(self, values: np.ndarray, left: int, right: int, mode: str) -> np.ndarray:
        """Unified pivot finder for highs ('high') and lows ('low')."""
        n      = len(values)
        pivots = np.full(n, np.nan)
        for i in range(left, n - right):
            if np.isnan(values[i]):
                continue
            pv       = values[i]
            is_pivot = True
            for j in range(i - left, i):
                if not np.isnan(values[j]):
                    if (mode == 'high' and values[j] >= pv) or \
                       (mode == 'low'  and values[j] <= pv):
                        is_pivot = False
                        break
            if is_pivot:
                for j in range(i + 1, i + right + 1):
                    if j < n and not np.isnan(values[j]):
                        if (mode == 'high' and values[j] >= pv) or \
                           (mode == 'low'  and values[j] <= pv):
                            is_pivot = False
                            break
            if is_pivot and i + right < n:
                pivots[i + right] = pv
        return pivots

    def calculate_gainzy_colors(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
        rsi_length: int = 14,
        pivot_length: int = 10,
    ) -> pd.Series:
        df           = df.copy()
        close_prices = df[close_col].values
        n            = len(close_prices)
        rsi          = self.calculate_rsi(close_prices, rsi_length)
        rsi_ph       = self._find_pivots(rsi, pivot_length, pivot_length, 'high')
        rsi_pl       = self._find_pivots(rsi, pivot_length, pivot_length, 'low')

        prev_high_val = prev_high_bar = last_high_val = last_high_bar = np.nan
        prev_low_val  = prev_low_bar  = last_low_val  = last_low_bar  = np.nan
        trend         = 0
        trend_results = np.zeros(n, dtype=int)
        hl_pts        = np.full(n, np.nan)   # high-line history
        ll_pts        = np.full(n, np.nan)   # low-line  history

        for i in range(n):
            # ── detect new pivots ──────────────────────────────────────────
            last_rsi_pivot_high = last_rsi_pivot_low = np.nan
            pbi = i - pivot_length
            if pbi >= 0 and pbi + 1 < len(rsi):
                if not np.isnan(rsi_ph[i]) and rsi[pbi] != rsi[pbi + 1]:
                    last_rsi_pivot_high = rsi_ph[i]
                if not np.isnan(rsi_pl[i]) and rsi[pbi] != rsi[pbi + 1]:
                    last_rsi_pivot_low  = rsi_pl[i]

            # ── update pivot trackers ──────────────────────────────────────
            if not np.isnan(last_rsi_pivot_high):
                prev_high_val, prev_high_bar = last_high_val, last_high_bar
                last_high_val, last_high_bar = last_rsi_pivot_high, i - pivot_length
            if not np.isnan(last_rsi_pivot_low):
                prev_low_val,  prev_low_bar  = last_low_val,  last_low_bar
                last_low_val,  last_low_bar  = last_rsi_pivot_low,  i - pivot_length

            # ── compute extended trend-line values ─────────────────────────
            def _extend(pv, pb, lv, lb):
                """Extend the pivot line to bar i, return value at i."""
                if any(np.isnan(x) for x in [pv, pb, lv, lb]) or lb == pb:
                    return pv
                j = int(lb)
                while j < i:
                    nxt = pv + (lv - pv) * (j + 1 - pb) / (lb - pb)
                    if nxt >= 100 or nxt <= 0:
                        break
                    j += 1
                return pv + (lv - pv) * (j - pb) / (lb - pb)

            cur_hl = _extend(prev_high_val, prev_high_bar, last_high_val, last_high_bar)
            cur_ll = _extend(prev_low_val,  prev_low_bar,  last_low_val,  last_low_bar)
            hl_pts[i] = cur_hl
            ll_pts[i] = cur_ll

            # ── carry forward trend ────────────────────────────────────────
            if i > 0:
                trend = trend_results[i - 1]
            if np.isnan(rsi[i]):
                trend_results[i] = trend
                continue

            cr  = rsi[i]
            chl = cur_hl if not np.isnan(cur_hl) else 0
            cll = cur_ll if not np.isnan(cur_ll) else 0
            phl = hl_pts[i-1] if i > 0 and not np.isnan(hl_pts[i-1]) else 0
            pll = ll_pts[i-1] if i > 0 and not np.isnan(ll_pts[i-1]) else 0
            nph = np.isnan(last_rsi_pivot_high)
            npl = np.isnan(last_rsi_pivot_low)

            # ── bullish conditions ─────────────────────────────────────────
            if   cr > chl and chl > cll and nph and chl > phl and trend <= 0: trend = 3
            elif cr > cll and chl < cll and nph and cll > pll and trend <= 0: trend = 3
            elif cr > chl and chl > cll and nph and trend < 3:                trend = 1
            elif cr > cll and chl < cll and nph and trend < 3:                trend = 1
            elif cr > chl and cr < cll  and chl < cll:                        trend = 2

            if cr < chl and trend > 0:  trend = 0   # reset bullish

            # ── bearish conditions ─────────────────────────────────────────
            if   cr < cll and chl > cll and npl and cll < pll and trend >= 0 and trend != 2: trend = -3
            elif cr < chl and chl < cll and npl and cll < pll and trend >= 0 and trend != 2: trend = -3
            elif cr < cll and chl > cll and npl and trend > -3:                              trend = -1
            elif cr < chl and chl < cll and npl and trend > -3:                              trend = -1
            elif cr < cll and cr > chl  and chl < cll:                                       trend = 2

            if cr > cll and trend < 0:  trend = 0   # reset bearish

            trend_results[i] = trend

        color_map = {3: 'light_green', 1: 'green', 2: 'blue', 0: 'black', -1: 'red', -3: 'pink'}
        colors = pd.Series([color_map.get(t, 'black') for t in trend_results], index=df.index)
        return colors, pd.Series(trend_results, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline and return enriched DataFrame."""
    required = ["open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.startswith('unnamed')]]

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isna().all():
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time']).set_index('time')

    df = df.sort_index()

    # ── Heikin Ashi ────────────────────────────────────────────────────────
    df['ha_open'],df['ha_high'],df['ha_low'],df['ha_close'] = heikin_ashi(df)

    # ── Trend: SMA / EMA ──────────────────────────────────────────────────
    for w in [10,20,50,200]:
        df[f'sma_{w}']=ta.trend.sma_indicator(df['close'],window=w)
    for w in [9,21,50,200]:
        df[f'ema_{w}']=ta.trend.ema_indicator(df['close'],window=w)

    # ── MACD ──────────────────────────────────────────────────────────────
    m=ta.trend.MACD(df['close'],26,12,9)
    df['macd']=m.macd(); df['macd_signal']=m.macd_signal(); df['macd_hist']=m.macd_diff()

    # ── ADX ───────────────────────────────────────────────────────────────
    adx=ta.trend.ADXIndicator(df['high'],df['low'],df['close'],14)
    df['adx']=adx.adx(); df['adx_pos']=adx.adx_pos(); df['adx_neg']=adx.adx_neg()

    # ── Ichimoku ──────────────────────────────────────────────────────────
    ich=ta.trend.IchimokuIndicator(df['high'],df['low'],9,26,52)
    df['ichimoku_conversion']=ich.ichimoku_conversion_line()
    df['ichimoku_base']=ich.ichimoku_base_line()
    df['ichimoku_a']=ich.ichimoku_a(); df['ichimoku_b']=ich.ichimoku_b()

    # ── PSAR ──────────────────────────────────────────────────────────────
    ps=ta.trend.PSARIndicator(df['high'],df['low'],df['close'])
    df['psar']=ps.psar(); df['psar_up']=ps.psar_up(); df['psar_down']=ps.psar_down()

    # ── Aroon / Vortex / CCI / DPO ───────────────────────────────────────
    ar=ta.trend.AroonIndicator(df['high'],df['low'],25)
    df['aroon_up']=ar.aroon_up(); df['aroon_down']=ar.aroon_down(); df['aroon_ind']=ar.aroon_indicator()
    vx=ta.trend.VortexIndicator(df['high'],df['low'],df['close'],14)
    df['vortex_pos']=vx.vortex_indicator_pos(); df['vortex_neg']=vx.vortex_indicator_neg()
    df['cci']=ta.trend.cci(df['high'],df['low'],df['close'],20)

    # ── Momentum ──────────────────────────────────────────────────────────
    df['rsi_14']=ta.momentum.rsi(df['close'],14)
    df['rsi_7']=ta.momentum.rsi(df['close'],7)
    st=ta.momentum.StochasticOscillator(df['high'],df['low'],df['close'],14,3)
    df['stoch_k']=st.stoch(); df['stoch_d']=st.stoch_signal()
    sr=ta.momentum.StochRSIIndicator(df['close'],14,3,3)
    df['stoch_rsi_k']=sr.stochrsi_k(); df['stoch_rsi_d']=sr.stochrsi_d()
    df['williams_r']=ta.momentum.williams_r(df['high'],df['low'],df['close'],14)
    df['roc_9']=ta.momentum.roc(df['close'],9); df['roc_21']=ta.momentum.roc(df['close'],21)
    df['ultimate_osc']=ta.momentum.ultimate_oscillator(df['high'],df['low'],df['close'],7,14,28)

    # ── AlphaTrend ────────────────────────────────────────────────────────────
    df = alpha_trend(df, coeff=1.0, ap=14, no_volume=False)

    # ── Chandelier Exit ───────────────────────────────────────────────────────
    df = chandelier_exit(df, length=22, mult=3.0, use_close=True)

    # ── QQE Signals ───────────────────────────────────────────────────────────
    df = qqe_signals(df, rsi_period=14, sf=5, qqe_factor=4.238)

    # ── HalfTrend ─────────────────────────────────────────────────────────────
    df = half_trend(df, amplitude=2, channel_deviation=2)

    # ── RMI Trend Sniper ──────────────────────────────────────────────────────
    df = rmi_trend_sniper(df, length=14, pmom=66, nmom=30)

    # ── Volatility ────────────────────────────────────────────────────────
    bb=ta.volatility.BollingerBands(df['close'],20,2)
    df['bb_upper']=bb.bollinger_hband(); df['bb_middle']=bb.bollinger_mavg()
    df['bb_lower']=bb.bollinger_lband(); df['bb_width']=bb.bollinger_wband()
    df['bb_pct']=bb.bollinger_pband()
    df['bb_hband_ind']=bb.bollinger_hband_indicator()
    df['bb_lband_ind']=bb.bollinger_lband_indicator()
    df['atr_14']=ta.volatility.average_true_range(df['high'],df['low'],df['close'],14)
    df['atr_7']=ta.volatility.average_true_range(df['high'],df['low'],df['close'],7)
    kc=ta.volatility.KeltnerChannel(df['high'],df['low'],df['close'],20)
    df['kc_upper']=kc.keltner_channel_hband(); df['kc_middle']=kc.keltner_channel_mband()
    df['kc_lower']=kc.keltner_channel_lband(); df['kc_width']=kc.keltner_channel_wband()
    df['kc_pct']=kc.keltner_channel_pband()
    df['kc_hband_ind']=kc.keltner_channel_hband_indicator()
    df['kc_lband_ind']=kc.keltner_channel_lband_indicator()
    dc=ta.volatility.DonchianChannel(df['high'],df['low'],df['close'],20)
    df['dc_upper']=dc.donchian_channel_hband(); df['dc_middle']=dc.donchian_channel_mband()
    df['dc_lower']=dc.donchian_channel_lband(); df['dc_width']=dc.donchian_channel_wband()
    df['dc_pct']=dc.donchian_channel_pband()

    # ── Volume ────────────────────────────────────────────────────────────
    df['obv']=ta.volume.on_balance_volume(df['close'],df['volume'])
    df['vwap']=ta.volume.volume_weighted_average_price(df['high'],df['low'],df['close'],df['volume'],14)
    df['mfi']=ta.volume.money_flow_index(df['high'],df['low'],df['close'],df['volume'],14)
    df['adi']=ta.volume.acc_dist_index(df['high'],df['low'],df['close'],df['volume'])

    # ── Custom MAs ────────────────────────────────────────────────────────
    df['hull_ma_9']=_hull_ma(df['close'],9)
    df['tema_9']=_tema(df['close'],9)
    df['vwma_20']=_vwma(df['close'],df['volume'],20)
    df['wma_20']=_wma(df['close'],20)

    # ── Supertrend ────────────────────────────────────────────────────────
    df['supertrend'],df['supertrend_direction']=supertrend(df['high'],df['low'],df['close'])

    # ── Add-ons ───────────────────────────────────────────────────────────
    df["tema_21"]    = tema(df["close"], 21)
    df["hull_ma_20"] = hull_ma(df["close"], 20)
    df["tsi"]        = tsi(df["close"])
    df["cmf"]        = cmf(df)

    # ── Williams Vix Fix ──────────────────────────────────────────────────
    (df['wvf'],df['wvf_upper'],df['wvf_lower'],
     df['wvf_range_high'],df['wvf_range_low'],
     df['wvf_alert1'],df['wvf_alert2']) = williams_vix_fix(df['close'],df['low'])

    # ── WaveTrend ─────────────────────────────────────────────────────────
    (df['wt1'],df['wt2'],df['wt_diff'],df['wt1_norm'],
     df['wt_overbought'],df['wt_oversold'],df['wt_ob_mild'],df['wt_os_mild'],
     df['wt_cross_up'],df['wt_cross_down']) = wavetrend(df['high'],df['low'],df['close'])

    # ── UT Bot ────────────────────────────────────────────────────────────
    (df['ut_trailing_stop'],df['ut_pos'],df['ut_atr'],df['ut_n_loss'],
     df['ut_buy'],df['ut_sell'],df['ut_bar_buy'],df['ut_bar_sell']) = ut_bot_fn(df['close'],df['high'],df['low'])

    # ── Range Filter ──────────────────────────────────────────────────────
    rf=RangeFilter()
    df['RF_Filter'],df['RF_UpperBand'],df['RF_LowerBand'],df['RF_Trend'] = rf.run(df)
    df['RF_BuySignal']=(df['RF_Trend']==1).astype(int)
    df['RF_SellSignal']=(df['RF_Trend']==-1).astype(int)

    # ── Inside Bar ────────────────────────────────────────────────────────
    ib=ib_box(df)
    for c in ib.columns: df[c]=ib[c].values

    # ── Fibonacci ─────────────────────────────────────────────────────────
    for k,v in fibonacci_levels(df['high'],df['low']).items():
        df[k]=v

    # ── Derived features ──────────────────────────────────────────────────
    df['delta_normalized']=(df['close']-df['open'])/(df['high']-df['low']+1e-9)*df['volume']
    df['candle_body']=df['close']-df['open']
    df['candle_range']=df['high']-df['low']
    df['is_bullish']=(df['close']>df['open']).astype(int)
    df['volume_sma_20']=df['volume'].rolling(20).mean()
    df['volume_ratio']=df['volume']/df['volume_sma_20']
    df['return_1']=df['close'].pct_change(1)
    df['squeeze']=((df['bb_upper']<df['kc_upper'])&(df['bb_lower']>df['kc_lower'])).astype(int)

    # ══════════════════════════════════════════════════════════════════════
    #  RSI GAINZY — compute and attach to DataFrame
    # ══════════════════════════════════════════════════════════════════════
    gainzy          = RSIGainzy()
    gainzy_color, gainzy_trend = gainzy.calculate_gainzy_colors(
        df, close_col='close', rsi_length=14, pivot_length=10
    )
    df['gainzy_color'] = gainzy_color           # string label (for readability)
    df['gainzy_trend'] = gainzy_trend           # integer: -3,-1,0,1,2,3

    # ── boolean helper columns (signal-ready) ─────────────────────────────
    # Strong bullish: light_green (+3)
    df['gainzy_strong_bull']  = (df['gainzy_trend'] == 3).astype(int)
    # Any bullish: green or light_green (+1, +3)
    df['gainzy_any_bull']     = (df['gainzy_trend'] > 0).astype(int)
    # Momentum zone: blue (+2) — RSI between the two trend lines
    df['gainzy_momentum']     = (df['gainzy_trend'] == 2).astype(int)
    # Strong bearish: pink (-3)
    df['gainzy_strong_bear']  = (df['gainzy_trend'] == -3).astype(int)
    # Any bearish: red or pink (-1, -3)
    df['gainzy_any_bear']     = (df['gainzy_trend'] < 0).astype(int)
    # Trend just flipped bullish (0 → positive)
    df['gainzy_bull_flip']    = (
        (df['gainzy_trend'] > 0) & (df['gainzy_trend'].shift(1) <= 0)
    ).astype(int)
    # Trend just flipped bearish (0 → negative)
    df['gainzy_bear_flip']    = (
        (df['gainzy_trend'] < 0) & (df['gainzy_trend'].shift(1) >= 0)
    ).astype(int)

    # Drop helper capitalised cols
    df.drop(columns=['Open','High','Low','Close'], inplace=True, errors='ignore')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.bfill().ffill()
    # Drop rows where > 40% features are NaN (warm-up period)
    df.dropna(thresh=int(len(df.columns)*0.60), inplace=True)
    df.reset_index(inplace=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER — top-level so multiprocessing can pickle it
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_one(args):
    strat, df = args
    metrics = run_backtest(strat, df)
    if metrics is None:
        return None
    if not filter_metrics(metrics):
        return None
    return {
        "strategy": strat,
        "metrics":  metrics,
        "score":    score_strategy(metrics),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(
    csv_path: str,
    n_strategies: int  = 1000,
    top_n: int         = 30,
    n_workers: int     = 4,
    output_csv: str    = "strategy_results.csv",
    seed: int          = 42,
):
    print(f"\n{'═'*60}")
    print("  STRATEGY FACTORY — Alpha Generator")
    print(f"{'═'*60}")

    # ── 1. Load & engineer ─────────────────────────────────────────────
    print(f"\n[1/6] Loading data from: {csv_path}")
    raw = pd.read_csv(csv_path)
    print(f"      Raw shape: {raw.shape}")

    print("[1/6] Running feature engineering …")
    t0 = time.time()
    df = feature_engineer(raw)
    print(f"      Engineered shape: {df.shape}  ({time.time()-t0:.1f}s)")

    # ── Signal validation ──────────────────────────────────────────────
    from signals import SIGNALS
    print("\n[DEBUG] Validating signals...")
    broken = 0
    for name, meta in SIGNALS.items():
        try:
            out = meta["fn"](df)
            if not isinstance(out, pd.Series):
                print(f"SIGNAL NOT SERIES: {name}")
                broken += 1
        except Exception as e:
            print(f"SIGNAL BROKEN: {name} → {e}")
            broken += 1
    print(f"[DEBUG] Broken signals: {broken}")

    if len(df) < 300:
        print("⚠  WARNING: fewer than 300 rows after feature engineering.")

    # ── 2. Train / test split ──────────────────────────────────────────
    print("\n[2/6] Splitting train (80%) / test (20%) …")
    train_df, test_df = train_test_split(df, 0.80)
    print(f"      Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    # ── 3. Generate strategies ─────────────────────────────────────────
    print(f"\n[3/6] Generating {n_strategies} strategies (seed={seed}) …")
    t0 = time.time()
    strategies = generate_strategies(n=n_strategies, seed=seed)
    print(f"      Generated: {len(strategies)}  ({time.time()-t0:.1f}s)")

    # ── 4. Evaluate on TRAIN ───────────────────────────────────────────
    print(f"\n[4/6] Backtesting on TRAIN with {n_workers} workers …")
    t0 = time.time()
    raw_results = []
    for i, s in enumerate(strategies):
        debug   = i < 3
        metrics = run_backtest(s, train_df, debug=debug)
        if metrics is None:
            continue
        if not filter_metrics(metrics):
            continue
        raw_results.append({
            "strategy": s,
            "metrics":  metrics,
            "score":    score_strategy(metrics),
        })
    passed  = raw_results
    elapsed = time.time() - t0
    print(f"      Evaluated: {len(strategies)}  |  Passed filter: {len(passed)}  ({elapsed:.1f}s)")

    if not passed:
        print("\n✗ No strategies passed the filter.")
        print("Try relaxing filters in evaluator.py or supplying more data.")
        return

    # ── 5. Rank ────────────────────────────────────────────────────────
    print(f"\n[5/6] Ranking — keeping top {top_n} …")
    top = rank_strategies(passed, top_n=top_n)
    print_summary(top, title="TOP STRATEGIES (TRAIN)")

    # ── 6. Validate on TEST ────────────────────────────────────────────
    print(f"\n[6/6] Validating top {len(top)} strategies on TEST set …")
    t0 = time.time()
    validated = validate_on_test(top, test_df)
    print(f"      Validated: {len(validated)} strategies consistent  ({time.time()-t0:.1f}s)")

    if validated:
        print_summary(validated, title="VALIDATED STRATEGIES")

    # ── 7. Save ────────────────────────────────────────────────────────
    save_target = validated if validated else top
    out_df = save_results(save_target, path=output_csv)
    print(f"\n✓ Results saved → {output_csv}  ({len(out_df)} rows)")
    print(f"{'═'*60}\n")

    return save_target


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy Factory")
    parser.add_argument("--csv",     default=r"C:\Users\vaibh\OneDrive\Desktop\multi_strategy_generator\data\ethusd_15m.csv", help="Path to OHLCV CSV")
    parser.add_argument("--n",       type=int, default=1000,  help="Strategies to generate")
    parser.add_argument("--top",     type=int, default=10,    help="Top N to validate")
    parser.add_argument("--workers", type=int, default=4,     help="Parallel workers")
    parser.add_argument("--out",     default=r"C:\Users\vaibh\OneDrive\Desktop\multi_strategy_generator\results\strategy_results_15m_top10_lookback_commission_standard_new_data_ethusdt.csv", help="Output CSV path")
    parser.add_argument("--seed",    type=int, default=314,   help="Random seed")
    args = parser.parse_args()

    run(
        csv_path     = args.csv,
        n_strategies = args.n,
        top_n        = args.top,
        n_workers    = args.workers,
        output_csv   = args.out,
        seed         = args.seed,
    )