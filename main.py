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
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

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
import pandas as pd

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

    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf

def tsi(series: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    diff = series.diff()

    ema1 = diff.ewm(span=r, adjust=False).mean()
    ema2 = ema1.ewm(span=s, adjust=False).mean()

    abs_diff = diff.abs()
    abs_ema1 = abs_diff.ewm(span=r, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=s, adjust=False).mean()

    tsi = 100 * (ema2 / abs_ema2)
    return tsi

import numpy as np
import pandas as pd

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


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering pipeline and return enriched DataFrame."""
    required = ["open","high","low","close","volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    # Ensure lowercase columns
    df.columns = [c.lower().strip() for c in df.columns]
    # Drop unnamed index columns if present
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
    # df['cmf']=ta.volume.chaikin_money_flow(df['high'],df['low'],df['close'],df['volume'],20)
    df['adi']=ta.volume.acc_dist_index(df['high'],df['low'],df['close'],df['volume'])

    # ── Custom MAs ────────────────────────────────────────────────────────
    df['hull_ma_9']=_hull_ma(df['close'],9)
    df['tema_9']=_tema(df['close'],9)
    df['vwma_20']=_vwma(df['close'],df['volume'],20)
    df['wma_20']=_wma(df['close'],20)

    # ── Supertrend ────────────────────────────────────────────────────────
    df['supertrend'],df['supertrend_direction']=supertrend(df['high'],df['low'],df['close'])

    # add ons
    df["tema_21"]     = tema(df["close"], 21)
    df["hull_ma_20"]  = hull_ma(df["close"], 20)
    df["tsi"]         = tsi(df["close"])
    df["cmf"]         = cmf(df)

    # df = df.replace([np.inf, -np.inf], np.nan).bfill()    

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
        "metrics": metrics,
        "score": score_strategy(metrics),  # ✅ THIS LINE IS MISSING IN YOUR CODE
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

    # 🔥 SIGNAL VALIDATION (CRITICAL)
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
            print(f"SIGNAL BROKEN: {name}")
            print(e)
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

    # ── 4. Evaluate on TRAIN (parallel) ───────────────────────────────
    print(f"\n[4/6] Backtesting on TRAIN with {n_workers} workers …")
    t0 = time.time()

    args = [(s, train_df) for s in strategies]
    raw_results = []
    for i, s in enumerate(strategies):
        print(f"\nRunning strategy {i}")
        
        # 🔥 force debug ON for first few
        debug = i < 3
        
        # result = run_backtest(s, train_df, debug=debug)
        # raw_results.append(result)
        metrics = run_backtest(s, train_df, debug=debug)
        if metrics is None:
            continue
        if not filter_metrics(metrics):
            continue
        result = {
            "strategy": s,
            "metrics": metrics,
            "score": score_strategy(metrics),
        }
        raw_results.append(result)
    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=n_workers) as pool:
    #     raw_results = pool.map(_evaluate_one, args, chunksize=10)

    # passed = [r for r in raw_results if r is not None]
    passed = raw_results
    elapsed = time.time() - t0

    print(f"      Evaluated: {len(strategies)}  |  Passed filter: {len(passed)}  ({elapsed:.1f}s)")

    if not passed:
        print("\n✗ No strategies passed the filter.")
        print("Try:")
        print("  • More data (5000+ rows)")
        print("  • Relax filters in evaluator.py")
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
    parser.add_argument("--csv",     default=r"C:\Users\vaibh\OneDrive\Desktop\multi_strategy_generator\data\btcusd_60m.csv",help="Path to OHLCV CSV")
    parser.add_argument("--n",       type=int,   default=100000,         help="Strategies to generate")
    parser.add_argument("--top",     type=int,   default=100,          help="Top N to validate")
    parser.add_argument("--workers", type=int,   default=4,           help="Parallel workers")
    parser.add_argument("--out",     default="strategy_results_60m_top100_lookback_commission_standard_new_data_btcusdt.csv",  help="Output CSV path")
    parser.add_argument("--seed",    type=int,   default=314,          help="Random seed")
    args = parser.parse_args()

    run(
        csv_path     = args.csv,
        n_strategies = args.n,
        top_n        = args.top,
        n_workers    = args.workers,
        output_csv   = args.out,
        seed         = args.seed,
    )