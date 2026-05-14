"""
regime_detection.py
-------------------
Detect market regimes from OHLCV data using Gaussian Naive Bayes.

The detector creates rolling volatility, trend, range, and volume features,
builds deterministic seed labels, then trains GaussianNB to classify each bar
as one of: volatile, trendy, chop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REGIME_VOLATILE = "volatile"
REGIME_TRENDY = "trendy"
REGIME_CHOP = "chop"
REGIME_ORDER = [REGIME_CHOP, REGIME_TRENDY, REGIME_VOLATILE]
REGIME_TO_ID = {name: i for i, name in enumerate(REGIME_ORDER)}

FEATURE_COLUMNS = [
    "regime_return_1",
    "regime_realized_vol",
    "regime_atr_pct",
    "regime_range_pct",
    "regime_bb_width",
    "regime_trend_strength",
    "regime_efficiency_ratio",
    "regime_volume_z",
]


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 50
    fast_window: int = 20
    slow_window: int = 100
    volatility_quantile: float = 0.70
    trend_quantile: float = 0.60
    chop_quantile: float = 0.45
    min_train_rows: int = 120


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data[[c for c in data.columns if not c.lower().startswith("unnamed")]]

    # Normalize only the raw OHLCV/time fields. Engineered signal columns are
    # case-sensitive elsewhere in the project, e.g. RF_Trend and IB_IsIB.
    rename_map = {}
    seen_lower = {c.lower(): c for c in data.columns}
    for name in ["time", "open", "high", "low", "close", "volume"]:
        original = seen_lower.get(name)
        if original is not None and original != name:
            rename_map[original] = name
    if rename_map:
        data = data.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for regime detection: {missing}")

    if "time" in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        parsed = pd.to_datetime(data["time"], unit="ms", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(data["time"], errors="coerce")
        if not parsed.isna().all():
            data = data.assign(time=parsed).dropna(subset=["time"]).set_index("time")

    return data.sort_index()


def compute_regime_features(
    df: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.DataFrame:
    """Return OHLCV data with rolling features used by the regime model."""
    cfg = config or RegimeConfig()
    data = _clean_ohlcv(df)

    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    returns = close.pct_change()
    prev_close = close.shift(1)

    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    min_periods = max(5, cfg.lookback // 4)
    atr = true_range.rolling(cfg.lookback, min_periods=min_periods).mean()
    sma = close.rolling(cfg.lookback, min_periods=min_periods).mean()
    std = close.rolling(cfg.lookback, min_periods=min_periods).std()
    fast_ma = close.ewm(span=cfg.fast_window, adjust=False).mean()
    slow_ma = close.ewm(span=cfg.slow_window, adjust=False).mean()
    path = close.diff().abs().rolling(cfg.lookback, min_periods=min_periods).sum()
    distance = (close - close.shift(cfg.lookback)).abs()
    volume_mean = volume.rolling(cfg.lookback, min_periods=min_periods).mean()
    volume_std = volume.rolling(cfg.lookback, min_periods=min_periods).std()

    out = data.copy()
    out["regime_return_1"] = returns
    out["regime_realized_vol"] = returns.rolling(cfg.lookback, min_periods=min_periods).std()
    out["regime_atr_pct"] = atr / close.replace(0, np.nan)
    out["regime_range_pct"] = (high - low) / close.replace(0, np.nan)
    out["regime_bb_width"] = (4 * std) / sma.replace(0, np.nan)
    out["regime_trend_strength"] = (fast_ma - slow_ma).abs() / close.replace(0, np.nan)
    out["regime_efficiency_ratio"] = distance / path.replace(0, np.nan)
    out["regime_volume_z"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def _quantile(series: pd.Series, q: float) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(q))


def seed_regime_labels(
    feature_df: pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Create heuristic labels that GaussianNB learns to generalize."""
    cfg = config or RegimeConfig()
    vol_score = feature_df[["regime_realized_vol", "regime_atr_pct", "regime_bb_width"]].rank(pct=True).mean(axis=1)
    trend_score = feature_df[["regime_trend_strength", "regime_efficiency_ratio"]].rank(pct=True).mean(axis=1)
    chop_score = 1.0 - feature_df["regime_efficiency_ratio"].rank(pct=True)

    volatile_cut = _quantile(vol_score, cfg.volatility_quantile)
    trend_cut = _quantile(trend_score, cfg.trend_quantile)
    chop_cut = _quantile(chop_score, cfg.chop_quantile)

    labels = pd.Series(REGIME_CHOP, index=feature_df.index, dtype="object")
    labels.loc[trend_score >= trend_cut] = REGIME_TRENDY
    labels.loc[vol_score >= volatile_cut] = REGIME_VOLATILE
    labels.loc[(chop_score >= chop_cut) & (vol_score < volatile_cut)] = REGIME_CHOP
    labels.loc[feature_df[FEATURE_COLUMNS].isna().any(axis=1)] = np.nan
    return labels


def detect_regimes(
    df: pd.DataFrame,
    config: RegimeConfig | None = None,
    include_features: bool = True,
) -> pd.DataFrame:
    """Add regime columns to a DataFrame using GaussianNB predictions."""
    cfg = config or RegimeConfig()
    features = compute_regime_features(df, cfg)
    seed_labels = seed_regime_labels(features, cfg)
    train_mask = seed_labels.notna() & features[FEATURE_COLUMNS].notna().all(axis=1)

    if int(train_mask.sum()) < cfg.min_train_rows:
        raise ValueError(
            f"Need at least {cfg.min_train_rows} labeled rows for regime detection; "
            f"got {int(train_mask.sum())}."
        )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("nb", GaussianNB()),
        ]
    )
    model.fit(features.loc[train_mask, FEATURE_COLUMNS], seed_labels.loc[train_mask])

    predict_mask = features[FEATURE_COLUMNS].notna().all(axis=1)
    predictions = pd.Series(np.nan, index=features.index, dtype="object")
    probabilities = pd.DataFrame(0.0, index=features.index, columns=[f"regime_prob_{r}" for r in REGIME_ORDER])

    predictions.loc[predict_mask] = model.predict(features.loc[predict_mask, FEATURE_COLUMNS])
    prob_values = model.predict_proba(features.loc[predict_mask, FEATURE_COLUMNS])
    classes = list(model.named_steps["nb"].classes_)
    for class_idx, class_name in enumerate(classes):
        probabilities.loc[predict_mask, f"regime_prob_{class_name}"] = prob_values[:, class_idx]

    output = features.copy() if include_features else _clean_ohlcv(df)
    output["market_regime"] = predictions.ffill().bfill().fillna(REGIME_CHOP)
    output["regime_id"] = output["market_regime"].map(REGIME_TO_ID).astype(int)
    output["regime_confidence"] = probabilities.max(axis=1)
    output["regime_seed_label"] = seed_labels

    for col in probabilities.columns:
        output[col] = probabilities[col]

    if not include_features:
        output = output.drop(columns=[c for c in FEATURE_COLUMNS if c in output.columns], errors="ignore")

    return output


def summarize_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize distribution and basic behavior of detected regimes."""
    if "market_regime" not in df.columns:
        df = detect_regimes(df)

    data = df.copy()
    if "regime_return_1" not in data.columns:
        data["regime_return_1"] = data["close"].pct_change()

    summary = (
        data.groupby("market_regime")
        .agg(
            bars=("market_regime", "size"),
            avg_return=("regime_return_1", "mean"),
            avg_abs_return=("regime_return_1", lambda s: s.abs().mean()),
            avg_atr_pct=("regime_atr_pct", "mean"),
            avg_trend_strength=("regime_trend_strength", "mean"),
            avg_efficiency=("regime_efficiency_ratio", "mean"),
            avg_confidence=("regime_confidence", "mean"),
        )
        .reindex(REGIME_ORDER)
    )
    summary["pct_bars"] = summary["bars"] / summary["bars"].sum()
    return summary.reset_index()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect OHLCV market regimes with GaussianNB")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--out", default="", help="Optional output CSV with regime columns")
    parser.add_argument("--lookback", type=int, default=50, help="Rolling lookback for regime features")
    parser.add_argument("--tail", type=int, default=10, help="Rows of labeled data to print")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = pd.read_csv(args.csv)
    cfg = RegimeConfig(lookback=args.lookback)
    detected = detect_regimes(raw, cfg)
    summary = summarize_regimes(detected)

    print("\nRegime summary")
    print(summary.to_string(index=False))
    print("\nLatest regimes")
    print(detected[["market_regime", "regime_confidence"]].tail(args.tail).to_string())

    if args.out:
        detected.reset_index().to_csv(args.out, index=False)
        print(f"\nSaved regime output -> {args.out}")


if __name__ == "__main__":
    main()
