from __future__ import annotations

import json
import math
import os
import pickle
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PREDICTION_CACHE_SECONDS = 60
MODEL_VERSION = "bilstm-local-v1"

FEATURE_COLUMNS = [
    "return_1",
    "candle_body",
    "candle_range",
    "is_bullish",
    "rsi_14",
    "rsi_7",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "bb_width",
    "atr_14",
    "atr_7",
    "mfi",
    "obv",
    "cmf",
    "tsi",
    "volume_ratio",
    "supertrend_direction",
    "gainzy_trend",
    "gainzy_any_bull",
    "gainzy_any_bear",
    "ut_pos",
    "RF_Trend",
    "squeeze",
    "close_vs_1h_ema20",
    "close_vs_4h_ema20",
    "context_1h_return_1",
    "context_4h_return_1",
]


try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

import numpy as np
import pandas as pd


if nn is not None:
    class BiLSTMClassifier(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])
else:
    BiLSTMClassifier = None


from fetch_data import get_klines
from feature_engineering import cached_feature_engineer


def _csv_path(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{symbol.lower()}_{timeframe.lower()}.csv"


def load_market_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    path = _csv_path(symbol, timeframe)
    if path.exists():
        return pd.read_csv(path)

    interval = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}.get(timeframe.lower(), timeframe)
    df = get_klines(symbol.upper(), interval, limit=limit)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _context_features(symbol: str, base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for tf in ("1h", "4h"):
        try:
            ctx = cached_feature_engineer(load_market_data(symbol, tf), symbol, tf)
        except Exception:
            continue
        if "ema_20" not in ctx.columns:
            ctx["ema_20"] = ctx["close"].ewm(span=20, adjust=False).mean()
        keep = [c for c in ("time", "return_1", "ema_20") if c in ctx.columns]
        if "time" not in keep:
            continue
        ctx = ctx[keep].rename(
            columns={
                "return_1": f"context_{tf}_return_1",
                "ema_20": f"context_{tf}_ema20",
            }
        )
        out = pd.merge_asof(
            out.sort_values("time"),
            ctx.sort_values("time"),
            on="time",
            direction="backward",
        )
        ema_col = f"context_{tf}_ema20"
        if ema_col in out.columns:
            out[f"close_vs_{tf}_ema20"] = out["close"] / out[ema_col].replace(0, np.nan) - 1
    return out


def prepare_realtime_features(symbol: str, timeframe: str = "15m") -> pd.DataFrame:
    raw = load_market_data(symbol, timeframe)
    features = cached_feature_engineer(raw, symbol, timeframe)
    if "time" in features.columns:
        features = _context_features(symbol, features)
    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0
    features = features.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    return features


def get_news_sentiment_cached(symbol: str) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"news_sentiment_{symbol.lower()}.json"
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("date") == today:
            return cached
    payload = {
        "symbol": symbol.lower(),
        "date": today,
        "sentiment": "neutral",
        "score": 0.0,
        "source": "daily_local_cache",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _weights_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / "bilstm_probabilities.pt"


def _scaler_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / "bilstm_scaler.pkl"


def _load_torch_model(n_features: int):
    if torch is None or BiLSTMClassifier is None or not _weights_path().exists():
        return None
    model = BiLSTMClassifier(n_features=n_features)
    model.load_state_dict(torch.load(_weights_path(), map_location="cpu"))
    model.eval()
    return model


def _keras_model_path() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / "bilstm_probabilities.keras"


def _has_local_model() -> bool:
    return _keras_model_path().exists() or _weights_path().exists()


def _heuristic_probabilities(features: pd.DataFrame) -> tuple[float, float]:
    tail = features.tail(30)
    momentum = float(tail["return_1"].mean()) if "return_1" in tail else 0.0
    trend = float(tail.get("close_vs_1h_ema20", pd.Series([0.0])).iloc[-1])
    context = float(tail.get("context_4h_return_1", pd.Series([0.0])).mean())
    risk = float(tail.get("atr_14", pd.Series([0.0])).iloc[-1] / max(tail["close"].iloc[-1], 1e-9))
    score = 9.0 * momentum + 3.5 * trend + 4.0 * context - 1.5 * risk
    buy = 1.0 / (1.0 + math.exp(-max(min(score, 8), -8)))
    buy = float(np.clip(buy, 0.02, 0.98))
    return buy, 1.0 - buy


def _quick_probabilities_from_ohlcv(raw: pd.DataFrame) -> tuple[float, float]:
    df = raw.copy().tail(120)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    ret_1 = close.pct_change().fillna(0)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    trend = float((ema_fast.iloc[-1] / max(ema_slow.iloc[-1], 1e-9)) - 1.0)
    momentum = float(ret_1.tail(12).mean())
    vol_pressure = float((volume.tail(12).mean() / max(volume.tail(60).mean(), 1e-9)) - 1.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    risk = float(tr.tail(14).mean() / max(close.iloc[-1], 1e-9))
    score = 12.0 * momentum + 5.0 * trend + 0.35 * vol_pressure - 2.0 * risk
    buy = 1.0 / (1.0 + math.exp(-max(min(score, 8), -8)))
    buy = float(np.clip(buy, 0.02, 0.98))
    return buy, 1.0 - buy


def _predict_from_model(features: pd.DataFrame) -> tuple[float, float] | None:
    keras_path = _keras_model_path()
    if keras_path.exists():
        try:
            import tensorflow as tf

            scaler = None
            if _scaler_path().exists():
                with _scaler_path().open("rb") as fh:
                    scaler = pickle.load(fh)
            values = features[FEATURE_COLUMNS].tail(64).to_numpy(dtype=np.float32)
            if scaler is not None:
                values = _apply_scaler(values, scaler)
            if len(values) < 64:
                pad = np.zeros((64 - len(values), values.shape[1]), dtype=np.float32)
                values = np.vstack([pad, values])
            model = tf.keras.models.load_model(keras_path)
            probs = model.predict(values[None, :, :], verbose=0)[0]
            return float(probs[0]), float(probs[1])
        except Exception:
            pass

    model = _load_torch_model(len(FEATURE_COLUMNS))
    if model is None:
        return None
    scaler = None
    if _scaler_path().exists():
        with _scaler_path().open("rb") as fh:
            scaler = pickle.load(fh)
    values = features[FEATURE_COLUMNS].tail(64).to_numpy(dtype=np.float32)
    if scaler is not None:
        values = _apply_scaler(values, scaler)
    if len(values) < 64:
        pad = np.zeros((64 - len(values), values.shape[1]), dtype=np.float32)
        values = np.vstack([pad, values])
    with torch.no_grad():
        logits = model(torch.tensor(values[None, :, :], dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return float(probs[0]), float(probs[1])


def _fit_scaler(values: np.ndarray) -> dict:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _apply_scaler(values: np.ndarray, scaler) -> np.ndarray:
    if hasattr(scaler, "transform"):
        return scaler.transform(values)
    return ((values - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _make_sequences(features: pd.DataFrame, sequence_length: int = 64) -> tuple[np.ndarray, np.ndarray, dict]:
    frame = features.copy()
    frame["future_return"] = frame["close"].pct_change().shift(-1)
    frame["target"] = (frame["future_return"] > 0).astype(int)
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)
    x_raw = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaler = _fit_scaler(x_raw)
    x_raw = _apply_scaler(x_raw, scaler)

    xs, ys = [], []
    for end in range(sequence_length, len(frame) - 1):
        xs.append(x_raw[end - sequence_length:end])
        ys.append(int(frame.loc[end, "target"]))
    if not xs:
        raise ValueError("Not enough rows to build BiLSTM training sequences.")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), scaler


def train_bilstm(
    symbol: str = "ethusdt",
    timeframe: str = "15m",
    epochs: int = 8,
    sequence_length: int = 64,
    lr: float = 0.001,
) -> dict:
    if torch is None or BiLSTMClassifier is None:
        raise RuntimeError("PyTorch is required for BiLSTM training.")

    features = prepare_realtime_features(symbol, timeframe)
    x, y, scaler = _make_sequences(features, sequence_length)
    split = max(1, int(len(x) * 0.8))
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    if len(x_test) == 0:
        x_test, y_test = x_train, y_train

    model = BiLSTMClassifier(n_features=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_t)
        test_loss = float(loss_fn(test_logits, y_test_t).item())
        preds = test_logits.argmax(dim=1)
        accuracy = float((preds == y_test_t).float().mean().item())

    torch.save(model.state_dict(), _weights_path())
    with _scaler_path().open("wb") as fh:
        pickle.dump(scaler, fh)

    payload = {
        "result_type": "ml_training",
        "symbol": symbol.lower(),
        "timeframe": timeframe.lower(),
        "model_version": MODEL_VERSION,
        "train_sequences": int(len(x_train)),
        "test_sequences": int(len(x_test)),
        "epochs": int(epochs),
        "sequence_length": int(sequence_length),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(accuracy, 6),
        "weights_path": str(_weights_path()),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    _append_result_row(payload)
    return payload


def _cache_path(symbol: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"prediction_{symbol.lower()}_{timeframe.lower()}.json"


def predict_probabilities(symbol: str, timeframe: str = "15m", force: bool = False) -> dict:
    cache_path = _cache_path(symbol, timeframe)
    if not force and cache_path.exists() and time.time() - cache_path.stat().st_mtime < PREDICTION_CACHE_SECONDS:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    sentiment = get_news_sentiment_cached(symbol)
    model_probs = None
    if _has_local_model():
        try:
            features = prepare_realtime_features(symbol, timeframe)
            model_probs = _predict_from_model(features)
            buy, sell = model_probs or _heuristic_probabilities(features)
        except Exception:
            raw = load_market_data(symbol, timeframe)
            buy, sell = _quick_probabilities_from_ohlcv(raw)
    else:
        raw = load_market_data(symbol, timeframe)
        buy, sell = _quick_probabilities_from_ohlcv(raw)
    payload = {
        "symbol": symbol.lower(),
        "timeframe": timeframe.lower(),
        "buy_probability": round(buy * 100, 2),
        "sell_probability": round(sell * 100, 2),
        "model_version": MODEL_VERSION,
        "source": "local_bilstm_weights" if model_probs else "local_heuristic_fallback",
        "news_sentiment": sentiment,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_prediction_csv(payload)
    return payload


def _append_prediction_csv(payload: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "strategy_results_unified.csv"
    flat = {k: v for k, v in payload.items() if k != "news_sentiment"}
    sentiment = payload.get("news_sentiment") or {}
    flat["news_sentiment_score"] = sentiment.get("score", 0.0)
    flat["news_sentiment"] = sentiment.get("sentiment", "neutral")
    _append_result_row({**flat, "result_type": "ml_inference"})


def _append_result_row(payload: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "strategy_results_unified.csv"
    row = pd.DataFrame([payload])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local BiLSTM training/inference pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--symbol", default="ethusdt")
    train.add_argument("--timeframe", default="15m")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--sequence-length", type=int, default=64)
    train.add_argument("--lr", type=float, default=0.001)

    predict = sub.add_parser("predict")
    predict.add_argument("--symbol", default="ethusdt")
    predict.add_argument("--timeframe", default="15m")
    predict.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train_bilstm(args.symbol, args.timeframe, args.epochs, args.sequence_length, args.lr), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict_probabilities(args.symbol, args.timeframe, args.force), indent=2))


if __name__ == "__main__":
    main()
