"""Next trading session close forecast for a single NEPSE stock."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "close", "range_pct", "body_pct", "return_1", "return_5",
    "sma_5_ratio", "sma_10_ratio", "ema_10_ratio", "volatility_10",
    "volume_ratio_5", "rsi_14", "macd", "macd_signal",
]


def load_prices(path):
    data = pd.read_csv(path)
    required = {"published_date", "open", "high", "low", "close", "traded_quantity"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    data["published_date"] = pd.to_datetime(data["published_date"], errors="raise")
    for column in required - {"published_date"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["published_date"].duplicated().any():
        raise ValueError("Duplicate trading dates: check the input before modeling")
    data = data.sort_values("published_date").reset_index(drop=True)
    data = data.dropna(subset=list(required - {"published_date"}))
    data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return data.reset_index(drop=True)


def make_features(data):
    """All features for row t use observations available by the close of t."""
    frame = data.copy()
    close = frame["close"]
    returns = close.pct_change()
    frame["range_pct"] = (frame["high"] - frame["low"]) / close
    frame["body_pct"] = (close - frame["open"]) / frame["open"]
    frame["return_1"] = returns
    frame["return_5"] = close.pct_change(5)
    frame["sma_5_ratio"] = close / close.rolling(5).mean() - 1
    frame["sma_10_ratio"] = close / close.rolling(10).mean() - 1
    frame["ema_10_ratio"] = close / close.ewm(span=10, adjust=False).mean() - 1
    frame["volatility_10"] = returns.rolling(10).std()
    frame["volume_ratio_5"] = frame["traded_quantity"] / frame["traded_quantity"].rolling(5).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    frame["rsi_14"] = 100 * gain / (gain + loss)
    frame.loc[(gain == 0) & (loss == 0), "rsi_14"] = 50
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame


def evaluate(path, test_fraction=0.2, alpha=10.0, output=None):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    frame = make_features(load_prices(path))
    frame["target_close"] = frame["close"].shift(-1)
    frame["target_date"] = frame["published_date"].shift(-1)
    latest = frame.dropna(subset=FEATURES).iloc[-1] if frame[FEATURES].notna().all(axis=1).any() else None
    labeled = frame.dropna(subset=FEATURES + ["target_close", "target_date"]).copy()
    cut = int(len(labeled) * (1 - test_fraction))
    if cut < 30 or len(labeled) - cut < 5:
        raise ValueError("Need at least 30 training and 5 test rows after indicator warm-up")
    train, test = labeled.iloc[:cut], labeled.iloc[cut:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train[FEATURES], train["target_close"])
    result = test[["published_date", "target_date", "close", "target_close"]].copy()
    result["baseline_close"] = test["close"]
    result["predicted_close"] = model.predict(test[FEATURES])
    actual_up = result["target_close"] > result["close"]
    predicted_up = result["predicted_close"] > result["close"]
    metrics = {
        "training_rows": len(train),
        "test_rows": len(test),
        "baseline_mae": mean_absolute_error(result["target_close"], result["baseline_close"]),
        "model_mae": mean_absolute_error(result["target_close"], result["predicted_close"]),
        "direction_accuracy": (actual_up == predicted_up).mean(),
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    print(f"Train: {metrics['training_rows']} | Test: {metrics['test_rows']}")
    print(f"Test target dates: {test['target_date'].iloc[0].date()} to {test['target_date'].iloc[-1].date()}")
    print(f"Today's-close baseline MAE: {metrics['baseline_mae']:.3f}")
    print(f"Ridge model MAE: {metrics['model_mae']:.3f}")
    print(f"Up/down accuracy: {metrics['direction_accuracy']:.1%} (ties counted as not up)")
    if latest is not None:
        model.fit(labeled[FEATURES], labeled["target_close"])
        print(f"After {latest['published_date'].date()} close: next observed session forecast = "
              f"{model.predict(latest[FEATURES].to_frame().T)[0]:.2f}")
        print("Future calendar date is unknown until the exchange schedule is confirmed.")
    return result, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Local ShareSansar-style single-ticker CSV")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output", default="data/processed/test_predictions.csv")
    args = parser.parse_args()
    evaluate(args.csv, args.test_fraction, args.alpha, args.output)
