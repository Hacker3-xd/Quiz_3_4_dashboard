"""ML Forecasting Module — COMP-834 Advanced Data Visualization.

Trains three models on CoinGecko cryptocurrency data and saves all
artefacts (models + forecast CSVs + metrics) to disk for the Quiz 4
Streamlit dashboard to consume.

Models
------
  A. Linear Regression      — price-level trend prediction
  B. ARIMA(p,d,q)           — time-series forecasting (30 / 60 / 90 days)
  C. Random Forest Classifier — directional (up/down) classification

Output files
------------
  models/linear_regression.pkl
  models/arima_model.pkl
  models/random_forest_classifier.pkl
  models/model_metrics.csv
  models/bitcoin_arima_forecast_30d.csv
  models/bitcoin_arima_forecast_60d.csv
  models/bitcoin_arima_forecast_90d.csv
  models/bitcoin_lr_predictions.csv

Usage:
    python ml_forecasting.py
"""

import glob
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_crypto_data(path_or_dir: str = "data/crypto_coingecko") -> pd.DataFrame:
    """Load CoinGecko crypto data from a folder of per-coin CSVs or a single CSV.

    Normalises column names so the rest of the pipeline always works with:
      date, close, volume, market_cap, coin
    """
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {path_or_dir}"
            )
        frames = []
        for f in files:
            coin_name = os.path.splitext(os.path.basename(f))[0]
            tmp = pd.read_csv(f, low_memory=False)
            tmp["coin"] = coin_name
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
    else:
        csv_path = path_or_dir if path_or_dir.endswith(".csv") else path_or_dir + ".csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found: {path_or_dir} (folder) or {csv_path}"
            )
        df = pd.read_csv(csv_path, low_memory=False)

    # Normalise column names (CoinGecko history endpoint variants)
    df.rename(
        columns={
            "snapped_at": "date",
            "Date": "date",
            "price": "close",
            "total_volume": "volume",
            "coin_name": "coin",
        },
        inplace=True,
    )

    # Parse dates and strip timezone so statsmodels / sklearn are happy
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)

    for col in ("close", "volume", "market_cap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, technical, and temporal features."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lag features
    for lag in (1, 7, 30):
        df[f"lag_{lag}"] = df["close"].shift(lag)

    # Rolling statistics
    df["rolling_mean_7"]  = df["close"].rolling(7,  min_periods=1).mean()
    df["rolling_mean_30"] = df["close"].rolling(30, min_periods=1).mean()
    df["rolling_std_7"]   = df["close"].rolling(7,  min_periods=1).std()

    # Technical indicators
    df["price_change_pct"]   = df["close"].pct_change() * 100
    df["volume_price_ratio"] = df.get("volume", pd.Series(0, index=df.index)) / (df["close"] + 1e-6)
    df["rsi"]                = calculate_rsi(df["close"])

    # Temporal features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter

    return df.bfill().fillna(0)


# ===========================================================================
# HELPERS
# ===========================================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _chronological_split(df: pd.DataFrame, test_ratio: float = 0.20):
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    "day_of_week", "month", "quarter",
    "rsi", "volume_price_ratio",
]


# ===========================================================================
# MODEL A — LINEAR REGRESSION
# ===========================================================================

def train_linear_regression(df: pd.DataFrame) -> Tuple[LinearRegression, Dict]:
    print("\n  📈 Training Linear Regression...")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].values
    y = df["close"].values

    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    print(f"     MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    return model, {
        "model": model,
        "mae": mae, "rmse": rmse, "r2": r2,
        "y_test": y_test, "y_pred": y_pred,
    }


# ===========================================================================
# MODEL B — ARIMA
# ===========================================================================

def train_arima(df: pd.DataFrame) -> Tuple[object, Dict]:
    print("\n  📊 Training ARIMA...")

    series = df.sort_values("date")["close"].dropna().astype(float)

    # Determine differencing order via ADF test
    try:
        p_value = adfuller(series)[1]
        d = 1 if p_value > 0.05 else 0
    except Exception:
        d = 1

    # Fit ARIMA
    for order in [(1, d, 1), (1, d, 0), (0, d, 1), (1, 0, 1)]:
        try:
            fitted = ARIMA(series, order=order).fit()
            print(f"     ARIMA{order} converged.")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("ARIMA fitting failed for all attempted orders.")

    forecasts = {}
    for steps in (30, 60, 90):
        fc = fitted.get_forecast(steps=steps)
        forecasts[steps] = {
            "mean": fc.predicted_mean,
            "conf_int": fc.conf_int(),
        }
        print(f"     {steps}-day avg forecast: ${fc.predicted_mean.mean():.2f}")

    return fitted, {
        "model": fitted,
        "forecast_30": forecasts[30]["mean"],
        "forecast_60": forecasts[60]["mean"],
        "forecast_90": forecasts[90]["mean"],
        "conf_int_30": forecasts[30]["conf_int"],
    }


# ===========================================================================
# MODEL C — RANDOM FOREST CLASSIFIER
# ===========================================================================

def train_random_forest_classifier(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict]:
    print("\n  🌲 Training Random Forest Classifier...")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].iloc[:-1]                        # drop last row (no next-day label)
    y = (df["close"].shift(-1) > df["close"]).astype(int).iloc[:-1]

    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    print(f"     Acc={accuracy:.4f}  Prec={precision:.4f}  Rec={recall:.4f}  "
          f"F1={f1:.4f}  AUC={auc:.4f}")

    importance_df = pd.DataFrame(
        {"feature": feat_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\n     Top 5 Features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"       {row['feature']}: {row['importance']:.4f}")

    return model, {
        "model": model,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "auc": auc,
        "feature_importance": importance_df,
        "y_test": y_test.values, "y_pred": y_pred,
    }


# ===========================================================================
# SAVE ARTEFACTS
# ===========================================================================

def save_all_artefacts(
    lr_results: Dict,
    arima_results: Dict,
    rf_results: Dict,
    df: pd.DataFrame,
) -> None:
    """Persist models, forecasts, predictions, and metrics to disk."""
    print("\n  💾 Saving artefacts...")
    ensure_dir("models")

    # ---- Models ----
    joblib.dump(lr_results["model"],    "models/linear_regression.pkl")
    joblib.dump(arima_results["model"], "models/arima_model.pkl")
    joblib.dump(rf_results["model"],    "models/random_forest_classifier.pkl")
    print("     Saved: models/*.pkl")

    # ---- ARIMA forecasts (with date index) ----
    last_date = df["date"].max()
    for steps, key in [(30, "forecast_30"), (60, "forecast_60"), (90, "forecast_90")]:
        fc_series = arima_results[key]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
        )
        fc_df = pd.DataFrame({"date": future_dates, "predicted_price": fc_series.values})

        # Attach confidence interval for 30-day forecast
        if steps == 30 and "conf_int_30" in arima_results:
            ci = arima_results["conf_int_30"]
            fc_df["lower_ci"] = ci.iloc[:, 0].values
            fc_df["upper_ci"] = ci.iloc[:, 1].values

        out = f"models/bitcoin_arima_forecast_{steps}d.csv"
        fc_df.to_csv(out, index=False)
        print(f"     Saved: {out}")

    # ---- LR actual vs predicted ----
    lr_preds_df = pd.DataFrame({
        "actual":    lr_results["y_test"],
        "predicted": lr_results["y_pred"],
    })
    lr_preds_df.to_csv("models/bitcoin_lr_predictions.csv", index=False)
    print("     Saved: models/bitcoin_lr_predictions.csv")

    # ---- Unified metrics CSV ----
    metrics_df = pd.DataFrame([
        {
            "model":     "Linear Regression",
            "task":      "Regression",
            "mae":       round(lr_results["mae"],  4),
            "rmse":      round(lr_results["rmse"], 4),
            "r2":        round(lr_results["r2"],   4),
            "accuracy":  None,
            "precision": None,
            "recall":    None,
            "f1":        None,
            "auc":       None,
        },
        {
            "model":     "Random Forest Classifier",
            "task":      "Classification",
            "mae":       None,
            "rmse":      None,
            "r2":        None,
            "accuracy":  round(rf_results["accuracy"],  4),
            "precision": round(rf_results["precision"], 4),
            "recall":    round(rf_results["recall"],    4),
            "f1":        round(rf_results["f1"],        4),
            "auc":       round(rf_results["auc"],       4),
        },
    ])
    metrics_df.to_csv("models/model_metrics.csv", index=False)
    print("     Saved: models/model_metrics.csv")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("🤖 ML FORECASTING — COMP-834 Advanced Data Visualization")
    print("=" * 70)

    # Load
    print("\n📂 Loading crypto data...")
    try:
        df_raw = load_crypto_data("data/crypto_coingecko")
    except FileNotFoundError as exc:
        print(f"  ❌ {exc}")
        return

    # Use only the most data-rich coin if multiple are present
    if "coin" in df_raw.columns:
        top_coin = df_raw["coin"].value_counts().index[0]
        print(f"  ℹ️  Using coin: {top_coin}  ({len(df_raw[df_raw['coin'] == top_coin])} rows)")
        df_raw = df_raw[df_raw["coin"] == top_coin].copy()

    # Feature engineering
    print("\n⚙️  Feature engineering...")
    df = feature_engineer(df_raw)
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    print(f"  Dataset shape after feature engineering: {df.shape}")

    if len(df) < 100:
        print("  ❌ Too few rows for meaningful ML (need > 100).")
        return

    # Train
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    lr_results    = None
    arima_results = None
    rf_results    = None

    try:
        _, lr_results = train_linear_regression(df)
    except Exception as exc:
        print(f"  ❌ Linear Regression failed: {exc}")

    try:
        _, arima_results = train_arima(df)
    except Exception as exc:
        print(f"  ❌ ARIMA failed: {exc}")

    try:
        _, rf_results = train_random_forest_classifier(df)
    except Exception as exc:
        print(f"  ❌ Random Forest failed: {exc}")

    # Save
    if lr_results and arima_results and rf_results:
        print("\n" + "=" * 70)
        save_all_artefacts(lr_results, arima_results, rf_results, df)
        print("=" * 70)
        print("\n✅ ML FORECASTING COMPLETED SUCCESSFULLY\n")
    else:
        print("\n⚠️  One or more models failed — partial artefacts may have been saved.")


if __name__ == "__main__":
    main()