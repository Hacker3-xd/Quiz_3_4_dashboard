"""Quiz 4 Dashboard — ML-Based Cryptocurrency Price Forecasting.

COMP-834 Advanced Data Visualization
Instructor: Dr. Muhammad Zeeshan
PAK-AUSTRIA Fachhochschule (PAF-IAST)

Features
--------
- CoinGecko historical data (folder of per-coin CSVs or single CSV)
- Candlestick chart, price + Bollinger Bands
- Technical indicators: RSI, MACD, volatility
- Actual vs Predicted line chart (Linear Regression)
- ARIMA 30 / 60 / 90-day forecast curves
- Error metrics table (MAE, RMSE, R², Accuracy, F1, AUC)
- KPI cards: prediction accuracy, forecast growth, current RSI
- Interactive coin & date-range slicers
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ===========================================================================
# DATA LOADING
# ===========================================================================

@st.cache_data(ttl=3600)
def load_crypto(path: str = "data/crypto_coingecko") -> pd.DataFrame:
    """Load CoinGecko data from a folder of CSVs or a single CSV."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        frames = []
        for f in files:
            tmp = pd.read_csv(f, low_memory=False)
            tmp["coin"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
    elif os.path.exists(path + ".csv"):
        df = pd.read_csv(path + ".csv", low_memory=False)
    elif os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise FileNotFoundError(f"Dataset not found: {path}")

    df.rename(
        columns={
            "snapped_at": "date", "Date": "date",
            "price": "close", "total_volume": "volume",
            "coin_name": "coin",
        },
        inplace=True,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)

    for col in ("close", "volume", "market_cap"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "close" in df.columns:
        if "open"  not in df.columns:
            df["open"]  = df["close"].shift(1).fillna(df["close"])
        if "high"  not in df.columns:
            df["high"]  = df[["close", "open"]].max(axis=1)
        if "low"   not in df.columns:
            df["low"]   = df[["close", "open"]].min(axis=1)

    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_lr_predictions(path: str = "models/bitcoin_lr_predictions.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_arima_forecast(steps: int = 30) -> pd.DataFrame | None:
    path = f"models/bitcoin_arima_forecast_{steps}d.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_model_metrics(path: str = "models/model_metrics.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ===========================================================================
# TECHNICAL INDICATORS
# ===========================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-8)))


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid   = prices.rolling(window, min_periods=1).mean()
    std   = prices.rolling(window, min_periods=1).std()
    return mid + num_std * std, mid, mid - num_std * std


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    for lag in (1, 7, 30):
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df["return_1d"] = df["close"].pct_change()
    df["rsi"] = calculate_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = calculate_bollinger_bands(df["close"])
    df["vol_30d"] = df["close"].pct_change().rolling(30).std() * 100
    return df.bfill().fillna(0)


# ===========================================================================
# CHART BUILDERS
# ===========================================================================

_DARK = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(14,17,23,0.9)",
)


def chart_price_bb(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper",
                             line=dict(color="rgba(255,215,0,0.35)", dash="dash"), showlegend=True))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower",
                             fill="tonexty", fillcolor="rgba(255,215,0,0.07)",
                             line=dict(color="rgba(255,215,0,0.35)", dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close",
                             line=dict(color="#00d4ff", width=2)))
    fig.update_layout(title="Price Series with Bollinger Bands", height=520, **_DARK)
    return fig


def chart_candlestick(df: pd.DataFrame) -> go.Figure:
    daily = (
        df.groupby(df["date"].dt.date)
        .agg(open=("open", "first"), high=("high", "max"),
             low=("low", "min"),   close=("close", "last"),
             volume=("volume", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    fig = go.Figure(
        data=[go.Candlestick(
            x=daily["date"],
            open=daily["open"], high=daily["high"],
            low=daily["low"],   close=daily["close"],
        )]
    )
    fig.update_layout(title="Candlestick Chart", height=560,
                      xaxis_rangeslider_visible=False, **_DARK)
    return fig


def chart_rsi_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"],
                             name="RSI(14)", line=dict(color="#ff6b6b")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"],
                             name="MACD", line=dict(color="#00d4ff")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"],
                             name="Signal", line=dict(color="#ffd700")))
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,100,100,0.5)",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(100,200,100,0.5)",
                  annotation_text="Oversold (30)")
    fig.update_layout(title="Technical Indicators — RSI & MACD", height=500, **_DARK)
    return fig


def chart_volatility(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[go.Scatter(x=df["date"], y=df["vol_30d"],
                         fill="tozeroy", line=dict(color="#ff6b6b"))]
    )
    fig.update_layout(title="30-Day Rolling Volatility (%)", height=400,
                      xaxis_title="Date", yaxis_title="Volatility (%)", **_DARK)
    return fig


def chart_returns_distribution(df: pd.DataFrame) -> go.Figure:
    returns = (df["close"].pct_change() * 100).dropna()
    fig = go.Figure(
        data=[go.Histogram(x=returns, nbinsx=60,
                           marker_color="rgba(0,212,255,0.7)")]
    )
    fig.add_vline(x=float(returns.mean()), line_dash="dash", line_color="#ffd700",
                  annotation_text=f"Mean: {returns.mean():.2f}%")
    fig.update_layout(title="Daily Returns Distribution", xaxis_title="Return (%)",
                      yaxis_title="Frequency", height=420, **_DARK)
    return fig


# ---------------------------------------------------------------------------
# QUIZ-4 SPECIFIC: Actual vs Predicted
# ---------------------------------------------------------------------------

def chart_actual_vs_predicted(lr_df: pd.DataFrame, coin_df: pd.DataFrame) -> go.Figure:
    """Line chart — actual close price overlay with LR predictions."""
    actual   = lr_df["actual"].values
    predicted = lr_df["predicted"].values
    idx = range(len(actual))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(idx), y=actual,
                             name="Actual Price",
                             line=dict(color="#00d4ff", width=2)))
    fig.add_trace(go.Scatter(x=list(idx), y=predicted,
                             name="LR Predicted",
                             line=dict(color="#ffd700", width=2, dash="dash")))
    fig.update_layout(
        title="Actual vs Predicted Price — Linear Regression (Test Set)",
        xaxis_title="Test Sample Index",
        yaxis_title="Price (USD)",
        height=500, hovermode="x unified", **_DARK,
    )
    return fig


def chart_arima_forecast(coin_df: pd.DataFrame, fc_df: pd.DataFrame, steps: int) -> go.Figure:
    """Historical price + ARIMA future forecast with optional CI ribbon."""
    fig = go.Figure()

    # Historical (last 180 days for clarity)
    hist = coin_df.tail(180)
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                             name="Historical Price",
                             line=dict(color="#00d4ff", width=2)))

    # Forecast
    fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["predicted_price"],
                             name=f"ARIMA Forecast ({steps}d)",
                             line=dict(color="#ffd700", width=2, dash="dot")))

    # Confidence interval ribbon (available for 30d)
    if "lower_ci" in fc_df.columns and "upper_ci" in fc_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
            y=pd.concat([fc_df["upper_ci"], fc_df["lower_ci"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,215,0,0.15)",
            line=dict(color="rgba(255,215,0,0)"),
            name="95% Confidence Interval",
        ))

    fig.update_layout(
        title=f"ARIMA {steps}-Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=520, hovermode="x unified", **_DARK,
    )
    return fig


def chart_error_scatter(lr_df: pd.DataFrame) -> go.Figure:
    """Scatter — actual vs predicted with perfect-fit diagonal."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lr_df["actual"], y=lr_df["predicted"],
        mode="markers",
        marker=dict(color="#00d4ff", opacity=0.5, size=5),
        name="Predictions",
    ))
    mn = min(lr_df["actual"].min(), lr_df["predicted"].min())
    mx = max(lr_df["actual"].max(), lr_df["predicted"].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                             mode="lines",
                             line=dict(color="#ff6b6b", dash="dash", width=2),
                             name="Perfect Fit"))
    fig.update_layout(
        title="Actual vs Predicted — Scatter (LR)",
        xaxis_title="Actual Price (USD)",
        yaxis_title="Predicted Price (USD)",
        height=480, **_DARK,
    )
    return fig


# ===========================================================================
# KPI CARDS (Quiz 4 — with ML accuracy)
# ===========================================================================

def show_price_kpis(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Current Price",   f"${latest['close']:,.2f}")
    c2.metric("📊 24h Return",      f"{latest.get('return_1d', 0) * 100:+.2f}%")
    c3.metric("🔬 RSI(14)",         f"{latest.get('rsi', 0):.2f}")


def show_ml_kpis(metrics_df: pd.DataFrame) -> None:
    """Display KPI cards for LR and RF model metrics."""
    lr_row = metrics_df[metrics_df["model"] == "Linear Regression"]
    rf_row = metrics_df[metrics_df["model"] == "Random Forest Classifier"]

    c1, c2, c3, c4 = st.columns(4)

    if not lr_row.empty:
        c1.metric("📉 LR MAE",  f"${lr_row.iloc[0].get('mae', 'N/A')}")
        c2.metric("📐 LR R²",   f"{lr_row.iloc[0].get('r2',  'N/A')}")
    else:
        c1.metric("📉 LR MAE", "—")
        c2.metric("📐 LR R²",  "—")

    if not rf_row.empty:
        c3.metric("🎯 RF Accuracy", f"{float(rf_row.iloc[0].get('accuracy', 0)) * 100:.1f}%")
        c4.metric("🏅 RF F1 Score", f"{float(rf_row.iloc[0].get('f1', 0)):.4f}")
    else:
        c3.metric("🎯 RF Accuracy", "—")
        c4.metric("🏅 RF F1 Score", "—")


def show_forecast_kpis(fc_df: pd.DataFrame, coin_df: pd.DataFrame) -> None:
    """Show forecast growth KPIs."""
    if fc_df is None or coin_df.empty:
        return
    last_actual  = float(coin_df["close"].iloc[-1])
    last_forecast = float(fc_df["predicted_price"].iloc[-1])
    growth_pct = ((last_forecast - last_actual) / (last_actual + 1e-8)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("📍 Last Actual Price",    f"${last_actual:,.2f}")
    c2.metric("🔮 30d Forecast End",     f"${last_forecast:,.2f}")
    c3.metric("📈 Forecast Growth (30d)", f"{growth_pct:+.2f}%")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def show_quiz4() -> None:
    """Render the complete Quiz 4 — Crypto ML & Forecasting dashboard."""
    st.markdown(
        "<h2 style='color:#ffd700;'>🤖 Quiz 4 — Crypto ML & Forecasting Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ---- Load raw data ----
    try:
        df_all = load_crypto("data/crypto_coingecko")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Place CoinGecko CSV(s) in data/crypto_coingecko/ or as data/crypto_coingecko.csv")
        return

    if "coin" not in df_all.columns:
        st.error("No `coin` column found in the dataset.")
        return

    # ---- Sidebar controls ----
    st.sidebar.markdown("### 🎯 Quiz 4 Filters")
    coins = sorted(df_all["coin"].unique())
    coin  = st.sidebar.selectbox("Select Coin", coins)

    coin_df_raw = df_all[df_all["coin"] == coin].copy()
    if coin_df_raw.empty:
        st.warning(f"No data for {coin}.")
        return

    # Date range slicer
    date_min = coin_df_raw["date"].min()
    date_max = coin_df_raw["date"].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date(),
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        coin_df_raw = coin_df_raw[
            (coin_df_raw["date"] >= start_dt) & (coin_df_raw["date"] <= end_dt)
        ]

    coin_df = feature_engineer(coin_df_raw)

    # Forecast horizon slicer (for ARIMA)
    horizon = st.sidebar.selectbox("ARIMA Forecast Horizon", [30, 60, 90], index=0)

    # ---- Load ML artefacts ----
    lr_df       = load_lr_predictions()
    fc_df       = load_arima_forecast(horizon)
    metrics_df  = load_model_metrics()
    models_ready = lr_df is not None and fc_df is not None and metrics_df is not None

    # ---- Tabs ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Technicals",
        "🤖 Actual vs Predicted",
        "🔮 ARIMA Forecast",
        "📊 Error Metrics",
        "🖼️ EDA Diagrams",
    ])

    # ------------------------------------------------------------------
    # TAB 1 — Price & Technicals
    # ------------------------------------------------------------------
    with tab1:
        st.markdown(f"### 📌 {coin.upper()} — Live KPIs")
        show_price_kpis(coin_df)
        st.divider()

        st.plotly_chart(chart_price_bb(coin_df),           use_container_width=True)
        st.plotly_chart(chart_candlestick(coin_df),        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_volatility(coin_df),         use_container_width=True)
        with c2:
            st.plotly_chart(chart_returns_distribution(coin_df), use_container_width=True)

        st.plotly_chart(chart_rsi_macd(coin_df),           use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2 — Actual vs Predicted (Linear Regression)
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("### 📉 Actual vs Predicted — Linear Regression")
        if not models_ready or lr_df is None:
            st.info(
                "⚠️ No LR predictions found. "
                "Click **Train Models** in the sidebar to generate them."
            )
        else:
            # ML KPI cards
            st.markdown("#### 🏅 Model Performance KPIs")
            if metrics_df is not None:
                show_ml_kpis(metrics_df)
            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_actual_vs_predicted(lr_df, coin_df),
                                use_container_width=True)
            with c2:
                st.plotly_chart(chart_error_scatter(lr_df),
                                use_container_width=True)

            # Residual histogram
            lr_df_copy = lr_df.copy()
            lr_df_copy["residual"] = lr_df_copy["actual"] - lr_df_copy["predicted"]
            fig_res = go.Figure(
                data=[go.Histogram(x=lr_df_copy["residual"], nbinsx=50,
                                   marker_color="rgba(0,212,255,0.7)")]
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ff6b6b")
            fig_res.update_layout(title="Residual Distribution (Actual − Predicted)",
                                  xaxis_title="Residual (USD)",
                                  yaxis_title="Frequency", height=420, **_DARK)
            st.plotly_chart(fig_res, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3 — ARIMA Forecast
    # ------------------------------------------------------------------
    with tab3:
        st.markdown(f"### 🔮 ARIMA {horizon}-Day Price Forecast")
        if fc_df is None:
            st.info(
                f"⚠️ No {horizon}-day forecast found. "
                "Click **Train Models** in the sidebar to generate forecasts."
            )
        else:
            # Forecast KPI cards
            st.markdown("#### 📊 Forecast KPIs")
            show_forecast_kpis(fc_df, coin_df)
            st.divider()

            st.plotly_chart(chart_arima_forecast(coin_df, fc_df, horizon),
                            use_container_width=True)

            # Show raw forecast table
            with st.expander("📋 Forecast Data Table"):
                st.dataframe(fc_df.round(4), use_container_width=True)

            # Download forecast CSV
            st.download_button(
                label=f"📥 Download {horizon}d Forecast CSV",
                data=fc_df.to_csv(index=False),
                file_name=f"arima_forecast_{horizon}d.csv",
                mime="text/csv",
            )

        # Compare horizons if all three are available
        st.markdown("### 📊 Multi-Horizon Forecast Comparison")
        horizon_frames = {}
        for h in (30, 60, 90):
            tmp = load_arima_forecast(h)
            if tmp is not None:
                horizon_frames[h] = tmp

        if len(horizon_frames) > 1:
            fig_multi = go.Figure()
            hist = coin_df.tail(120)
            fig_multi.add_trace(go.Scatter(x=hist["date"], y=hist["close"],
                                           name="Historical", line=dict(color="#00d4ff")))
            colors = {30: "#ffd700", 60: "#ff6b6b", 90: "#b388ff"}
            for h, frame in horizon_frames.items():
                fig_multi.add_trace(go.Scatter(
                    x=frame["date"], y=frame["predicted_price"],
                    name=f"{h}-Day Forecast",
                    line=dict(color=colors.get(h, "#ffffff"), dash="dot"),
                ))
            fig_multi.update_layout(title="30 / 60 / 90-Day ARIMA Forecast Overlay",
                                    xaxis_title="Date", yaxis_title="Price (USD)",
                                    height=500, hovermode="x unified", **_DARK)
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.info("Train models to see 30 / 60 / 90-day comparison.")

    # ------------------------------------------------------------------
    # TAB 4 — Error Metrics
    # ------------------------------------------------------------------
    with tab4:
        st.markdown("### 📊 Model Evaluation Metrics")
        if metrics_df is None:
            st.info("⚠️ No metrics CSV found. Click **Train Models** in the sidebar.")
        else:
            # ---- Regression metrics (LR) ----
            lr_rows = metrics_df[metrics_df["task"] == "Regression"]
            if not lr_rows.empty:
                st.markdown("#### 📈 Regression Metrics — Linear Regression")
                display_cols = [c for c in ["model", "mae", "rmse", "r2"] if c in lr_rows.columns]
                st.dataframe(
                    lr_rows[display_cols].rename(columns={
                        "model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²"
                    }).reset_index(drop=True),
                    use_container_width=True,
                )
                # Visual MAE / RMSE bar
                mae_val  = float(lr_rows.iloc[0].get("mae",  0) or 0)
                rmse_val = float(lr_rows.iloc[0].get("rmse", 0) or 0)
                r2_val   = float(lr_rows.iloc[0].get("r2",   0) or 0)
                fig_lr = go.Figure(data=[
                    go.Bar(name="MAE",  x=["MAE"],  y=[mae_val],  marker_color="#00d4ff"),
                    go.Bar(name="RMSE", x=["RMSE"], y=[rmse_val], marker_color="#ffd700"),
                ])
                fig_lr.update_layout(title="LR Error Metrics", barmode="group",
                                     height=350, **_DARK)
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE",  f"${mae_val:,.4f}")
                c2.metric("RMSE", f"${rmse_val:,.4f}")
                c3.metric("R²",   f"{r2_val:.4f}")
                st.plotly_chart(fig_lr, use_container_width=True)

            st.divider()

            # ---- Classification metrics (RF) ----
            rf_rows = metrics_df[metrics_df["task"] == "Classification"]
            if not rf_rows.empty:
                st.markdown("#### 🌲 Classification Metrics — Random Forest")
                display_cols = [
                    c for c in ["model", "accuracy", "precision", "recall", "f1", "auc"]
                    if c in rf_rows.columns
                ]
                st.dataframe(
                    rf_rows[display_cols].rename(columns={
                        "model": "Model", "accuracy": "Accuracy",
                        "precision": "Precision", "recall": "Recall",
                        "f1": "F1 Score", "auc": "AUC-ROC"
                    }).reset_index(drop=True),
                    use_container_width=True,
                )
                # Radar-style bar for classification metrics
                clf_metrics = {
                    "Accuracy":  float(rf_rows.iloc[0].get("accuracy",  0) or 0),
                    "Precision": float(rf_rows.iloc[0].get("precision", 0) or 0),
                    "Recall":    float(rf_rows.iloc[0].get("recall",    0) or 0),
                    "F1 Score":  float(rf_rows.iloc[0].get("f1",        0) or 0),
                    "AUC-ROC":   float(rf_rows.iloc[0].get("auc",       0) or 0),
                }
                fig_rf = go.Figure(data=[
                    go.Bar(
                        x=list(clf_metrics.keys()),
                        y=list(clf_metrics.values()),
                        marker_color=["#00d4ff", "#ffd700", "#ff6b6b", "#b388ff", "#69f0ae"],
                    )
                ])
                fig_rf.update_layout(
                    title="Random Forest Classification Metrics",
                    yaxis=dict(range=[0, 1]),
                    height=420, **_DARK,
                )
                st.plotly_chart(fig_rf, use_container_width=True)

                c1, c2, c3, c4, c5 = st.columns(5)
                for col, (name, val) in zip(
                    [c1, c2, c3, c4, c5], clf_metrics.items()
                ):
                    col.metric(name, f"{val:.4f}")

        # ---- Raw metrics download ----
        if metrics_df is not None:
            st.divider()
            st.download_button(
                label="📥 Download Metrics CSV",
                data=metrics_df.to_csv(index=False),
                file_name="model_metrics.csv",
                mime="text/csv",
            )

    # ------------------------------------------------------------------
    # TAB 5 — EDA Diagrams (from disk)
    # ------------------------------------------------------------------
    with tab5:
        st.markdown("### 🖼️ Quiz 4 EDA & XAI Diagrams")
        eda_imgs = sorted(Path("diagrams/quiz4_eda").glob("*.png"))
        xai_imgs = sorted(Path("diagrams/quiz4_xai").glob("*.png"))

        if not eda_imgs and not xai_imgs:
            st.info("No diagrams found. Click **Generate EDA** in the sidebar to create them.")
        else:
            if eda_imgs:
                st.markdown("#### 📊 EDA Diagrams")
                cols = st.columns(2)
                for idx, img in enumerate(eda_imgs):
                    cols[idx % 2].image(
                        str(img),
                        caption=img.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )
            if xai_imgs:
                st.markdown("#### 🤖 XAI Diagrams")
                cols = st.columns(2)
                for idx, img in enumerate(xai_imgs):
                    cols[idx % 2].image(
                        str(img),
                        caption=img.stem.replace("_", " ").title(),
                        use_container_width=True,
                    )

    # ---- Footer ----
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#8b949e;font-size:12px;'>"
        "COMP-834 Advanced Data Visualization | PAK-AUSTRIA Fachhochschule | "
        "Dr. Muhammad Zeeshan"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    st.set_page_config(page_title="Quiz 4 — Crypto ML", layout="wide", page_icon="🤖")
    show_quiz4()