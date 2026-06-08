"""EDA Diagram Generator — COMP-834 Advanced Data Visualization.

Generates and saves exploratory data analysis (EDA) and XAI diagrams for:
  • Quiz 3  — Global Economy dataset  (World Bank)
  • Quiz 4  — Cryptocurrency dataset  (CoinGecko)

Output directories:
  diagrams/quiz3_eda/   diagrams/quiz3_xai/
  diagrams/quiz4_eda/   diagrams/quiz4_xai/

Usage:
    python eda_generator.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# FIX: import load_economy so Quiz-3 EDA functions can load data
from utils import load_economy

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global matplotlib style — dark theme matching the Streamlit dashboard
# ---------------------------------------------------------------------------
sns.set_style("darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "text.color": "#c9d1d9",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "axes.titlecolor": "#ffd700",
        "axes.labelcolor": "#c9d1d9",
        "figure.dpi": 150,
    }
)


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
    """Save a matplotlib figure with consistent dark-theme settings."""
    fig.tight_layout()
    fig.savefig(
        filepath,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="#0e1117",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  ✅ Saved: {filepath}")


# ===========================================================================
# QUIZ 3 — GLOBAL ECONOMY EDA
# ===========================================================================

def generate_quiz3_eda() -> None:
    """Generate EDA diagrams for the World Bank Global Economy dataset."""
    print("\n📊 Generating Quiz 3 EDA Diagrams...")
    ensure_dir("diagrams/quiz3_eda")

    try:
        df = load_economy("data/global_economy.csv")
    except FileNotFoundError:
        print("  ❌ data/global_economy.csv not found — skipping Quiz 3 EDA.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # 1. GDP Distribution Histogram
    # ------------------------------------------------------------------
    if "GDP" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df["GDP"].dropna().hist(bins=50, ax=ax, edgecolor="black", color="#00d4ff")
        ax.set_title("GDP Distribution", fontsize=16, fontweight="bold")
        ax.set_xlabel("GDP (USD)")
        ax.set_ylabel("Frequency")
        save_figure(fig, "diagrams/quiz3_eda/distribution_gdp.png")

    # ------------------------------------------------------------------
    # 2. Correlation Heatmap
    # ------------------------------------------------------------------
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            linewidths=0.4,
            cbar_kws={"label": "Correlation"},
        )
        ax.set_title("Correlation Matrix — Economic Indicators", fontsize=16, fontweight="bold")
        save_figure(fig, "diagrams/quiz3_eda/correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 3. GDP Box Plot by Region
    # ------------------------------------------------------------------
    if "Region" in df.columns and "GDP" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        regions = df["Region"].dropna().unique()
        data_by_region = [df.loc[df["Region"] == r, "GDP"].dropna().values for r in regions]
        ax.boxplot(data_by_region, labels=regions, patch_artist=True)
        ax.set_title("GDP Distribution by Region", fontsize=16, fontweight="bold")
        ax.set_xlabel("Region")
        ax.set_ylabel("GDP (USD)")
        plt.xticks(rotation=30, ha="right")
        save_figure(fig, "diagrams/quiz3_eda/boxplot_by_region.png")

    # ------------------------------------------------------------------
    # 4. Missing Values Heatmap
    # ------------------------------------------------------------------
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        missing_counts = df.isnull().sum().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 5))
        missing_counts.plot.barh(ax=ax, color="#ff6b6b", edgecolor="black")
        ax.set_title("Top-20 Columns by Missing Value Count", fontsize=16, fontweight="bold")
        ax.set_xlabel("Missing Count")
        save_figure(fig, "diagrams/quiz3_eda/missing_values_heatmap.png")
    else:
        print("  ⏭️  No missing values — skipping missing-values chart.")

    # ------------------------------------------------------------------
    # 5. Time-Series Overview (2×2 grid of key indicators)
    # ------------------------------------------------------------------
    if "Year" in df.columns and len(numeric_cols) >= 1:
        key_cols = [c for c in ["GDP", "Population", "Exports", "Imports"] if c in df.columns][:4]
        if key_cols:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            for idx, col in enumerate(key_cols):
                ax = axes[idx]
                df.groupby("Year")[col].mean().plot(ax=ax, color="#00d4ff", linewidth=2)
                ax.set_title(f"{col} Over Time", fontweight="bold")
                ax.set_xlabel("Year")
                ax.set_ylabel(col)
            # hide any unused subplots
            for idx in range(len(key_cols), 4):
                axes[idx].set_visible(False)
            save_figure(fig, "diagrams/quiz3_eda/time_series_overview.png")

    # ------------------------------------------------------------------
    # 6. PCA Cumulative Variance Explained
    # ------------------------------------------------------------------
    if len(numeric_cols) > 2:
        try:
            x_data = df[numeric_cols].fillna(df[numeric_cols].median())
            x_scaled = StandardScaler().fit_transform(x_data)
            pca = PCA()
            pca.fit(x_scaled)
            cumsum = np.cumsum(pca.explained_variance_ratio_)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(cumsum, marker="o", linestyle="-", color="#00d4ff", linewidth=2)
            ax.fill_between(range(len(cumsum)), cumsum, alpha=0.25, color="#00d4ff")
            ax.axhline(0.90, color="#ffd700", linestyle="--", linewidth=1.5, label="90% threshold")
            ax.set_title("PCA — Cumulative Variance Explained", fontsize=14, fontweight="bold")
            ax.set_xlabel("Number of Principal Components")
            ax.set_ylabel("Cumulative Variance Explained")
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_figure(fig, "diagrams/quiz3_eda/feature_importance_pca.png")
        except Exception as exc:
            print(f"  ⚠️  PCA skipped: {exc}")

    print("  ✅ Quiz 3 EDA complete.")


# ===========================================================================
# QUIZ 3 — XAI (Random Forest Feature Importance)
# ===========================================================================

def generate_quiz3_xai() -> None:
    """Generate XAI diagrams for Quiz 3 (feature importance via Random Forest)."""
    print("\n🤖 Generating Quiz 3 XAI Diagrams...")
    ensure_dir("diagrams/quiz3_xai")

    try:
        df = load_economy("data/global_economy.csv")
    except FileNotFoundError:
        print("  ❌ data/global_economy.csv not found — skipping Quiz 3 XAI.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "GDP" not in df.columns or len(numeric_cols) < 3:
        print("  ⚠️  Not enough numeric columns for XAI — skipping.")
        return

    try:
        feature_cols = [c for c in numeric_cols if c != "GDP"]
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = (df["GDP"] > df["GDP"].median()).astype(int)

        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Feature importance bar chart
        importances = rf.feature_importances_
        indices = np.argsort(importances)[-15:]  # top 15

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))  # type: ignore[attr-defined]
        ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor="black")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
        ax.set_title("Random Forest — Feature Importance (Quiz 3)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", alpha=0.3)
        save_figure(fig, "diagrams/quiz3_xai/permutation_importance.png")

        # Cumulative importance chart
        sorted_imp = np.sort(importances)[::-1]
        cumulative = np.cumsum(sorted_imp)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(cumulative, marker="o", color="#00d4ff", linewidth=2, markersize=4)
        ax.axhline(0.90, color="#ffd700", linestyle="--", linewidth=1.5, label="90% threshold")
        ax.set_title("Cumulative Feature Importance", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Cumulative Importance")
        ax.legend()
        ax.grid(alpha=0.3)
        save_figure(fig, "diagrams/quiz3_xai/cumulative_importance.png")

    except Exception as exc:
        print(f"  ⚠️  XAI generation skipped: {exc}")

    print("  ✅ Quiz 3 XAI complete.")


# ===========================================================================
# QUIZ 4 — CRYPTOCURRENCY EDA
# ===========================================================================

def _load_crypto_for_eda() -> pd.DataFrame:
    """Load crypto CSV (single file or folder of per-coin CSVs)."""
    import glob
    import os

    folder = "data/crypto_coingecko"
    single = "data/crypto_coingecko.csv"

    if os.path.isdir(folder):
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not files:
            raise FileNotFoundError(folder)
        frames = []
        for f in files:
            tmp = pd.read_csv(f, low_memory=False)
            tmp["coin"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
    elif os.path.exists(single):
        df = pd.read_csv(single, low_memory=False)
    else:
        raise FileNotFoundError("data/crypto_coingecko (folder) or data/crypto_coingecko.csv")

    # Normalise column names
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

    return df.sort_values("date").reset_index(drop=True)


def generate_quiz4_eda() -> None:
    """Generate EDA diagrams for the CoinGecko cryptocurrency dataset."""
    print("\n🪙 Generating Quiz 4 EDA Diagrams...")
    ensure_dir("diagrams/quiz4_eda")

    try:
        df = _load_crypto_for_eda()
    except FileNotFoundError as exc:
        print(f"  ❌ {exc} — skipping Quiz 4 EDA.")
        return

    # ------------------------------------------------------------------
    # 1. Price Distribution per Coin (up to 5 coins)
    # ------------------------------------------------------------------
    if "coin" in df.columns and "close" in df.columns:
        top_coins = df["coin"].value_counts().index[:5]
        fig, ax = plt.subplots(figsize=(12, 6))
        for coin in top_coins:
            data = df.loc[df["coin"] == coin, "close"].dropna()
            ax.hist(data, bins=40, alpha=0.55, label=coin, edgecolor="black")
        ax.set_title("Price Distribution by Coin", fontsize=14, fontweight="bold")
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Frequency")
        ax.legend()
        save_figure(fig, "diagrams/quiz4_eda/price_distribution.png")

    # ------------------------------------------------------------------
    # 2. Daily Trading Volume Over Time
    # ------------------------------------------------------------------
    if "volume" in df.columns and "date" in df.columns:
        daily_vol = df.groupby("date")["volume"].sum()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(daily_vol.index, daily_vol.values, alpha=0.4, color="#00d4ff")
        ax.plot(daily_vol.index, daily_vol.values, color="#00d4ff", linewidth=1)
        ax.set_title("Daily Trading Volume Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Volume")
        save_figure(fig, "diagrams/quiz4_eda/volume_trend.png")

    # ------------------------------------------------------------------
    # 3. Daily Returns Distribution
    # ------------------------------------------------------------------
    if "close" in df.columns:
        df["returns"] = df["close"].pct_change() * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        returns_clean = df["returns"].dropna()
        ax.hist(returns_clean, bins=60, color="#00d4ff", alpha=0.7, edgecolor="black")
        ax.axvline(returns_clean.mean(), color="#ff6b6b", linestyle="--",
                   linewidth=2, label=f"Mean: {returns_clean.mean():.2f}%")
        ax.axvline(0, color="#ffd700", linestyle="-", linewidth=1.5, label="Zero")
        ax.set_title("Daily Returns Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        save_figure(fig, "diagrams/quiz4_eda/return_distribution.png")

    # ------------------------------------------------------------------
    # 4. 30-Day Rolling Volatility
    # ------------------------------------------------------------------
    if "close" in df.columns and "date" in df.columns:
        df["volatility"] = df["close"].pct_change().rolling(30).std() * 100
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(df["date"], df["volatility"], alpha=0.4, color="#ff6b6b")
        ax.plot(df["date"], df["volatility"], color="#ff6b6b", linewidth=1)
        ax.set_title("30-Day Rolling Volatility (%)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (%)")
        save_figure(fig, "diagrams/quiz4_eda/volatility_analysis.png")

    # ------------------------------------------------------------------
    # 5. ACF / PACF for top coin
    # ------------------------------------------------------------------
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore

        top_coin = df["coin"].value_counts().index[0] if "coin" in df.columns else None
        series = (
            df.loc[df["coin"] == top_coin, "close"].dropna().values
            if top_coin
            else df["close"].dropna().values
        )
        if len(series) > 50:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_acf(series, lags=40, ax=ax, color="#00d4ff")
            ax.set_title(f"ACF — {top_coin or 'Close Price'}", fontweight="bold")
            save_figure(fig, "diagrams/quiz4_eda/acf_plot.png")

            fig, ax = plt.subplots(figsize=(10, 4))
            plot_pacf(series, lags=40, ax=ax, method="ywm", color="#00d4ff")
            ax.set_title(f"PACF — {top_coin or 'Close Price'}", fontweight="bold")
            save_figure(fig, "diagrams/quiz4_eda/pacf_plot.png")
    except Exception as exc:
        print(f"  ⚠️  ACF/PACF skipped: {exc}")

    # ------------------------------------------------------------------
    # 6. Price Series for top coin
    # ------------------------------------------------------------------
    if "coin" in df.columns and "close" in df.columns:
        top_coin = df["coin"].value_counts().index[0]
        coin_df = df[df["coin"] == top_coin].dropna(subset=["date", "close"])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(coin_df["date"], coin_df["close"], color="#00d4ff", linewidth=1.5)
        ax.set_title(f"Price Series — {top_coin}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        save_figure(fig, f"diagrams/quiz4_eda/{top_coin}_price_series.png")

    print("  ✅ Quiz 4 EDA complete.")


# ===========================================================================
# QUIZ 4 — XAI (Actual vs Predicted + Residuals)
# ===========================================================================

def generate_quiz4_xai() -> None:
    """Generate XAI diagrams for Quiz 4 — actual vs predicted, residuals."""
    print("\n🤖 Generating Quiz 4 XAI Diagrams...")
    ensure_dir("diagrams/quiz4_xai")

    try:
        df = _load_crypto_for_eda()
    except FileNotFoundError as exc:
        print(f"  ❌ {exc} — skipping Quiz 4 XAI.")
        return

    if "close" not in df.columns or len(df) < 20:
        print("  ⚠️  Insufficient data for XAI — skipping.")
        return

    df = df.sort_values("date").copy()
    df["ma_7"] = df["close"].rolling(7).mean()
    valid = df.dropna(subset=["ma_7", "close"]).copy()
    valid["residuals"] = valid["close"] - valid["ma_7"]

    # ------------------------------------------------------------------
    # 1. Actual vs Predicted Scatter
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid["close"], valid["ma_7"], alpha=0.4, s=15, color="#00d4ff", label="Predictions")
    min_val = min(valid["close"].min(), valid["ma_7"].min())
    max_val = max(valid["close"].max(), valid["ma_7"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")
    ax.set_title("Actual vs Predicted Price (7-day MA)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.legend()
    save_figure(fig, "diagrams/quiz4_xai/actual_vs_predicted_scatter.png")

    # ------------------------------------------------------------------
    # 2. Residuals vs Fitted Values
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(valid["ma_7"], valid["residuals"], alpha=0.4, s=15, color="#00d4ff")
    ax.axhline(0, color="#ff6b6b", linestyle="--", linewidth=2, label="Zero residual")
    ax.set_title("Residual Plot (Actual − 7-day MA)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fitted Values (7-day MA)")
    ax.set_ylabel("Residuals")
    ax.legend()
    save_figure(fig, "diagrams/quiz4_xai/residual_plot.png")

    # ------------------------------------------------------------------
    # 3. Residual Distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(valid["residuals"], bins=50, kde=True, ax=ax, color="#00d4ff")
    ax.axvline(0, color="#ff6b6b", linestyle="--", linewidth=2)
    ax.set_title("Residual Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Residual (USD)")
    save_figure(fig, "diagrams/quiz4_xai/residual_distribution.png")

    # ------------------------------------------------------------------
    # 4. Actual vs Predicted Over Time (line chart)
    # ------------------------------------------------------------------
    if "date" in valid.columns:
        plot_df = valid.tail(365)  # last year for clarity
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(plot_df["date"], plot_df["close"], color="#00d4ff",
                linewidth=1.5, label="Actual Price")
        ax.plot(plot_df["date"], plot_df["ma_7"], color="#ffd700",
                linewidth=1.5, linestyle="--", label="Predicted (7-day MA)")
        ax.fill_between(plot_df["date"],
                        plot_df["close"], plot_df["ma_7"],
                        alpha=0.2, color="#ff6b6b", label="Error")
        ax.set_title("Actual vs Predicted Price Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        save_figure(fig, "diagrams/quiz4_xai/actual_vs_predicted_line.png")

    print("  ✅ Quiz 4 XAI complete.")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 70)
    print("🎨 EDA DIAGRAM GENERATOR — COMP-834 Advanced Data Visualization")
    print("=" * 70)

    generate_quiz3_eda()
    generate_quiz3_xai()
    generate_quiz4_eda()
    generate_quiz4_xai()

    print("\n" + "=" * 70)
    print("✅ ALL DIAGRAMS GENERATED SUCCESSFULLY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()