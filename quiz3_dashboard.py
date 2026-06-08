"""Quiz 3 Dashboard — Interactive Global Economy Data Analysis.

COMP-834 Advanced Data Visualization
Instructor: Dr. Muhammad Zeeshan
PAK-AUSTRIA Fachhochschule (PAF-IAST)

Features
--------
- World Bank Global Economy data via load_economy()
- Full data transformation pipeline with summary metrics
- Dimensional data model (fact + dim tables)
- KPI cards, bar, pie, line, choropleth, scatter, heatmap, box, area charts
- Sidebar slicers (region, year range, top-N)
- Drill-down tab with country-level trend selection
- Download tab with CSV export
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import load_economy


# ===========================================================================
# DATA LOADING
# ===========================================================================

@st.cache_data(ttl=3600)
def load_global_economy(path: str = "data/global_economy.csv") -> pd.DataFrame:
    """Load and cache the World Bank global economy dataset."""
    return load_economy(path)


# ===========================================================================
# DATA TRANSFORMATION
# ===========================================================================

def transform_economy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, deduplicate, and engineer features on the raw DataFrame."""
    before_shape = df.shape
    before_nulls = int(df.isnull().sum().sum())

    # Deduplicate columns first (safety net)
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    df = df.drop_duplicates()

    # Ensure Year is integer
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Coerce everything except known text columns to numeric
    text_cols = {"Country", "Region", "Currency", "CountryID"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Derived / engineered columns ---
    try:
        if "GDP" in df.columns and "Population" in df.columns:
            df["GDP_per_Capita"] = (df["GDP"] / df["Population"]).round(2)
    except Exception:
        pass

    try:
        if "GDP" in df.columns and "Country" in df.columns:
            df = df.sort_values(["Country", "Year"])
            df["GDP_Growth_Rate"] = (
                df.groupby("Country", observed=True)["GDP"]
                .pct_change()
                * 100
            ).round(2)
    except Exception:
        pass

    try:
        if "Exports" in df.columns and "GDP" in df.columns:
            df["Trade_Pct_GDP"] = (df["Exports"] / df["GDP"] * 100).round(2)
    except Exception:
        pass

    # Fill remaining numeric nulls with column median
    for col in df.columns:
        if col not in text_cols and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    after_nulls = int(df.isnull().sum().sum())

    # Final column dedup (in case derived cols clashed)
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    after_shape = df.shape

    summary: Dict[str, Any] = {
        "before_shape": before_shape,
        "after_shape": after_shape,
        "before_nulls": before_nulls,
        "after_nulls": after_nulls,
        "duplicates_removed": before_shape[0] - after_shape[0],
        "features_created": 3,
    }
    return df, summary


# ===========================================================================
# DATA MODEL
# ===========================================================================

def create_data_model(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build fact and dimension tables from the transformed DataFrame."""
    # Fact table — everything except the dimension keys
    fact_cols = [c for c in df.columns if c not in ("Country", "Region", "Year")]
    economic_facts = df[["Country", "Year"] + fact_cols].copy()

    dim_country = (
        df[["Country", "Region"]].drop_duplicates()
        if "Country" in df.columns and "Region" in df.columns
        else pd.DataFrame()
    )
    dim_year = (
        df[["Year"]].drop_duplicates().sort_values("Year")
        if "Year" in df.columns
        else pd.DataFrame()
    )

    return {
        "economic_facts": economic_facts,
        "dim_country": dim_country,
        "dim_year": dim_year,
    }


# ===========================================================================
# VISUALIZATION FUNCTIONS
# ===========================================================================

def create_kpi_cards(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n = df["Country"].nunique() if "Country" in df.columns else 0
        st.metric("🌍 Total Countries", n)

    with col2:
        if "GDP" in df.columns and "Year" in df.columns:
            latest = int(df["Year"].max())
            total_gdp = df[df["Year"] == latest]["GDP"].sum()
            st.metric("💰 Latest Year Total GDP", f"${total_gdp / 1e12:.2f}T")
        else:
            st.metric("💰 Latest Year Total GDP", "N/A")

    with col3:
        if "GDP" in df.columns and "Country" in df.columns:
            idx = df["GDP"].idxmax()
            country = df.loc[idx, "Country"]
            val = df.loc[idx, "GDP"]
            st.metric("🏆 Top GDP Country", country, f"${val / 1e12:.2f}T")
        else:
            st.metric("🏆 Top GDP Country", "N/A")

    with col4:
        if "Inflation" in df.columns:
            avg = df["Inflation"].mean()
            st.metric("📈 Avg Inflation Rate", f"{avg:.2f}%")
        else:
            st.metric("📈 Avg Inflation Rate", "N/A")


def create_top_countries_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure | None:
    if "GDP" not in df.columns or "Country" not in df.columns:
        return None
    latest = int(df["Year"].max()) if "Year" in df.columns else None
    data = df[df["Year"] == latest] if latest else df
    top = data.nlargest(top_n, "GDP")

    fig = go.Figure(
        data=[
            go.Bar(
                x=top["GDP"],
                y=top["Country"],
                orientation="h",
                marker=dict(color=top["GDP"], colorscale="Viridis", showscale=True),
                text=[f"${v / 1e12:.2f}T" for v in top["GDP"]],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Top {top_n} Countries by GDP ({latest})",
        xaxis_title="GDP (USD)",
        yaxis_title="Country",
        template="plotly_dark",
        height=500,
        hovermode="y unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig


def create_region_pie_chart(df: pd.DataFrame) -> go.Figure | None:
    if "GDP" not in df.columns or "Region" not in df.columns:
        return None
    region_gdp = df.groupby("Region")["GDP"].sum().sort_values(ascending=False)
    fig = px.pie(
        values=region_gdp.values,
        names=region_gdp.index,
        title="GDP Distribution by Region",
        template="plotly_dark",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=500, paper_bgcolor="rgba(14,17,23,0.8)")
    return fig


def create_gdp_trend_chart(df: pd.DataFrame, countries: List[str]) -> go.Figure | None:
    if "GDP" not in df.columns or "Year" not in df.columns:
        return None
    plot_df = df[df["Country"].isin(countries)]
    fig = px.line(
        plot_df, x="Year", y="GDP", color="Country",
        title="GDP Trend Over Time",
        template="plotly_dark", markers=True,
    )
    fig.update_layout(
        height=500, hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig


def create_choropleth_map(df: pd.DataFrame) -> go.Figure | None:
    if "GDP" not in df.columns or "Country" not in df.columns:
        return None
    latest = int(df["Year"].max()) if "Year" in df.columns else None
    data = df[df["Year"] == latest].copy() if latest else df.copy()
    fig = px.choropleth(
        data,
        locations="Country",
        locationmode="country names",
        color="GDP",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title=f"GDP by Country — Choropleth Map ({latest})",
        template="plotly_dark",
    )
    fig.update_layout(
        height=520,
        paper_bgcolor="rgba(14,17,23,0.8)",
        geo=dict(bgcolor="rgba(20,30,40,0.5)"),
    )
    return fig


def create_scatter_chart(df: pd.DataFrame) -> go.Figure | None:
    if not all(c in df.columns for c in ["GDP", "Inflation", "Region"]):
        return None
    fig = px.scatter(
        df.dropna(subset=["GDP", "Inflation"]),
        x="Inflation", y="GDP", color="Region",
        size="Population" if "Population" in df.columns else None,
        title="GDP vs Inflation by Region",
        template="plotly_dark",
        hover_data=["Country", "Year"] if "Year" in df.columns else ["Country"],
    )
    fig.update_layout(
        height=500, hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdYlGn",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title="Correlation Matrix — Economic Indicators",
        height=600, template="plotly_dark",
        paper_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig


def create_boxplot_chart(df: pd.DataFrame) -> go.Figure | None:
    if "GDP" not in df.columns:
        return None
    group = "Income_Group" if "Income_Group" in df.columns else "Region"
    if group not in df.columns:
        fig = px.box(df, y="GDP", title="GDP Distribution", template="plotly_dark")
    else:
        fig = px.box(df, x=group, y="GDP",
                     title=f"GDP Distribution by {group}", template="plotly_dark")
    fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(14,17,23,0.8)")
    return fig


def create_area_chart(df: pd.DataFrame) -> go.Figure | None:
    if not all(c in df.columns for c in ["Year", "Region", "GDP"]):
        return None
    regional = df.groupby(["Year", "Region"])["GDP"].sum().reset_index()
    fig = px.area(regional, x="Year", y="GDP", color="Region",
                  title="Regional GDP Over Time", template="plotly_dark")
    fig.update_layout(height=500, hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.8)")
    return fig


def create_gdp_growth_chart(df: pd.DataFrame, countries: List[str]) -> go.Figure | None:
    if "GDP_Growth_Rate" not in df.columns or "Year" not in df.columns:
        return None
    plot_df = df[df["Country"].isin(countries)].dropna(subset=["GDP_Growth_Rate"])
    fig = px.bar(plot_df, x="Year", y="GDP_Growth_Rate", color="Country",
                 barmode="group", title="GDP Growth Rate (%) Over Time",
                 template="plotly_dark")
    fig.update_layout(height=500, hovermode="x unified",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(14,17,23,0.8)")
    return fig


# ===========================================================================
# MAIN DASHBOARD ENTRY POINT
# ===========================================================================

def show_quiz3() -> None:
    """Render the complete Quiz 3 — Global Economy dashboard."""
    st.markdown(
        "<h2 style='color:#00d4ff;'>📊 Quiz 3 — Global Economy Dashboard</h2>",
        unsafe_allow_html=True,
    )

    # ---- Load data ----
    try:
        with st.spinner("Loading global economy dataset..."):
            df_raw = load_global_economy()
    except FileNotFoundError:
        st.error("❌ Global economy dataset not found at data/global_economy.csv")
        st.info("Please place the World Bank CSV in the data/ folder and restart.")
        return

    # ---- Transformation summary ----
    with st.expander("📋 Data Transformation Summary", expanded=False):
        st.markdown(f"**Original shape:** {df_raw.shape}")
        with st.spinner("Transforming data..."):
            df_transformed, summary = transform_economy(df_raw)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Before Shape", f"{summary['before_shape'][0]} × {summary['before_shape'][1]}")
        c2.metric("After Shape",  f"{summary['after_shape'][0]} × {summary['after_shape'][1]}")
        c3.metric("Nulls Fixed",   summary["before_nulls"] - summary["after_nulls"])
        c4.metric("Features Created", summary["features_created"])
        st.markdown("**Sample (transformed):**")
        st.dataframe(df_transformed.head(10), use_container_width=True)

    # ---- Data model ----
    data_model = create_data_model(df_transformed)

    # ---- Sidebar slicers ----
    st.sidebar.markdown("### 🎯 Dashboard Filters")
    region_options = ["All"] + sorted(df_transformed["Region"].dropna().unique()) \
        if "Region" in df_transformed.columns else ["All"]
    selected_regions = st.sidebar.multiselect("Region", region_options, default=["All"])
    filtered_df = (
        df_transformed
        if "All" in selected_regions
        else df_transformed[df_transformed["Region"].isin(selected_regions)]
    )

    if "Year" in df_transformed.columns:
        yr_min = int(df_transformed["Year"].min())
        yr_max = int(df_transformed["Year"].max())
        year_range = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
        filtered_df = filtered_df[
            (filtered_df["Year"] >= year_range[0]) &
            (filtered_df["Year"] <= year_range[1])
        ]

    top_n = st.sidebar.slider("Top N Countries", 5, 50, 15)

    # ---- Dashboard Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🗺️ Map & Trends", "🔍 Detailed Analysis", "📥 Download"]
    )

    # ------------------------------------------------------------------
    # TAB 1 — Overview
    # ------------------------------------------------------------------
    with tab1:
        st.markdown("### 📌 Key Performance Indicators")
        create_kpi_cards(filtered_df)
        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            fig = create_top_countries_chart(filtered_df, top_n)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = create_region_pie_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Area chart (full width)
        fig = create_area_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2 — Choropleth Map & Trends
    # ------------------------------------------------------------------
    with tab2:
        # FIX: choropleth now properly rendered in its own dedicated tab
        st.markdown("### 🗺️ World Map — GDP by Country")
        fig = create_choropleth_map(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choropleth requires Country and GDP columns.")

        st.divider()
        st.markdown("### 📈 GDP Trend & Growth Rate — Country Drill-Down")
        if "Country" in df_transformed.columns:
            default_countries = list(df_transformed["Country"].unique()[:5])
            countries = st.multiselect(
                "Select Countries",
                options=sorted(df_transformed["Country"].unique()),
                default=default_countries,
            )
            if countries:
                c1, c2 = st.columns(2)
                with c1:
                    fig = create_gdp_trend_chart(df_transformed, countries)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = create_gdp_growth_chart(df_transformed, countries)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3 — Detailed Analysis
    # ------------------------------------------------------------------
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = create_scatter_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = create_boxplot_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        fig = create_correlation_heatmap(df_transformed)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # EDA diagrams from disk
        from pathlib import Path
        eda_imgs = sorted(Path("diagrams/quiz3_eda").glob("*.png"))
        xai_imgs = sorted(Path("diagrams/quiz3_xai").glob("*.png"))
        if eda_imgs or xai_imgs:
            st.markdown("### 🖼️ EDA & XAI Diagrams")
            cols = st.columns(2)
            for idx, img in enumerate(list(eda_imgs) + list(xai_imgs)):
                cols[idx % 2].image(str(img), caption=img.stem.replace("_", " ").title(),
                                    use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 4 — Download
    # ------------------------------------------------------------------
    with tab4:
        st.markdown("### 📥 Download Datasets")

        st.download_button(
            label="📊 Transformed Dataset (CSV)",
            data=df_transformed.to_csv(index=False),
            file_name="global_economy_transformed.csv",
            mime="text/csv",
        )

        st.markdown("### 🗂️ Data Model Preview")
        st.markdown("**Fact Table: economic_facts**")
        st.dataframe(data_model["economic_facts"].head(10), use_container_width=True)
        st.markdown("**Dimension Table: dim_country**")
        st.dataframe(data_model["dim_country"].head(10), use_container_width=True)
        st.markdown("**Dimension Table: dim_year**")
        st.dataframe(data_model["dim_year"].head(10), use_container_width=True)

    # ---- Footer ----
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#8b949e;font-size:12px;'>"
        "COMP-834 Advanced Data Visualization | PAK-AUSTRIA Fachhochschule | "
        "Dr. Muhammad Zeeshan"
        "</div>",
        unsafe_allow_html=True,
    )