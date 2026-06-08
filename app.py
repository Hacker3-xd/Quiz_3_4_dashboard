"""Main Streamlit app to host Quiz 3 and Quiz 4 dashboards.

COMP-834 Advanced Data Visualization
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import streamlit as st


def set_page() -> None:
    st.set_page_config(page_title="COMP-834 Dashboards", page_icon="📊", layout="wide")


def initialize_directories() -> None:
    dirs = ["data", "diagrams/quiz3_eda", "diagrams/quiz3_xai",
            "diagrams/quiz4_eda", "diagrams/quiz4_xai", "models"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_script(cmd: List[str], description: str = "") -> None:
    try:
        with st.spinner(f"⏳ {description or ' '.join(cmd)}..."):
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=900)
            if result.returncode == 0:
                st.success(f"✅ {description or ' '.join(cmd)}")
            else:
                # show stderr in expander if errors present
                stderr = result.stderr or ""
                if stderr.strip():
                    with st.expander("Show script errors"):
                        st.code(stderr)
                st.warning(f"⚠️ Completed with warnings or errors. See expander for details.")
    except subprocess.TimeoutExpired:
        st.error("❌ Process timed out")
    except FileNotFoundError:
        st.error(f"❌ Script not found: {cmd[0]}")
    except Exception as e:
        st.error(f"❌ Error running script: {e}")


def sidebar_status() -> None:
    st.sidebar.markdown("### 📁 Dataset Status")

    economy_exists = os.path.exists("data/global_economy.csv")
    st.sidebar.markdown(f"{'✅' if economy_exists else '❌'} Global Economy CSV")

    crypto_exists = (os.path.isdir("data/crypto_coingecko") or
                     os.path.exists("data/crypto_coingecko.csv"))
    st.sidebar.markdown(f"{'✅' if crypto_exists else '❌'} Crypto Dataset")

    models_exist = (os.path.exists("models/linear_regression.pkl") and
                    os.path.exists("models/arima_model.pkl"))
    st.sidebar.markdown(f"{'✅' if models_exist else '❌'} ML Models Trained")

    diagrams_exist = os.path.exists("diagrams/quiz3_eda")
    st.sidebar.markdown(f"{'✅' if diagrams_exist else '❌'} EDA Diagrams Generated")


def sidebar_gallery() -> None:
    st.sidebar.markdown("### 📁 Diagram Gallery")
    with st.sidebar.expander("Quiz 3 Diagrams", expanded=False):
        for p in sorted(Path("diagrams/quiz3_eda").glob("*.png")):
            try:
                st.image(str(p), width="always", caption=p.name)
            except Exception:
                st.write(p.name)
    with st.sidebar.expander("Quiz 3 XAI", expanded=False):
        for p in sorted(Path("diagrams/quiz3_xai").glob("*.png")):
            try:
                st.image(str(p), width="always", caption=p.name)
            except Exception:
                st.write(p.name)
    with st.sidebar.expander("Quiz 4 Diagrams", expanded=False):
        for p in sorted(Path("diagrams/quiz4_eda").glob("*.png")):
            try:
                st.image(str(p), width="always", caption=p.name)
            except Exception:
                st.write(p.name)
    with st.sidebar.expander("Quiz 4 XAI", expanded=False):
        for p in sorted(Path("diagrams/quiz4_xai").glob("*.png")):
            try:
                st.image(str(p), width="always", caption=p.name)
            except Exception:
                st.write(p.name)


def render_sidebar() -> None:
    sidebar_status()
    st.sidebar.divider()
    sidebar_gallery()
    st.sidebar.divider()
    st.sidebar.markdown("### Utilities")
    if st.sidebar.button("Generate EDA"):
        run_script([sys.executable, "eda_generator.py"], "Generating EDA diagrams...")
    if st.sidebar.button("Train Models"):
        run_script([sys.executable, "ml_forecasting.py"], "Training ML models...")


def main() -> None:
    set_page()
    initialize_directories()

    st.markdown("<h1 style='text-align:center;color:#00d4ff'>📊 COMP-834 Advanced Data Visualization</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Navigation buttons using session_state
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "quiz3"

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 Quiz 3 — Global Economy"):
            st.session_state.active_tab = "quiz3"
    with col2:
        if st.button("🤖 Quiz 4 — Crypto ML"):
            st.session_state.active_tab = "quiz4"

    st.markdown("---")

    render_sidebar()

    try:
        if st.session_state.active_tab == "quiz3":
            from quiz3_dashboard import show_quiz3  # noqa: E402
            show_quiz3()
        else:
            from quiz4_dashboard import show_quiz4  # noqa: E402
            show_quiz4()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()
