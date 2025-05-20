import streamlit as st
import pandas as pd

from predictors.multi_level_predictor import (
    run_level_1,
    run_level_2,
    run_level_3,
    run_level_4
)
from ui.dashboard import (
    render_level_1_dashboard,
    render_level_2_dashboard,
    render_level_3_dashboard,
    render_level_4_dashboard,
    render_level_5_dashboard,
    render_reward_learning_dashboard
)
from core.logger import setup_logger
from data_loader import load_draw_history

logger = setup_logger("level_main")

st.set_page_config(page_title="Lotto Predictor — ML Levels", layout="wide")
st.title("🎯 Multi-Level Lotto Predictor (Experimental)")

# Load data
draw_data = load_draw_history()
if draw_data.empty:
    st.error("❌ Failed to load draw history. Please check the file at 'data/draw_history.xlsx'.")
    st.stop()

# Tabbed interface for Levels 1 to 5 + Reward Learning
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧠 Level 1: Individual Methods",
    "🤝 Level 2: Method Pairs",
    "➕ Level 3: Pairs + Rules",
    "🎛 Level 4: All Methods",
    "🧬 Level 5: Meta Predictor",
    "🧪 Reward Learning"
])

with tab1:
    try:
        level1_results = run_level_1(draw_data)
        render_level_1_dashboard(level1_results)
    except Exception as e:
        st.error(f"❌ Failed to run Level 1 predictions: {e}")

with tab2:
    try:
        level2_results = run_level_2(draw_data)
        render_level_2_dashboard(level2_results)
    except Exception as e:
        st.warning("⚠️ Level 2 logic not yet implemented or failed to run.")
        render_level_2_dashboard({})

with tab3:
    try:
        level3_results = run_level_3(draw_data)
        render_level_3_dashboard(level3_results)
    except Exception as e:
        st.warning("⚠️ Level 3 logic not yet implemented or failed to run.")
        render_level_3_dashboard({})

with tab4:
    try:
        level4_results = run_level_4(draw_data)
        render_level_4_dashboard(level4_results)
    except Exception as e:
        st.warning("⚠️ Level 4 logic not yet implemented or failed to run.")
        render_level_4_dashboard({})

with tab5:
    render_level_5_dashboard()

with tab6:
    render_reward_learning_dashboard()
