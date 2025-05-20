# level_main.py

import streamlit as st
import pandas as pd

from predictors.multi_level_predictor import run_level_1
from predictors.multi_level_predictor import run_level_2
from predictors.reward_learning_predictor import render_reward_learning_dashboard
from ui.dashboard import render_level_1_dashboard
from core.logger import setup_logger
from data_loader import load_draw_history, load_active_tickets

logger = setup_logger("level_main")

st.set_page_config(page_title="Lotto Predictor — ML Levels", layout="wide")
st.title("🎯 Multi-Level Lotto Predictor (Experimental)")

# Load data
draw_data = load_draw_history()
if draw_data.empty:
    st.error("❌ Failed to load draw history. Please check the file at 'data/draw_history.xlsx'.")
    st.stop()

# Tabbed interface for Levels 1 to 5
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 Level 1: Individual Methods",
    "🤝 Level 2: Method Pairs",
    "➕ Level 3: Pairs + Rules",
    "🎛 Level 4: All Methods",
    "🧬 Level 5: Meta Predictor"
])

with tab1:
    st.subheader("Level 1 — Individual Predictors")
    level1_results = run_level_1(draw_data)

    for method_name, prediction_sets in level1_results.items():
        st.markdown(f"### {method_name}")
        for i, pred in enumerate(prediction_sets):
            try:
                numbers = [int(n) for n in pred[:6]]
                powerball = int(pred[6]) if len(pred) > 6 else None
                st.markdown(f"**Set {i+1}:** 🎱 {numbers} | ⭐ Powerball: {powerball}")
            except Exception as e:
                st.markdown(f"Set {i+1}: ❌ Error parsing prediction: {e}")
                st.write(pred)

    # Load and render accuracy log if available
    try:
        accuracy_df = pd.read_csv("data/overall_predictions_log.csv")
        latest_draw = draw_data["Draw Number"].max()
        latest = accuracy_df[accuracy_df["Draw Number"] == latest_draw]
        if not latest.empty:
            st.subheader("📊 Accuracy Summary (Latest Draw)")
            st.dataframe(latest[["Method", "Accuracy %"]].sort_values("Accuracy %", ascending=False))
        else:
            st.info("No accuracy data found for the latest draw.")
    except Exception as e:
        st.warning(f"⚠️ Could not load accuracy data: {e}")

    # Add reward learning dashboard
    st.markdown("---")
    with st.expander("📘 Reward Learning Predictor Dashboard", expanded=False):
        render_reward_learning_dashboard()

with tab2:
    st.subheader("Level 2 — Combined Predictors (Pairs)")
    level2_results = run_level_2(draw_data)
    for method_pair, prediction in level2_results.items():
        st.markdown(f"**{method_pair}**")
        st.write({
            "Main Numbers": prediction["main"],
            "Powerball": prediction["powerball"]
        })

with tab3:
    st.info("Level 3 (Combinations + Rules) coming soon.")

with tab4:
    st.info("Level 4 (All Predictors Merged) coming soon.")

with tab5:
    st.info("Level 5 (Meta Predictor via ML) coming soon.")
