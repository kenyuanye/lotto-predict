# main.py

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import setup_logger
from utils.predictor import get_excluded_numbers
from utils.analysis import plot_hot_cold_numbers
from utils.custom_rules import calculate_powerball_gaps
from utils.hybrid_predictor import filter_and_rank_sets
from utils.accuracy_tracker import track_accuracy, plot_accuracy_breakdown
from utils.symbolic_vs_rf_comparison import compare_model_accuracy
from utils.walkforward_simulation import walkforward_simulation
from multi_level_predictor import run_prediction_levels, plot_level_accuracy_comparison
from utils.logger import save_predictions, save_predicted_sets, compare_with_actual

# --- Logger ---
logger = setup_logger()
logger.info("🚀 Lotto Predictor started.")

# --- Page Setup ---
st.set_page_config(page_title="Lotto Predictor Pro", layout="wide")
st.title("🎯 Multi-Level Lotto Predictor")

# --- Load Data ---
try:
    draw_df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
    tickets_df = pd.read_excel("data/active_tickets.xlsx")
    draw_df.columns = draw_df.columns.astype(str).str.strip()
    tickets_df.columns = tickets_df.columns.astype(str).str.strip()
    st.success(f"✅ Loaded {len(draw_df)} draws and {len(tickets_df)} tickets.")
except Exception as e:
    logger.error(f"❌ Error loading data: {e}", exc_info=True)
    st.error("❌ Failed to load Excel files.")
    st.stop()

# --- Prepare ---
draw_df = draw_df.sort_values("Draw Number").reset_index(drop=True)
draw_df["DrawIndex"] = draw_df.index
latest_draw_number = draw_df["Draw Number"].max()
exclude_input = st.text_input("🔧 Exclude numbers (comma-separated):")
exclude_list = get_excluded_numbers(exclude_input) if exclude_input else []

expected_cols = [str(i) for i in range(1, 7)]
ticket_sets = tickets_df[expected_cols].dropna(how="any").values.tolist()
ticket_excludes = [set(map(int, row)) for row in ticket_sets]

st.subheader("📋 Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Draws", len(draw_df))
col2.metric("Active Tickets", len(ticket_excludes))
col3.metric("Excluded Numbers", len(exclude_list))

# --- 🔮 Prediction ---
st.subheader("🔮 Multi-Level Prediction")
apply_hybrid = st.checkbox("🔬 Apply Hybrid Filtering", value=True)
run_button = st.button("▶️ Run All Prediction Levels")

if run_button:
    with st.spinner("Running all prediction levels..."):
        prediction_outputs, level_accuracy_df = run_prediction_levels(
            draw_df,
            exclude_list=exclude_list,
            ticket_excludes=ticket_excludes,
            top_n=10,
            apply_hybrid=apply_hybrid
        )

        for label, sets in prediction_outputs.items():
            st.markdown(f"**🔹 {label}**")
            for i, val in enumerate(sets):
                if isinstance(val, (list, tuple)) and len(val) >= 7:
                    main_nums = list(map(int, val[:6]))
                    powerball = int(val[6])
                    st.write(f"{i+1}. {main_nums} + Powerball: {powerball}")
                else:
                    st.write(f"{i+1}. {val}")

        # Save prediction sets
        try:
            save_predictions(prediction_outputs, draw_number=latest_draw_number + 1, draw_date=pd.Timestamp.today())
            save_predicted_sets(latest_draw_number + 1, prediction_outputs)
        except Exception as e:
            logger.warning(f"⚠️ Failed saving predictions: {e}", exc_info=True)

        # Visualize level comparison
        st.subheader("📊 Prediction Accuracy Comparison Level")
        if not level_accuracy_df.empty:
            fig = plot_level_accuracy_comparison(level_accuracy_df)
            st.pyplot(fig)
        else:
            st.info("No accuracy data available.")