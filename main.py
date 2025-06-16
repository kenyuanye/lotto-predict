import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Core utilities
from utils.logger import setup_logger, save_predictions, save_predicted_sets, compare_with_actual
from utils.model_utils import get_available_models
from legacy_prediction.predictor import get_excluded_numbers
from legacy_prediction.hybrid_predictor import filter_and_rank_sets

# Rules and analysis
from rules.custom_rules import calculate_powerball_gaps
from analysis.accuracy_tracker import track_accuracy, plot_accuracy_breakdown
from analysis.symbolic_vs_rf_comparison import compare_model_accuracy
from analysis.walkforward_simulation import walkforward_simulation
from analysis.analysis import plot_hot_cold_numbers

# Data
from data_loader import load_draw_history, load_active_tickets

# Multi-level predictor controller
from predictors.level_predictor import run_prediction_levels
from ui.dashboard import plot_level_accuracy_comparison


# --- Logger ---
logger = setup_logger()
logger.info("🚀 Lotto Predictor started.")

# --- Page Setup ---
st.set_page_config(page_title="Lotto Predictor Pro", layout="wide")
st.title("🎯 Multi-Level Lotto Predictor")

# --- Load Data ---
draw_df = load_draw_history()
ticket_excludes = load_active_tickets()

if draw_df.empty:
    st.error("❌ Failed to load draw history.")
    st.stop()

latest_draw_number = draw_df["Draw Number"].max()

# --- Manual Exclusions ---
exclude_input = st.text_input("🔧 Exclude numbers (comma-separated):")
exclude_list = get_excluded_numbers(exclude_input) if exclude_input else []

# --- Summary UI ---
st.subheader("📋 Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Draws", len(draw_df))
col2.metric("Active Tickets", len(ticket_excludes))
col3.metric("Excluded Numbers", len(exclude_list))

# --- 🔮 Prediction Section ---
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

        # Save predictions
        try:
            save_predictions(prediction_outputs, draw_number=latest_draw_number + 1, draw_date=pd.Timestamp.today())
            save_predicted_sets(latest_draw_number + 1, prediction_outputs)
        except Exception as e:
            logger.warning(f"⚠️ Failed saving predictions: {e}", exc_info=True)

        # Visualize comparison
        st.subheader("📊 Prediction Accuracy Comparison Level")
        fig = plot_level_accuracy_comparison(level_accuracy_df)
        if fig:
            st.pyplot(fig)
        else:
            st.info("No accuracy data available.")