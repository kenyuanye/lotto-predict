import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict

from data_loader import load_draw_history
from predictors.reward_learning_predictor import (
    predict_with_reward_learning,
    reward_learning_predictor,
)

CSV_LOG_FILE = "data/reward_learning_log.csv"
BEST_FORMULA_FILE = "data/best_formula.json"

@st.cache_data(show_spinner=False)
def get_draw_history() -> List[List[int]]:
    df = load_draw_history()
    return df[["1", "2", "3", "4", "5", "6", "Power Ball"]].values.tolist()

def render_level_1_dashboard(level1_predictions: Dict[str, List[int]]) -> None:
    st.subheader("🔍 Level 1: Individual Method Predictions")
    if not level1_predictions:
        st.warning("No Level 1 predictions available.")
        return

    cols = st.columns(3)
    for idx, (method, numbers) in enumerate(level1_predictions.items()):
        with cols[idx % 3]:
            st.markdown(f"**{method}**")
            st.write(f"🎯 Main: {numbers[:6]}")
            st.write(f"⭐ Powerball: {numbers[6]}")

def render_level_2_dashboard(level2_predictions: Dict[str, List[int]]) -> None:
    st.subheader("🧪 Level 2: Method Combinations")
    st.info("Level 2 prediction logic not yet implemented.")

def render_level_3_dashboard(level3_predictions: Dict[str, List[int]]) -> None:
    st.subheader("➕ Level 3: Combinations + Custom Rules")
    st.info("Level 3 prediction logic not yet implemented.")

def render_level_4_dashboard(level4_predictions: Dict[str, List[int]]) -> None:
    st.subheader("🎛 Level 4: All Methods Combined")
    st.info("Level 4 prediction logic not yet implemented.")

def render_level_5_dashboard() -> None:
    st.subheader("🧬 Level 5: Advanced / Meta Prediction")
    st.info("Level 5 logic not yet implemented.")

def render_reward_learning_dashboard() -> None:
    st.subheader("🧠 Reward Learning Formula Comparison")

    st.markdown("### 🧪 Train New Reward Learning Formula")
    if st.button("Run Reward Learning Predictor Now"):
        with st.status("Training reward learning model... please wait.", expanded=True) as status:
            try:
                draw_history = get_draw_history()
                reward_learning_predictor(draw_history)
                status.update(label="✅ Training complete!", state="complete")
            except Exception as e:
                st.error(f"❌ Failed to run predictor: {e}")
                status.update(label="❌ Training failed", state="error")

    try:
        log_df = pd.read_csv(CSV_LOG_FILE)
        st.dataframe(log_df)

        st.markdown("### 📈 Formula A vs B Accuracy")
        fig, ax = plt.subplots()
        ax.plot(log_df["DrawIndex"], log_df["FormulaA_MainMatch"], label="Formula A", marker="o")
        ax.plot(log_df["DrawIndex"], log_df["FormulaB_MainMatch"], label="Formula B", marker="x")
        ax.set_xlabel("Draw Index")
        ax.set_ylabel("Main Match Count")
        ax.legend()
        st.pyplot(fig)

        st.markdown("### 🔁 Attempts Until Match")
        fig2, ax2 = plt.subplots()
        ax2.bar(log_df["DrawIndex"], log_df["Attempts"], color="orange")
        ax2.set_xlabel("Draw Index")
        ax2.set_ylabel("Attempts")
        st.pyplot(fig2)

    except FileNotFoundError:
        st.warning("⚠️ Reward learning log not found. Run the predictor first.")

    st.markdown("### 🧮 Current Best Formula (A)")
    try:
        with open(BEST_FORMULA_FILE, "r") as f:
            formula: Dict = json.load(f)
        st.json(formula)
    except FileNotFoundError:
        st.warning("⚠️ Best formula not found.")

    st.markdown("### 🎰 Generate Sets with Formula")
    n_sets: int = st.slider("How many sets?", 1, 20, 10)
    if st.button("Generate Sets"):
        sets: List[List[int]] = predict_with_reward_learning(n_sets)
        if sets:
            for i, pred in enumerate(sets, 1):
                st.write(f"{i}. 🎯 {pred[:6]} + PB: {pred[6]}")
        else:
            st.error("No predictions generated. Ensure best formula file exists.")