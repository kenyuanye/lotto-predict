# main.py

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import setup_logger, save_predictions, compare_with_actual, save_predicted_sets
from utils.model_selector import train_and_save_best_models, evaluate_models
from utils.trainer import train_models, run_sequential_training
from utils.predictor import predict_all, get_excluded_numbers
from utils.analysis import plot_hot_cold_numbers, calculate_accuracy_table
from utils.model_auto_selector import auto_select_best_model
from utils.hybrid_predictor import filter_and_rank_sets, generate_unlikely_sets, generate_partial_match_sets
from utils.accuracy_tracker import track_accuracy, plot_accuracy_breakdown
from utils.symbolic_trainer import train_symbolic_models
from utils.symbolic_predictor import predict_symbolic, load_powerball_model_with_fallback
from utils.symbolic_vs_rf_comparison import compare_model_accuracy
from utils.walkforward_simulation import walkforward_simulation

# --- Logger ---
logger = setup_logger()
logger.info("🚀 Lotto Predictor started.")

# --- Page Setup ---
st.set_page_config(page_title="Lotto Predictor Pro", layout="wide")
st.title("🎯 Advanced Lotto Predictor")

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

full_model = None 

# --- Manual Exclusions ---
exclude_input = st.text_input("🔧 Exclude numbers (comma-separated):")
exclude_list = get_excluded_numbers(exclude_input) if exclude_input else []

# --- Ticket Exclusions ---
ticket_excludes = []
try:
    expected_cols = [str(i) for i in range(1, 7)]
    if all(col in tickets_df.columns for col in expected_cols):
        ticket_sets = tickets_df[expected_cols].dropna(how="any").values.tolist()
        ticket_excludes = [set(map(int, row)) for row in ticket_sets]
except Exception as e:
    logger.warning(f"⚠️ Failed processing ticket exclusions: {e}", exc_info=True)

# --- Prepare Data ---
draw_df = draw_df.sort_values("Draw Number").reset_index(drop=True)
draw_df["DrawIndex"] = draw_df.index

# --- Summary Dashboard ---
st.subheader("📋 Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Draws", len(draw_df))
col2.metric("Active Tickets", len(ticket_excludes))
col3.metric("Excluded Numbers", len(exclude_list))

# --- Model Training Section ---
st.subheader("🧠 Train or Retrain Models")

if st.button("Train/Re-train Random Forest Models"):
    with st.spinner("Training Random Forest models..."):
        train_models(draw_df, logger=logger)
    st.success("✅ Random Forest models trained.")

if st.button("Train Symbolic Ultra-Accurate Models"):
    with st.spinner("Training Symbolic models (may take ~5-10min)..."):
        train_symbolic_models(draw_df, logger=logger)
    st.success("✅ Symbolic models trained.")

if st.button("Retrain Everything (Real + Synthetic + Blended)"):
    with st.spinner("Retraining full model suite..."):
        from train_all_models import main as train_all_main
        train_all_main()
    st.success("✅ All models retrained.")

# --- 🔮 Prediction Section ---
st.subheader("🔮 Predictions")

apply_hybrid = st.checkbox("🔬 Apply Hybrid Filtering", value=True)
use_symbolic = st.checkbox("🧠 Use Symbolic Models", value=False)

filtered_results = {}
ensemble_final_set = None

try:
    # Load best models
    from utils.symbolic_predictor import load_powerball_model_with_fallback
    from utils.custom_rules import calculate_powerball_gaps, powerball_bias_weight  # <-- NEW

    full_model, _ = auto_select_best_model(draw_df, for_powerball=False)
    pb_model_fallback, _ = load_powerball_model_with_fallback()

    prediction_models = {"Top Picks": full_model}

    # Attach PowerBall model separately
    if pb_model_fallback:
        prediction_models["Power Ball"] = pb_model_fallback
        logger.info("✅ PowerBall model loaded into predictions.")
    else:
        logger.warning("⚠️ No PowerBall model available.")

    latest_draw_number = draw_df["Draw Number"].max()

    # --- PowerBall Gaps Bias Map
    powerball_gaps = calculate_powerball_gaps(draw_df)

    if use_symbolic:
        prediction_output = predict_symbolic(draw_df, top_n=10, powerball_gaps=powerball_gaps)
        results = {"Symbolic Picks": prediction_output["predictions"]}
    else:
        raw_results = predict_all(draw_df, prediction_models, exclude_list + ticket_excludes, top_n=10)

        # --- Apply PowerBall Bias Adjustment ---
        results = {}
        for key, sets in raw_results.items():
            biased_sets = []
            for s in sets:
                main_nums = list(map(int, s[:6]))
                original_pb = int(s[6])

                if powerball_gaps:
                    # Boost the probability
                    bias = powerball_bias_weight(powerball_gaps.get(original_pb, 0))
                    if np.random.rand() < min(1, bias / 2):  # E.g., 1.5 bias = 75% chance to keep
                        biased_pb = original_pb
                    else:
                        # Replace with higher gap PowerBall
                        candidate_pbs = sorted(powerball_gaps.items(), key=lambda x: -x[1])
                        biased_pb = candidate_pbs[0][0] if candidate_pbs else original_pb
                else:
                    biased_pb = original_pb

                biased_sets.append(main_nums + [biased_pb])
            results[key] = biased_sets       

# --- 🎯 Show PowerBall Bias Chart (Optional) ---
    if use_symbolic and "pb_bias_weights" in prediction_output:
        st.subheader("🎯 PowerBall Bias Weight Chart (Overdue Numbers Boosted)")

        try:
            bias_data = pd.DataFrame({
                "PowerBall Number": list(prediction_output["pb_bias_weights"].keys()),
                "Bias Weight": list(prediction_output["pb_bias_weights"].values())
            }).sort_values("PowerBall Number")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(bias_data["PowerBall Number"], bias_data["Bias Weight"])
            ax.set_xlabel("PowerBall Number")
            ax.set_ylabel("Bias Weight (Higher = More Overdue)")
            ax.set_title("PowerBall Bias Weights Based on Draw Gaps")
            st.pyplot(fig)

        except Exception as e:
            logger.error(f"❌ Failed to render PowerBall bias chart: {e}", exc_info=True)
            st.error("Failed to generate PowerBall bias weight chart.")


    # Save Predictions
    try:
        save_predictions(results, draw_number=latest_draw_number + 1, draw_date=pd.Timestamp.today())
        save_predicted_sets(latest_draw_number + 1, results)
    except Exception as e:
        logger.error(f"⚠️ Failed saving predictions: {e}", exc_info=True)

    # --- Display Raw Predictions ---
    with st.expander("📜 Raw Model Predictions"):
        for key, sets in results.items():
            st.markdown(f"**{key}**")
            for i, val in enumerate(sets):
                if isinstance(val, (list, tuple)) and len(val) >= 7:
                    main_nums = list(map(int, val[:6]))
                    powerball = int(val[6])
                    st.write(f"{i+1}. {main_nums} + Powerball: {powerball}")
                else:
                    st.write(f"{i+1}. {val}")

    # --- Hybrid Filtering ---
    for key, sets in results.items():
        if apply_hybrid:
            filtered, _ = filter_and_rank_sets(
                sets,
                draw_history=draw_df,
                ticket_excludes=ticket_excludes,
                top_n=10
            )
            filtered_results[key] = filtered
        else:
            filtered_results[key] = sets

    with st.expander("🧹 Hybrid Filtered Predictions"):
        for key, sets in filtered_results.items():
            st.markdown(f"**{key}** (filtered):")
            for i, val in enumerate(sets):
                if isinstance(val, (list, tuple)) and len(val) >= 7:
                    main_nums = list(map(int, val[:6]))
                    powerball = int(val[6])
                    st.write(f"{i+1}. {main_nums} + Powerball: {powerball}")
                else:
                    st.write(f"{i+1}. {val}")

except Exception as e:
    logger.error(f"❌ Prediction failed: {e}", exc_info=True)
    st.error("Prediction failed.")


# --- 🧊 Top 10 Most Unlikely Sets ---
st.subheader("🧊 Top 10 Most Unlikely Sets")
try:
    unlikely_sets = generate_unlikely_sets(draw_df, top_n=10)
    if unlikely_sets:
        for i, s in enumerate(unlikely_sets):
            main_nums = [int(x) for x in s[:6]]
            powerball = int(s[6])
            st.write(f"❄️ {i+1}. {main_nums} + Powerball: {powerball}")
    else:
        st.warning("⚠️ No unlikely sets generated.")
except Exception as e:
    logger.error(f"⚠️ Failed unlikely sets: {e}", exc_info=True)
    st.error("Failed to generate unlikely sets.")

# --- 🧩 50% Match Sets ---
st.subheader("🧩 50% Match Sets")
try:
    base_sets = unlikely_sets + filtered_results.get("Top Picks", [])
    partials = generate_partial_match_sets(base_sets, top_n=10)
    if partials:
        for i, s in enumerate(partials):
            main_nums = [int(x) for x in s[:6]]
            powerball = int(s[6])
            st.write(f"🎯 {i+1}. {main_nums} + Powerball: {powerball}")
    else:
        st.warning("⚠️ No partial match sets generated.")
except Exception as e:
    logger.error(f"⚠️ Failed partial matches: {e}", exc_info=True)
    st.error("Failed to generate partial matches.")

# --- 🚶 Walkforward Simulation ---
st.subheader("🚶 Walk-Forward Simulation (Predicting One Draw Ahead Step-by-Step)")

walkforward_mode = st.selectbox("Walkforward Mode:", ["Traditional Walkforward", "Coverage Walkforward"])

if st.button("🚶 Run Walkforward Simulation", key="walkforward_run"):
    try:
        with st.spinner("Running walkforward simulation..."):
            wf_df = walkforward_simulation(draw_df)

        if walkforward_mode == "Traditional Walkforward":
            st.success("✅ Traditional Walkforward Completed.")
            avg_main = wf_df["Main Numbers Matched"].mean()
            avg_pb = wf_df["PowerBall Matched"].mean() * 100

            col1, col2 = st.columns(2)
            col1.metric("Avg Main Matches", f"{avg_main:.2f}")
            col2.metric("PowerBall Match Rate (%)", f"{avg_pb:.2f}")

            st.line_chart(wf_df.set_index("Draw Number")["Main Numbers Matched"])

        elif walkforward_mode == "Coverage Walkforward":
            st.success("✅ Coverage-Based Walkforward Completed.")
            avg_top = wf_df["Top Picks Coverage %"].mean()
            avg_backup = wf_df["Backup Picks Coverage %"].mean()
            avg_unlikely = wf_df["Unlikely Picks Coverage %"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Top Picks Coverage", f"{avg_top:.2f}%")
            col2.metric("Backup Picks Coverage", f"{avg_backup:.2f}%")
            col3.metric("Unlikely Picks Coverage", f"{avg_unlikely:.2f}%")

            st.line_chart(wf_df.set_index("Draw Number")[[
                "Top Picks Coverage %",
                "Backup Picks Coverage %",
                "Unlikely Picks Coverage %"
            ]])

        st.dataframe(wf_df)

    except Exception as e:
        logger.error(f"❌ Walkforward simulation failed: {e}", exc_info=True)
        st.error(f"Walkforward simulation failed: {e}")

# --- 📊 Hot, Cold, Frequency Analysis ---
st.subheader("📊 Hot, Cold, Frequency Analysis")
try:
    st.pyplot(plot_hot_cold_numbers(draw_df))
    logger.info("📊 Hot/Cold/Frequency chart rendered.")
except Exception as e:
    logger.error(f"❌ Failed hot/cold plot: {e}", exc_info=True)
    st.error("Failed to generate Hot/Cold Frequency Chart.")

# --- 📈 Model Accuracy Table ---
st.subheader("📈 Model Accuracy Table")
try:
    acc_table = calculate_accuracy_table(draw_df, {"full": full_model})
    st.dataframe(acc_table)
    logger.info("📈 Model Accuracy table displayed.")
except Exception as e:
    logger.error(f"❌ Failed accuracy table: {e}", exc_info=True)
    st.error("Failed to generate accuracy table.")

# --- 🔎 Symbolic vs Random Forest Model Comparison ---
st.subheader("🔎 Symbolic vs Random Forest Comparison")

if st.checkbox("Compare Symbolic vs RF Models"):
    try:
        comp_df = compare_model_accuracy(draw_df)
        if not comp_df.empty:
            def highlight(val):
                if val == "Symbolic":
                    return "background-color: lightgreen; font-weight: bold"
                elif val == "RF":
                    return "background-color: lightcoral; font-weight: bold"
                return ""

            styled = comp_df.style.applymap(highlight, subset=["Better Model"])
            st.dataframe(styled)
            st.success("✅ Symbolic vs RF Comparison done.")
        else:
            st.warning("⚠️ No comparison data available.")
    except Exception as e:
        logger.error(f"❌ Failed symbolic vs RF comparison: {e}", exc_info=True)
        st.error("Failed Symbolic vs RF comparison.")

# --- 🧾 Past Prediction Accuracy (Prediction Log) ---
st.subheader("🧾 Past Prediction Accuracy")

try:
    if os.path.exists("data/predicted_sets_log.csv") and os.path.exists("data/draw_history.xlsx"):
        pred_log_df = compare_with_actual(draw_df)
        if not pred_log_df.empty:
            st.dataframe(pred_log_df)
        else:
            st.warning("⚠️ No past prediction matches yet.")
    else:
        st.warning("⚠️ No prediction log found yet.")
except Exception as e:
    logger.error(f"❌ Failed past prediction comparison: {e}", exc_info=True)
    st.error("Failed to analyze past predictions.")

# --- 📈 Advanced Accuracy Breakdown ---
st.subheader("📈 Advanced Accuracy Breakdown")
try:
    breakdown = track_accuracy(draw_df, {"full": full_model})
    st.dataframe(breakdown)

    fig = plot_accuracy_breakdown(breakdown)
    if fig:
        st.pyplot(fig)
    else:
        st.info("No breakdown chart available.")
except Exception as e:
    logger.error(f"❌ Failed advanced breakdown: {e}", exc_info=True)
    st.error("Failed Advanced Accuracy Breakdown.")

# --- 📈 Prediction Set Analysis ---
st.subheader("📈 Prediction Set Analysis (Historical)")

try:
    if os.path.exists("data/predicted_sets_log.csv") and os.path.exists("data/draw_history.xlsx"):
        pred_df = pd.read_csv("data/predicted_sets_log.csv")
        draw_df_reloaded = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
        draw_df_reloaded.columns = draw_df_reloaded.columns.astype(str).str.strip()

        results = []
        for _, row in pred_df.iterrows():
            draw_num = row["Draw Number"]
            numbers = row["Numbers"]
            powerball_pred = row["Powerball"]

            actual_draw = draw_df_reloaded[draw_df_reloaded["Draw Number"] == draw_num]
            if actual_draw.empty:
                continue

            actual_nums = set(actual_draw.iloc[0][["1", "2", "3", "4", "5", "6"]])
            actual_pb = int(actual_draw.iloc[0]["Power Ball"])

            predicted_nums = set(map(int, numbers.split(", ")))

            matched_main = len(predicted_nums.intersection(actual_nums))
            matched_powerball = (int(powerball_pred) == actual_pb)

            results.append({
                "Draw Number": draw_num,
                "Source": row["Source"],
                "Matched Main Numbers": matched_main,
                "Matched Powerball": matched_powerball,
                "Prediction Time": row.get("Prediction Time", "Unknown")
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            st.dataframe(results_df)

            st.subheader("📊 Avg Main Number Match by Source")
            summary = results_df.groupby("Source")["Matched Main Numbers"].mean().reset_index()
            st.bar_chart(summary.set_index("Source"))
        else:
            st.warning("⚠️ No historical matching results found.")
    else:
        st.warning("⚠️ No historical prediction logs found.")
except Exception as e:
    logger.error(f"❌ Failed prediction set analysis: {e}", exc_info=True)
    st.error("Failed prediction set analysis.")

# --- 📈 PowerBall Gap Chart ---
st.subheader("📈 PowerBall Gap Analysis")

try:
    from utils.custom_rules import calculate_powerball_gaps

    pb_gaps = calculate_powerball_gaps(draw_df)
    if pb_gaps:
        gap_df = pd.DataFrame({
            "PowerBall Number": list(pb_gaps.keys()),
            "Draws Since Last Seen": list(pb_gaps.values())
        }).sort_values("Draws Since Last Seen", ascending=False)

        st.dataframe(gap_df)

        st.subheader("📊 PowerBall Number Gaps")
        fig, ax = plt.subplots()
        ax.bar(gap_df["PowerBall Number"], gap_df["Draws Since Last Seen"])
        ax.set_xlabel("PowerBall Number")
        ax.set_ylabel("Draws Since Last Seen")
        ax.set_title("PowerBall Numbers Overdue Analysis")
        st.pyplot(fig)
    else:
        st.info("No gap data available.")
except Exception as e:
    logger.error(f"❌ Failed PowerBall gap chart: {e}", exc_info=True)
    st.error("Failed to generate PowerBall Gap Chart.")

