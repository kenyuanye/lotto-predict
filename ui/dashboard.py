# ui/dashboard.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def render_legacy_dashboard(draw_data):
    st.subheader("Legacy Prediction View")
    st.info("Legacy logic remains in your main.py interface.")
    st.write("Prediction and accuracy charts are handled there.")

def render_level_1_dashboard(results: dict):
    st.markdown("### 🔍 Top 10 Predictions for Each Method")

    for method_name, predictions in results.items():
        is_residual = method_name.lower() == "residual"
        section_title = f"#### 🔢 {method_name.replace('_', ' ').title()}"
        if is_residual:
            section_title += " 🧠"

        st.markdown(section_title)

        if not predictions:
            st.warning(f"⚠️ No predictions returned for `{method_name}`.")
            if is_residual:
                st.info("ℹ️ Residual predictions might be missing due to model load failure, prediction error, or formatting issue.")
                st.text("🔍 Debug: results['residual'] was empty or None.")
            continue

        # Defensive debug logging for residuals
        if is_residual:
            st.code(f"[residual] Raw: {predictions}", language="text")

        if isinstance(predictions, list) and all(isinstance(p, list) and len(p) == 7 for p in predictions):
            for i, pred in enumerate(predictions[:10]):
                try:
                    main = ", ".join(str(n) for n in pred[:6])
                    pb = pred[6]
                    st.write(f"Set {i+1}: 🎱 [{main}] | ⭐ Powerball: {pb}")
                except Exception as e:
                    st.error(f"❌ Error rendering prediction #{i+1}: {e}")
                    if is_residual:
                        st.exception(e)
        else:
            st.warning(f"⚠️ Unexpected format for predictions from `{method_name}`")
            st.write(predictions)

def plot_level_accuracy_comparison(df):
    """
    Create a bar chart comparing prediction accuracy for each method/level.
    Includes checks for missing or non-numeric data.
    """
    st.markdown("### 📊 Prediction Accuracy by Method")

    if df.empty:
        st.warning("⚠️ Accuracy DataFrame is empty.")
        return None

    if "Accuracy %" not in df.columns:
        st.error("🚫 'Accuracy %' column not found in the data.")
        return None

    # Clean and ensure numeric accuracy column
    df["Accuracy %"] = pd.to_numeric(df["Accuracy %"], errors="coerce")
    df = df.dropna(subset=["Accuracy %"])

    if df.empty:
        st.warning("⚠️ No valid numeric accuracy data available for plotting.")
        return None

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(kind="bar", x="Method", y="Accuracy %", ax=ax, legend=False, color='skyblue')
    ax.set_title("📊 Prediction Accuracy by Method")
    ax.set_ylabel("Accuracy %")
    ax.set_xlabel("Prediction Method")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
