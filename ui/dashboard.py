# ui/dashboard.py

import streamlit as st
import matplotlib.pyplot as plt

def render_legacy_dashboard(draw_data):
    st.subheader("Legacy Prediction View")
    st.info("Legacy logic remains in your main.py interface.")
    st.write("Prediction and accuracy charts are handled there.")

def render_level_1_dashboard(results: dict):
    st.markdown("### 🔍 Top 10 Predictions for Each Method")
    for method_name, predictions in results.items():
        st.markdown(f"#### 🔢 {method_name.replace('_', ' ').title()}")

        if not predictions:
            st.warning("No predictions returned.")
            continue

        for i, pred in enumerate(predictions[:10]):
            try:
                if isinstance(pred, (list, tuple)) and len(pred) >= 7:
                    main = ", ".join(str(n) for n in pred[:6])
                    pb = pred[6]
                    st.write(f"{i+1}. 🎱 {main} + PB: {pb}")
                else:
                    st.write(f"{i+1}. {pred}")
            except Exception as e:
                st.error(f"Error rendering prediction #{i+1}: {e}")

def plot_level_accuracy_comparison(df):
    """
    Create a bar chart comparing prediction accuracy for each method/level.
    Includes checks for missing or non-numeric data.
    """
    import pandas as pd

    st.markdown("### 📊 Prediction Accuracy by Method")

    if df.empty:
        st.warning("⚠️ Accuracy DataFrame is empty.")
        return None

    if "Accuracy %" not in df.columns:
        st.error("🚫 'Accuracy %' column not found in the data.")
        return None

    # Convert to numeric and drop invalid rows
    df["Accuracy %"] = pd.to_numeric(df["Accuracy %"], errors="coerce")
    df = df.dropna(subset=["Accuracy %"])

    if df.empty:
        st.warning("⚠️ No numeric accuracy data available for plotting.")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    df.plot(kind="bar", x="Method", y="Accuracy %", ax=ax, legend=False)
    ax.set_title("📊 Prediction Accuracy by Method")
    ax.set_ylabel("Accuracy %")
    ax.set_xlabel("Prediction Method")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig
