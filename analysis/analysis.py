# utils/analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def plot_hot_cold_numbers(df):
    """Generate a frequency chart of number usage across historical draws."""
    try:
        all_nums = pd.Series(df[NUMBER_COLUMNS].values.ravel())
        freq = all_nums.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=freq.index, y=freq.values, ax=ax, hue=freq.index, legend=False)
        ax.set_title("Number Frequency Analysis")
        ax.set_xlabel("Number")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"❌ Failed to generate Hot/Cold plot: {e}", exc_info=True)
        return None

def calculate_accuracy_table(df, models):
    """Evaluate prediction accuracy for each model on historical data."""
    try:
        df = df.copy()
        acc_data = []

        if "DrawIndex" not in df.columns:
            df["DrawIndex"] = np.arange(len(df))

        X = df[["DrawIndex"]]  # Match training features

        # Position-wise models
        for i, col in enumerate(NUMBER_COLUMNS):
            model = models.get(col)
            if model:
                preds = model.predict(X)
                actual = df[col].values
                correct = sum(int(round(p)) == a for p, a in zip(preds, actual))
                acc = correct / len(df) * 100
                acc_data.append((f"Position {i+1}", f"{acc:.2f}%"))
            else:
                logging.warning(f"⚠️ No model found for position {col}")

        # Full set model
        if models.get("full"):
            preds_full = models["full"].predict(X)
            actual = df[NUMBER_COLUMNS].values
            correct_sets = sum(
                all(int(round(p)) == a for p, a in zip(p_row, a_row))
                for p_row, a_row in zip(preds_full, actual)
            )
            acc = correct_sets / len(df) * 100
            acc_data.append(("Full Set", f"{acc:.2f}%"))
        else:
            logging.warning("⚠️ No full-set model found.")

        # PowerBall model
        if models.get(POWERBALL_COLUMN):
            preds_pb = models[POWERBALL_COLUMN].predict(X)
            actual_pb = df[POWERBALL_COLUMN].values
            correct_pb = sum(p == a for p, a in zip(preds_pb, actual_pb))
            acc = correct_pb / len(df) * 100
            acc_data.append(("PowerBall", f"{acc:.2f}%"))
        else:
            logging.warning("⚠️ No PowerBall model found.")

        return pd.DataFrame(acc_data, columns=["Model", "Accuracy"])

    except Exception as e:
        logging.error(f"❌ Failed to calculate accuracy table: {e}", exc_info=True)
        return pd.DataFrame()
