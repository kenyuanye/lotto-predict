# utils/accuracy_tracker.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import logging

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"

def track_accuracy(df, models):
    df = df.copy().reset_index(drop=True)
    df["DrawIndex"] = df.index
    X = df[["DrawIndex"]]
    breakdown = []

    for idx, col in enumerate(NUMBER_COLUMNS):
        try:
            if col in models:
                model = models[col]
                y_true = df[col].values
                y_pred = np.round(model.predict(X)).astype(int)
            elif "full" in models:
                model = models["full"]
                y_true = df[col].values
                y_pred_full = np.round(model.predict(X)).astype(int)
                y_pred = y_pred_full[:, idx]
            else:
                raise ValueError(f"No model available for {col}")

            acc = accuracy_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            breakdown.append({
                "Position": f"Pos {col}",
                "Accuracy %": round(acc * 100, 2),
                "MAE": round(mae, 3)
            })

        except Exception as e:
            logging.error(f"❌ Accuracy failed for {col}: {e}", exc_info=True)
            breakdown.append({
                "Position": f"Pos {col}",
                "Accuracy %": np.nan,
                "MAE": np.nan
            })

    # Full set accuracy
    try:
        if "full" in models:
            model_full = models["full"]
            y_true = df[NUMBER_COLUMNS].values
            y_pred = np.round(model_full.predict(X)).astype(int)
            full_matches = np.sum(np.all(y_true == y_pred, axis=1))
            full_acc = full_matches / len(df)
            breakdown.append({
                "Position": "Full Set",
                "Accuracy %": round(full_acc * 100, 2),
                "MAE": np.nan
            })
        else:
            logging.warning("⚠️ Full set model not available for full accuracy.")
    except Exception as e:
        logging.error(f"❌ Full set accuracy failed: {e}", exc_info=True)
        breakdown.append({
            "Position": "Full Set",
            "Accuracy %": np.nan,
            "MAE": np.nan
        })

    # Power Ball accuracy
    try:
        if "Power Ball" in models:
            model_pb = models["Power Ball"]
            y_true = df[POWERBALL_COLUMN].values
            y_pred = np.round(model_pb.predict(X)).astype(int)
            acc = accuracy_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            breakdown.append({
                "Position": "Power Ball",
                "Accuracy %": round(acc * 100, 2),
                "MAE": round(mae, 3)
            })
        else:
            logging.warning("⚠️ PowerBall model not available for accuracy.")
    except Exception as e:
        logging.error(f"❌ Power Ball accuracy failed: {e}", exc_info=True)
        breakdown.append({
            "Position": "Power Ball",
            "Accuracy %": np.nan,
            "MAE": np.nan
        })

    return pd.DataFrame(breakdown)

def plot_accuracy_breakdown(df: pd.DataFrame):
    try:
        df_plot = df.dropna()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df_plot["Position"], df_plot["Accuracy %"], color="green")
        ax.set_title("Model Accuracy by Position")
        ax.set_ylabel("Accuracy %")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"❌ Failed to plot accuracy breakdown: {e}", exc_info=True)
        return None

def plot_historical_accuracy_trend(draw_df: pd.DataFrame, models: dict, window: int = 50):
    try:
        df = draw_df.copy().reset_index(drop=True)
        df["DrawIndex"] = df.index
        accuracies = []

        if "full" not in models:
            logging.warning("⚠️ Full set model not available for historical trend.")
            return None

        model_full = models["full"]

        for i in range(window, len(df)):
            df_slice = df.iloc[i - window:i]
            X = df_slice[["DrawIndex"]]
            y_true = df_slice[NUMBER_COLUMNS]
            y_pred = np.round(model_full.predict(X)).astype(int)
            matches = np.all(y_true.values == y_pred, axis=1)
            match_rate = np.mean(matches)
            accuracies.append((df_slice.iloc[-1]["Draw Date"], round(match_rate * 100, 2)))

        if not accuracies:
            return None

        dates, acc_vals = zip(*accuracies)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates, acc_vals, marker="o")
        ax.set_title("📈 Historical Full Set Accuracy (Rolling Window)")
        ax.set_ylabel("Accuracy %")
        ax.set_xlabel("Draw Date")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    except Exception as e:
        logging.error(f"❌ Failed to generate historical accuracy trend: {e}", exc_info=True)
        return None
