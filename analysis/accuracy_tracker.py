import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import logging
import os

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]
POWERBALL_COLUMN = "Power Ball"
ACCURACY_LOG_PATH = "data/overall_predictions_log.csv"

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

def update_accuracy_for_method(method_name, draw_df, latest_draw):
    try:
        required_columns = ["Method", "Draw Number", "Predicted Numbers", "Accuracy %"]

        # Load or initialize log
        if os.path.exists(ACCURACY_LOG_PATH):
            try:
                method_df = pd.read_csv(ACCURACY_LOG_PATH)
            except pd.errors.EmptyDataError:
                method_df = pd.DataFrame(columns=required_columns)
        else:
            method_df = pd.DataFrame(columns=required_columns)

        # Ensure required columns exist
        for col in required_columns:
            if col not in method_df.columns:
                method_df[col] = None

        # Convert latest_draw to Series if it's a DataFrame
        if isinstance(latest_draw, pd.DataFrame):
            latest_draw = latest_draw.iloc[0]

        draw_number = latest_draw["Draw Number"] if "Draw Number" in latest_draw else None
        if draw_number is None:
            logging.warning(f"⚠️ Draw Number missing for accuracy update for method {method_name}.")
            return

        pred_row = method_df[
            (method_df["Method"] == method_name) &
            (method_df["Draw Number"] == draw_number)
        ]

        if pred_row.empty:
            logging.warning(f"⚠️ No prediction recorded for {method_name} on draw {draw_number}")
            return

        pred_nums = pred_row.iloc[0]["Predicted Numbers"]
        try:
            pred_list = list(map(int, str(pred_nums).strip("[]").split(",")))
        except Exception as e:
            logging.error(f"❌ Failed parsing predictions for {method_name}: {e}")
            return

        actual = list(latest_draw[NUMBER_COLUMNS].values) + [latest_draw[POWERBALL_COLUMN]]
        matches = sum(1 for a, b in zip(actual, pred_list) if a == b)
        accuracy_pct = round(matches / 7 * 100, 2)

        method_df.loc[
            (method_df["Method"] == method_name) &
            (method_df["Draw Number"] == draw_number),
            "Accuracy %"
        ] = accuracy_pct

        method_df.to_csv(ACCURACY_LOG_PATH, index=False)
        logging.info(f"✅ Accuracy updated for {method_name} on draw {draw_number}: {accuracy_pct}%")

    except Exception as e:
        logging.error(f"❌ Failed to update accuracy for {method_name}: {e}", exc_info=True)
        

def build_accuracy_comparison_dataframe(results_dict):
    """
    Build a DataFrame showing accuracy % per method from the log file
    for the most recent draw found in the results_dict.
    """
    if not os.path.exists(ACCURACY_LOG_PATH):
        return pd.DataFrame(columns=["Method", "Accuracy %"])

    try:
        df = pd.read_csv(ACCURACY_LOG_PATH)
        if df.empty or "Draw Number" not in df.columns:
            return pd.DataFrame(columns=["Method", "Accuracy %"])

        latest_draw = df["Draw Number"].max()
        df_latest = df[df["Draw Number"] == latest_draw]
        return df_latest[["Method", "Accuracy %"]].dropna()
    except Exception as e:
        logging.error(f"❌ Could not load accuracy data: {e}", exc_info=True)
        return pd.DataFrame(columns=["Method", "Accuracy %"])


