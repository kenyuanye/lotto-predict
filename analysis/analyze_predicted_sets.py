# analyze_predicted_sets.py

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = "data"
PREDICTED_LOG = os.path.join(DATA_DIR, "predicted_sets_log.csv")
DRAW_HISTORY = os.path.join(DATA_DIR, "draw_history.xlsx")

def load_data():
    """Load predicted sets and draw history."""
    if not os.path.exists(PREDICTED_LOG) or not os.path.exists(DRAW_HISTORY):
        print("❌ Required files not found. Make sure predicted_sets_log.csv and draw_history.xlsx exist.")
        return None, None

    pred_df = pd.read_csv(PREDICTED_LOG)
    draw_df = pd.read_excel(DRAW_HISTORY, parse_dates=["Draw Date"])
    draw_df.columns = draw_df.columns.astype(str).str.strip()

    return pred_df, draw_df

def analyze_matches(pred_df, draw_df):
    """Compare predicted sets with actual draw results."""
    all_results = []

    for _, row in pred_df.iterrows():
        draw_number = row["Draw Number"]
        numbers = row["Numbers"]
        powerball_pred = row["Powerball"]

        # Find the actual draw
        actual_draw = draw_df[draw_df["Draw Number"] == draw_number]
        if actual_draw.empty:
            continue

        actual_numbers = set(actual_draw.iloc[0][["1", "2", "3", "4", "5", "6"]])
        actual_powerball = actual_draw.iloc[0]["Power Ball"]

        predicted_numbers = set(map(int, numbers.split(", ")))

        matched_main = len(predicted_numbers.intersection(actual_numbers))
        matched_powerball = int(powerball_pred) == int(actual_powerball)

        all_results.append({
            "Draw Number": draw_number,
            "Prediction Source": row["Source"],
            "Matched Main Numbers": matched_main,
            "Matched Powerball": matched_powerball,
            "Prediction Time": row["Prediction Time"]
        })

    return pd.DataFrame(all_results)

def plot_match_summary(results_df):
    """Visualize match summary."""
    if results_df.empty:
        print("⚠️ No matching results to plot.")
        return

    summary = results_df.groupby("Prediction Source")["Matched Main Numbers"].mean()

    plt.figure(figsize=(8,5))
    summary.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Matched Main Numbers per Prediction Source")
    plt.ylabel("Average Matches")
    plt.ylim(0, 6)
    plt.grid(axis="y")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

def main():
    pred_df, draw_df = load_data()
    if pred_df is None or draw_df is None:
        return

    results = analyze_matches(pred_df, draw_df)

    if results.empty:
        print("⚠️ No results to analyze.")
        return

    print(results.head())

    plot_match_summary(results)

if __name__ == "__main__":
    main()
