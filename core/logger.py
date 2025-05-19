import os
import logging
from datetime import datetime
import pandas as pd
import csv

LOG_DIR = "logs"
PRED_LOG = "data/prediction_log.csv"
PREDICTED_SETS_LOG = "data/predicted_sets_log.csv"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

def setup_logger(debug=False):
    logger = logging.getLogger("lotto")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, f"lotto_{datetime.today().strftime('%Y%m%d')}.log")

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == fh.baseFilename for h in logger.handlers):
        logger.addHandler(fh)

    # Optional: console output for dev
    if debug:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def save_predictions(results, draw_number, draw_date):
    """Append hybrid filtered predictions to CSV log file."""
    if not results:
        return

    rows = []
    for model_name, predictions in results.items():
        for i, pset in enumerate(predictions):
            if not isinstance(pset, (list, tuple)):
                continue
            rows.append({
                "Draw Number": draw_number,
                "Draw Date": draw_date,
                "Source": model_name,
                "Prediction Set": i + 1,
                "Numbers": ", ".join(map(str, pset))
            })

    df = pd.DataFrame(rows)
    if os.path.exists(PRED_LOG):
        df.to_csv(PRED_LOG, mode='a', header=False, index=False)
    else:
        df.to_csv(PRED_LOG, index=False)

def save_predicted_sets(draw_number, predictions):
    """Save raw predicted sets into predicted_sets_log.csv for tracking."""
    os.makedirs("data", exist_ok=True)
    fieldnames = ["Draw Number", "Prediction Time", "Source", "Numbers", "Powerball"]

    rows = []
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for source, sets in predictions.items():
        for s in sets:
            if len(s) >= 7:
                rows.append({
                    "Draw Number": draw_number,
                    "Prediction Time": prediction_time,
                    "Source": source,
                    "Numbers": ", ".join(map(str, s[:6])),
                    "Powerball": s[6]
                })

    file_exists = os.path.exists(PREDICTED_SETS_LOG)

    with open(PREDICTED_SETS_LOG, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logging.info(f"📝 Saved {len(rows)} raw predicted sets to {PREDICTED_SETS_LOG}")

def compare_with_actual(draw_df):
    """Compare historical hybrid predictions against actual draw results."""
    if not os.path.exists(PRED_LOG):
        return pd.DataFrame()

    try:
        pred_log_df = pd.read_csv(PRED_LOG, parse_dates=["Draw Date"])
    except Exception:
        return pd.DataFrame()

    all_matches = []
    for _, row in pred_log_df.iterrows():
        if pd.isna(row.get("Numbers")):
            continue

        draw = draw_df[draw_df["Draw Number"] == row["Draw Number"]]
        if draw.empty:
            continue

        actual_numbers = set(draw.iloc[0][["1", "2", "3", "4", "5", "6"]].values)
        predicted = set(map(int, str(row["Numbers"]).split(", ")))
        matched = len(actual_numbers.intersection(predicted))
        row["Matched Numbers"] = matched
        all_matches.append(row)

    return pd.DataFrame(all_matches)
