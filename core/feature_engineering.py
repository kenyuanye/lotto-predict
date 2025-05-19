import pandas as pd
import numpy as np

NUMBER_COLUMNS = ["1", "2", "3", "4", "5", "6"]

def get_feature_dataframe(draw_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = draw_df.copy()
    df = df.sort_values("Draw Number").reset_index(drop=True)

    features = []

    for i in range(len(df)):
        row = df.iloc[i]
        numbers = sorted([int(row[col]) for col in NUMBER_COLUMNS])
        past_df = df.iloc[:i]

        # Sum
        sum_val = sum(numbers)
        # Odd/Even counts
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = 6 - odd_count
        # Range
        range_val = max(numbers) - min(numbers)
        # Consecutive pairs
        consec_count = sum(1 for x, y in zip(numbers, numbers[1:]) if y - x == 1)
        # High/Low ratio (>20)
        high_count = sum(1 for n in numbers if n > 20)
        low_count = 6 - high_count
        # Repeats from previous draw
        if i > 0:
            prev_numbers = set(int(df.iloc[i-1][col]) for col in NUMBER_COLUMNS)
            repeats = len(set(numbers).intersection(prev_numbers))
        else:
            repeats = 0

        # Frequency in last N draws
        recent_numbers = (
            pd.Series(past_df[NUMBER_COLUMNS].values.ravel())
            .value_counts()
            .to_dict()
            if not past_df.empty else {}
        )
        hotness = sum(recent_numbers.get(n, 0) for n in numbers)

        # Time since last seen for each number
        last_seen = []
        if not past_df.empty:
            for n in numbers:
                seen_idx = past_df[NUMBER_COLUMNS].apply(lambda row: n in row.values, axis=1)
                if seen_idx.any():
                    last_index = seen_idx[::-1].idxmax()
                    last_seen.append(i - last_index)
                else:
                    last_seen.append(window + 1)
        else:
            last_seen = [window + 1] * 6

        avg_last_seen = np.mean(last_seen)

        features.append({
            "Sum": sum_val,
            "OddCount": odd_count,
            "EvenCount": even_count,
            "ConsecCount": consec_count,
            "Range": range_val,
            "HighCount": high_count,
            "LowCount": low_count,
            "RepeatsFromLast": repeats,
            "HotnessScore": hotness,
            "LastSeenGapAvg": avg_last_seen,
            "DrawIndex": i  # ensure included explicitly
        })

    return pd.DataFrame(features)

def build_features_for_prediction(draw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper that builds features for prediction.
    Ensures the standardized 11-column schema is returned.
    """
    features_df = get_feature_dataframe(draw_df)

    # Reorder to match expected feature order (ML + symbolic)
    EXPECTED_FEATURES = [
        "Sum", "OddCount", "EvenCount", "Range", "ConsecCount",
        "HighCount", "LowCount", "RepeatsFromLast",
        "HotnessScore", "LastSeenGapAvg", "DrawIndex"
    ]

    return features_df[EXPECTED_FEATURES]
