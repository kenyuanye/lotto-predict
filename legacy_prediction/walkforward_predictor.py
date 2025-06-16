# legacy_prediction/walkforward_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from utils.feature_engineering import build_features_for_prediction
from utils import NUMBER_COLUMNS, POWERBALL_COLUMN

def predict_with_walkforward(draw_history_df, exclusions=None):
    """
    Predict the next draw using a walkforward-style approach:
    - Train on full feature set
    - Predict 6 main numbers and Powerball.
    Returns:
        [n1, n2, n3, n4, n5, n6, Powerball] or [] on error
    """
    try:
        df = draw_history_df.copy().reset_index(drop=True)
        df["DrawIndex"] = df.index
        df["Draw Number"] = df.index

        # Build training feature matrix
        X_train = build_features_for_prediction(df)
        Y_main = df[NUMBER_COLUMNS]
        Y_pb = df[POWERBALL_COLUMN]

        # Prepare input for the next draw
        next_index = len(df)
        next_input = df.iloc[[-1]].copy()
        next_input["DrawIndex"] = next_index
        next_input["Draw Number"] = next_index

        X_pred = build_features_for_prediction(next_input)
        X_pred = X_pred[X_train.columns]  # Align columns

        # Train main and Powerball models
        main_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
        main_model.fit(X_train, Y_main)

        pb_model = RandomForestClassifier(n_estimators=200, random_state=42)
        pb_model.fit(X_train, Y_pb)

        # Predict main numbers
        pred_main = main_model.predict(X_pred)[0]
        pred_main = np.round(pred_main).astype(int)
        pred_main = np.clip(pred_main, 1, 40)

        # Ensure exactly 6 unique main numbers
        main_unique = sorted(set(pred_main))
        while len(main_unique) < 6:
            filler = np.random.randint(1, 41)
            if filler not in main_unique:
                main_unique.append(filler)
        final_main = sorted(main_unique[:6])

        # Predict Powerball
        pb_probs = pb_model.predict_proba(X_pred)[0]
        powerball = int(np.argmax(pb_probs) + 1)
        powerball = max(1, min(powerball, 10))  # Clamp Powerball

        final_set = final_main + [powerball]

        if len(final_set) != 7:
            print("[walkforward] Invalid prediction generated (bad length).")
            return []

        return final_set

    except Exception as e:
        print(f"[walkforward] Failed to predict: {e}")
        return []
