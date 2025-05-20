# legacy_prediction/walkforward_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from core.feature_engineering import build_features_for_prediction
from core import NUMBER_COLUMNS, POWERBALL_COLUMN

def predict_with_walkforward(draw_history_df, exclusions=None):
    """
    Predict the next draw using a walkforward-style approach:
    - Train on full feature set
    - Predict 6 main numbers and Powerball
    Returns:
        [n1, n2, n3, n4, n5, n6, Powerball]
    """
    df = draw_history_df.copy().reset_index(drop=True)
    df["DrawIndex"] = df.index
    df["Draw Number"] = df.index  # required for features

    # Build training feature matrix
    X_train = build_features_for_prediction(df)
    Y_main = df[NUMBER_COLUMNS]
    Y_pb = df[POWERBALL_COLUMN]

    # Predicting the next index
    next_index = len(df)
    next_input = pd.DataFrame({"DrawIndex": [next_index], "Draw Number": [next_index]})
    X_pred = build_features_for_prediction(next_input)

    # Align prediction columns with training columns
    feature_columns = X_train.columns
    X_pred = X_pred[feature_columns]

    # Train models
    main_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    main_model.fit(X_train, Y_main)

    pb_model = RandomForestClassifier(n_estimators=200, random_state=42)
    pb_model.fit(X_train, Y_pb)

    # Predict
    pred_main = main_model.predict(X_pred)[0]
    main_numbers = list(np.clip(np.round(pred_main).astype(int), 1, 40))

    pb_probs = pb_model.predict_proba(X_pred)[0]
    powerball = int(np.argmax(pb_probs) + 1)

    return main_numbers[:6] + [powerball]
