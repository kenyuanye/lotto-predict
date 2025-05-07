import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from utils.feature_engineering import build_features_for_prediction


def predict_with_walkforward(draw_history_df, exclusions=None):
    """
    Predict the next draw using a walkforward-style approach:
    - Train on DrawIndex
    - Predict 6 main numbers and Powerball
    """
    df = draw_history_df.copy().sort_values("Draw Number").reset_index(drop=True)
    df["DrawIndex"] = df.index

    # Build training features
    X = df[["DrawIndex"]]
    Y_main = df[["1", "2", "3", "4", "5", "6"]]
    Y_pb = df["Power Ball"]

    # Prepare input for next draw
    next_index = len(df)
    X_pred = pd.DataFrame({"DrawIndex": [next_index]})

    # Train models
    main_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    main_model.fit(X, Y_main)

    pb_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pb_model.fit(X, Y_pb)

    # Predict
    main_pred = main_model.predict(X_pred)[0]
    pb_pred = pb_model.predict_proba(X_pred)

    main_numbers = list(np.clip(np.round(main_pred).astype(int), 1, 40))
    powerball = int(np.argmax(pb_pred) + 1)

    return main_numbers[:6] + [powerball]
