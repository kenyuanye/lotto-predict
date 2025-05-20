import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from core.feature_engineering import build_features_for_prediction
from data_loader import load_draw_history

MODEL_DIR = "models/ml"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_ml_models():
    print("🔄 Loading draw history...")
    draw_df = load_draw_history()
    features_df = build_features_for_prediction(draw_df)

    print("🧪 Preparing targets...")
    # Targets for main numbers (positions 1–6) and Powerball
    number_targets = {f"pos{i}": draw_df[str(i)] for i in range(1, 7)}
    powerball_target = draw_df["Power Ball"]

    # Train models for main numbers
    for i in range(1, 7):
        print(f"🎯 Training model for position {i}")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features_df, number_targets[f"pos{i}"])
        joblib.dump(model, os.path.join(MODEL_DIR, f"rf_model_pos{i}.pkl"))

    # Train model for Powerball
    print("✨ Training Powerball model")
    pb_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pb_model.fit(features_df, powerball_target)
    joblib.dump(pb_model, os.path.join(MODEL_DIR, "rf_model_powerball.pkl"))

    print(f"✅ Models saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_ml_models()
