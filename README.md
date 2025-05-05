# Lotto Predictor Pro

An advanced lottery draw prediction system using machine learning. It predicts the next possible number combinations based on historical data and continuously evolves with each new draw.

## 🧠 Features

- Position-specific ML models (Random Forest, Gradient Boosting)
- Separate model for PowerBall prediction
- Combined weighted model for better accuracy
- Hot/Cold/Frequency visual analysis
- Exclusion-aware prediction
- Accuracy tracking across runs
- Streamlit UI dashboard
- Prediction logging and model persistence
- (Optional) Distributed execution-ready

## 🗂️ Project Structure
```
lotto-predictor/
├── data/
│   ├── draw_history.xlsx         # Historical draw data
│   ├── active_tickets.xlsx       # Current active tickets
│   └── predictions_log.csv       # Logged past predictions
├── models/                       # Saved ML models (via joblib)
├── utils/
│   ├── analysis.py               # Hot/cold/frequency & accuracy logic
│   ├── logger.py                 # Save/compare prediction results
│   ├── predictor.py              # Prediction logic for numbers/PowerBall
│   ├── trainer.py                # ML model training & persistence
├── main.py                       # Streamlit app interface
├── requirements.txt              # Python package dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Clone and Install
```bash
git clone https://github.com/yourname/lotto-predictor.git
cd lotto-predictor
pip install -r requirements.txt
```

### 2. Prepare Data
- Fill `data/draw_history.xlsx` with past draw data (must include headers: Draw Date, Draw Number, 1-6, Power Ball)
- Add `data/active_tickets.xlsx` for current tickets (optional)

### 3. Run the App
```bash
streamlit run main.py
```

### 4. Train and Predict
- Use "Train Models" button to build or rebuild models
- Get top 10 predictions (Top 5 highlighted 🔥)
- View historical accuracy

## 🧩 Advanced Usage
- Models are saved in `models/` and reused automatically
- Predictions are logged in `data/predictions_log.csv`
- Accuracy auto-updates when new draw is added to `draw_history.xlsx`

## 💻 Distributed Execution (Optional)
Use joblib/dask integration to split workloads between multiple machines (setup guide coming soon).

## 📦 Requirements
See [requirements.txt](requirements.txt) for full package list:
```
pandas
scikit-learn
matplotlib
seaborn
openpyxl
joblib
streamlit
```

## 📬 Contributing / Ideas
PRs and ideas are welcome! Open an issue or fork away.

