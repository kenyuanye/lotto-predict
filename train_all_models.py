import logging
import pandas as pd

from train_models import train_models
from trainers.symbolic_trainer import train_symbolic_models
from trainers.synthetic_training import train_on_synthetic_data
from trainers.blended_training import train_blended_model
from trainers.train_ml_models import train_ml_models  # ✅ NEW

def setup_basic_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("trainer")

def main():
    logger = setup_basic_logger()
    logger.info("🚀 Starting full training sequence...")

    try:
        # Step 1: Load and prepare draw history
        logger.info("📥 Loading draw history...")
        draw_df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
        draw_df.columns = draw_df.columns.astype(str).str.strip()
        draw_df = draw_df.sort_values("Draw Number").reset_index(drop=True)
        draw_df["DrawIndex"] = draw_df.index

    except Exception as e:
        logger.error("❌ Failed to load draw history!", exc_info=True)
        return

    try:
        # Step 2: Train ML Models (position-wise)
        logger.info("🧠 Training ML models...")
        train_ml_models()

    except Exception as e:
        logger.error("❌ Failed during ML model training!", exc_info=True)

    try:
        # Step 3: Train Symbolic Models
        logger.info("📐 Training symbolic models...")
        train_symbolic_models(draw_df)

    except Exception as e:
        logger.error("❌ Failed during symbolic model training!", exc_info=True)

    try:
        # Step 4: Train Synthetic Models
        logger.info("🤖 Training synthetic models...")
        train_on_synthetic_data(logger)

    except Exception as e:
        logger.error("❌ Failed during synthetic model training!", exc_info=True)

    try:
        # Step 5: Train Blended Models
        logger.info("🧬 Training blended models...")
        train_blended_model(real_weight=0.7, logger=logger)

    except Exception as e:
        logger.error("❌ Failed during blended model training!", exc_info=True)

    logger.info("🎯 All model training completed successfully!")

if __name__ == "__main__":
    main()
