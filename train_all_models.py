# train_all_models.py

import logging
from utils.trainer import train_models
from utils.synthetic_training import train_on_synthetic_data
from utils.blended_training import train_blended_model

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
        # Step 1: Train Real Models
        logger.info("🧠 Training real historical models...")
        import pandas as pd
        draw_df = pd.read_excel("data/draw_history.xlsx", parse_dates=["Draw Date"])
        draw_df.columns = draw_df.columns.astype(str).str.strip()
        train_models(draw_df, logger)

    except Exception as e:
        logger.error(f"❌ Failed during real model training: {e}", exc_info=True)

    try:
        # Step 2: Train Synthetic Models
        logger.info("🤖 Training synthetic models...")
        train_on_synthetic_data(logger)
    except Exception as e:
        logger.error(f"❌ Failed during synthetic model training: {e}", exc_info=True)

    try:
        # Step 3: Train Blended Models
        logger.info("🧬 Training blended models...")
        train_blended_model(real_weight=0.7, logger=logger)
    except Exception as e:
        logger.error(f"❌ Failed during blended model training: {e}", exc_info=True)

    logger.info("🎯 All model training completed!")

if __name__ == "__main__":
    main()
