import os
import pandas as pd
import logging
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# ---------- LOGGER SETUP ----------
def setup_logger(name="only_my_logs", log_dir="logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()


# ---------- LOAD TRAIN DATA ONLY ----------
def load_train_data(processed_dir: str = "data/processed"):
    try:
        train_path = os.path.join(processed_dir, "train_final.parquet")
        train_df = pd.read_parquet(train_path)
        logger.info(f"Loaded train dataset with shape {train_df.shape}")
        return train_df
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise


# ---------- PREPARE FEATURES ----------
def prepare_features_and_labels(df: pd.DataFrame):
    try:
        X = df.drop(columns=["target_category"])
        y = df["target_category"]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        logger.info(f"Label encoding completed. Classes: {list(le.classes_)}")
        return X, y_encoded, le
    except Exception as e:
        logger.error(f"Failed to prepare features/labels: {e}")
        raise


# ---------- TRAIN MODEL ----------
def train_model(X, y):
    try:
        model = lgb.LGBMClassifier(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            n_jobs=-1  # Use all CPU cores
        )
        model.fit(X, y)
        logger.info("Model training completed.")
        return model
    except Exception as e:
        logger.error(f"Failed to train LightGBM: {e}")
        raise


# ---------- SAVE MODEL ----------
def save_model(model, label_encoder, model_dir="models"):
    try:
        os.makedirs(model_dir, exist_ok=True)
        dump(model, os.path.join(model_dir, "lightgbm_model.joblib"))
        dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
        logger.info("Model and label encoder saved to models/")
    except Exception as e:
        logger.error(f"Saving model failed: {e}")
        raise


# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        train_df = load_train_data()
        X_train, y_train, le = prepare_features_and_labels(train_df)
        model = train_model(X_train, y_train)
        save_model(model, le)
    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")