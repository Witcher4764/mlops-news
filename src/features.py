import os
import pandas as pd
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.exceptions import NotFittedError
from typing import Tuple, Union

# === Logger Setup ===
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

# ---------- LOAD DATA ----------
def load_data(interim_dir: str = "data/interim") -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_path = os.path.join(interim_dir, "train.parquet")
        test_path = os.path.join(interim_dir, "test.parquet")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded train ({len(train_df)}) and test ({len(test_df)}) records from interim.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Failed to load data from interim/: {e}")
        raise


# ---------- VECTORIZE ----------
def vectorize_text(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   method: str = "tfidf",
                   max_features: int = 5000,
                   ngram_range: Tuple[int, int] = (1, 1),
                   output_dir: str = "models") -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if method == "tfidf":
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        elif method == "count":
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:
            raise ValueError(f"Unsupported vectorizer method: {method}")

        logger.info(f"Applying {method.upper()} vectorizer with max_features={max_features} and ngram_range={ngram_range}")

        X_train = vectorizer.fit_transform(train_df["content"])
        X_test = vectorizer.transform(test_df["content"])

        train_vectorized = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
        test_vectorized = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

        # Add category labels back with index alignment
        train_vectorized["target_category"] = train_df["category"].reset_index(drop=True)
        test_vectorized["target_category"] = test_df["category"].reset_index(drop=True)


        # Save vectorizer
        os.makedirs(output_dir, exist_ok=True)
        vec_path = os.path.join(output_dir, "vectorizer.joblib")
        joblib.dump(vectorizer, vec_path)
        logger.info(f"Saved vectorizer to {vec_path}")

        logger.info(f"Vectorization complete. Train shape: {train_vectorized.shape}, Test shape: {test_vectorized.shape}")
        return train_vectorized, test_vectorized

    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        raise


# ---------- SAVE FINAL DATA ----------
def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/processed"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_parquet(os.path.join(output_dir, "train_final.parquet"), index=False)
        test_df.to_parquet(os.path.join(output_dir, "test_final.parquet"), index=False)
        logger.info("Saved processed train and test data to data/processed/")
    except Exception as e:
        logger.error(f"Saving processed data failed: {e}")
        raise


# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        train_df, test_df = load_data()
        train_vec, test_vec = vectorize_text(train_df, test_df, method="tfidf", max_features=5000, ngram_range=(1, 2))
        save_processed(train_vec, test_vec)
    except Exception as main_e:
        logger.critical(f"Pipeline failed: {main_e}")
