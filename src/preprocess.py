import os
import re
import glob
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

# === NLTK Setup ===
def ensure_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

ensure_nltk_resources()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# === Data Loading ===
def load_parquet_files(folder_path="data/raw"):
    all_files = glob.glob(os.path.join(folder_path, '*.parquet'))
    if not all_files:
        logger.error(f"No .parquet files found in {folder_path}")
        return pd.DataFrame()

    logger.info(f"Found {len(all_files)} parquet files")

    dataframes = []
    for f in all_files:
        try:
            df = pd.read_parquet(f)
            dataframes.append(df)
            logger.info(f"Loaded {f} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {f}: {e}")
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

# === Data Cleaning ===
def clean_text(text: str) -> str:
    try:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
        text = re.sub(r"[^a-z\s]", '', text)  # Keep letters only
        text = re.sub(r"\s+", ' ', text).strip()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        return ""

# === Processing ===
def preprocess_dataframe(df: pd.DataFrame):
    try:
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        logger.info(f"Removed {initial_len - len(df)} duplicate rows. Remaining: {len(df)}")

        df = df[['content', 'category']].dropna()
        logger.info(f"Selected 'content' and 'category'. After dropna: {len(df)} rows")

        return df
    except Exception as e:
        logger.error(f"Error in preprocessing dataframe: {e}")
        return pd.DataFrame()

# === Train/Test Split ===
def split_data(df: pd.DataFrame, test_size=0.2):
    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['category'], random_state=42
        )
        logger.info(f"Train/Test split complete: Train={len(train_df)}, Test={len(test_df)}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error during train/test split: {e}")
        return pd.DataFrame(), pd.DataFrame()

# === Apply Cleaning ===
def apply_cleaning(df: pd.DataFrame):
    try:
        df['content'] = df['content'].apply(clean_text)
        return df
    except Exception as e:
        logger.error(f"Failed to apply cleaning: {e}")
        return df

# === Save Output ===
def save_to_parquet(df: pd.DataFrame, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved file to {path}")
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")

# === Main Script ===
if __name__ == "__main__":
    df = load_parquet_files("data/raw")

    if df.empty:
        logger.error("No data loaded from raw files.")
        exit()

    df = preprocess_dataframe(df)

    if df.empty:
        logger.error("Dataframe is empty after preprocessing.")
        exit()

    train_df, test_df = split_data(df, test_size=0.3)

    if train_df.empty or test_df.empty:
        logger.error("Train/test data split failed.")
        exit()

    logger.info("Starting text cleaning on train and test datasets...")

    train_df = apply_cleaning(train_df)
    test_df = apply_cleaning(test_df)

    save_to_parquet(train_df, "data/interim/train.parquet")
    save_to_parquet(test_df, "data/interim/test.parquet")
