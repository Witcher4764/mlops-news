import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

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
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()


# --------- LOAD DATA ---------
def load_processed_data(processed_dir="data/processed"):
    try:
        train_df = pd.read_parquet(os.path.join(processed_dir, "train_final.parquet"))
        test_df = pd.read_parquet(os.path.join(processed_dir, "test_final.parquet"))
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        raise


# --------- LOAD MODEL AND ENCODER ---------
def load_model_and_encoder(model_dir="models"):
    try:
        model = load(os.path.join(model_dir, "lightgbm_model.joblib"))
        le = load(os.path.join(model_dir, "label_encoder.joblib"))
        logger.info("Loaded model and label encoder successfully.")
        return model, le
    except Exception as e:
        logger.error(f"Failed to load model or encoder: {e}")
        raise


# ---------- GENERATE REPORT ----------
def generate_report(y_true, y_pred, labels, report_path):
    try:
        report = classification_report(y_true, y_pred, target_names=labels)
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved classification report to {report_path}")
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        raise


# ---------- PLOT CONFUSION MATRIX ----------
def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    try:
        cm = confusion_matrix(y_true, y_pred, normalize='true') * 100  # Normalize to % scale
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Normalized Confusion Matrix - Test Set")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved normalized confusion matrix to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot normalized confusion matrix: {e}")
        raise


# ---------- MAIN ----------
if __name__ == "__main__":
    try:
        os.makedirs("Reports", exist_ok=True)

        train_df, test_df = load_processed_data()
        model, le = load_model_and_encoder()

        # Split features and labels
        X_train, y_train = train_df.drop(columns=["target_category"]), train_df["target_category"]
        X_test, y_test = test_df.drop(columns=["target_category"]), test_df["target_category"]

        # Encode labels
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        labels = list(le.classes_)

        # Save classification reports
        generate_report(y_train_enc, y_train_pred, labels, "Reports/train_classification_report.txt")
        generate_report(y_test_enc, y_test_pred, labels, "Reports/test_classification_report.txt")

        # Save confusion matrix for test only
        plot_confusion_matrix(y_test_enc, y_test_pred, labels, "Reports/test_confusion_matrix.png")

    except Exception as e:
        logger.critical(f"Evaluation pipeline failed: {e}")
