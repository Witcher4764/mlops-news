import boto3
import os
import logging
from botocore.exceptions import BotoCoreError, ClientError


def setup_logger(name="only_my_logs", log_dir="logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()


def download_parquet_files(bucket_name: str, prefix: str):
    """
    Downloads all .parquet files from a specified S3 bucket and prefix.
    Saves files to 'data/raw/'.

    Args:
        bucket_name (str): The S3 bucket name.
        prefix (str): The folder/prefix in the bucket. "" for root.
    """
    s3 = boto3.client('s3')
    local_dir = 'data/raw/'
    os.makedirs(local_dir, exist_ok=True)

    prefix = prefix.strip('/')
    s3_path = f"{bucket_name}/" + (f"{prefix}/" if prefix else "")

    logger.debug(f"Starting download from s3://{s3_path}")

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        file_keys = []

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.parquet'):
                    file_keys.append(key)

        total_files = len(file_keys)
        if total_files == 0:
            logger.debug(f"No .parquet files found under prefix '{prefix}' in bucket '{bucket_name}'.")
            return

        logger.debug(f"Found {total_files} .parquet file(s). Beginning download...")

        for key in file_keys:
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            s3.download_file(bucket_name, key, local_path)

        logger.debug(f"Completed download of {total_files} file(s) from s3://{s3_path}")

    except (BotoCoreError, ClientError) as e:
        logger.error(f"AWS error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    # Download from root
    download_parquet_files(bucket_name="mltesting", prefix="")

    # Download from specific prefix
    # download_parquet_files(bucket_name="mltesting", prefix="news")
