#!/bin/bash

# Dataset slug
KAGGLE_DATASET="soil-classification-part-2/data"
TARGET_DIR="./data"

echo "Downloading dataset: $KAGGLE_DATASET"
mkdir -p "$TARGET_DIR"
kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

echo "Download complete. Files saved to $TARGET_DIR"

