"""

Author: Annam.ai IIT Ropar
Team Name: ARiES
Team Members: Aditya Pratap Singh Takuli, Piyush Gupta, Lay Gupta, Antik Sen, Satarupa Mishra
Leaderboard Rank: 18

"""
def preprocessing():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Paths
    TRAIN_DIR = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train"
    TRAIN_LABELS_CSV = "/kaggle/input/soil-classification-part-2/soil_competition-2025/train_labels.csv"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Load training labels (all soil images → label 1)
    df = pd.read_csv(TRAIN_LABELS_CSV)
    df["label"] = 1

    # Define augmentation for synthetic non-soil generation
    augmenter = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        brightness_range=[0.2, 0.8],
        shear_range=20,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=50.0,
        fill_mode="nearest",
        preprocessing_function=lambda x: tf.image.random_contrast(x, 0.5, 1.5)
    )

    # Validation split generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_dataframe(
        df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=IMG_SIZE,
        class_mode="raw",
        subset="training",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=IMG_SIZE,
        class_mode="raw",
        subset="validation",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("✅ Preprocessing complete. Data generators are ready.")
    return train_gen, val_gen
