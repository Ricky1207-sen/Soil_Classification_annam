"""

Author: Annam.ai IIT Ropar
Team Name: ARiES
Team Members: Aditya Pratap Singh Takuli, Piyush Gupta, Lay Gupta, Antik Sen, Satarupa Mishra
Leaderboard Rank: 29

"""
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

# Paths - Adjust if necessary
TRAIN_PATH = './data/train'
LABELS_CSV = './data/train_labels.csv'

# Label encoding dictionary (consistent with training/inference)
label2id = {"Alluvial soil": 0, "Black Soil": 1, "Clay soil": 2, "Red soil": 3}

def load_labels(csv_path=LABELS_CSV):
    """
    Load training labels CSV file as a DataFrame.
    """
    print(f"Loading labels from {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def get_image_path(image_id, image_dir=TRAIN_PATH):
    """
    Construct full image path from image_id and directory.
    """
    return os.path.join(image_dir, image_id)

def preprocess_image(image_path):
    """
    Load an image and apply standard preprocessing: resize and normalize.
    This can be used for quick checks or visualization.
    """
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess_transform(image)
    return image

def preprocessing():
    """
    Main preprocessing routine placeholder.
    Loads labels and prints dataset info.
    """
    print("Starting preprocessing...")
    df = load_labels()
    print(f"Total samples: {len(df)}")
    # Example: show first image path and its label
    first_img_path = get_image_path(df.iloc[0]['image_id'])
    print(f"First image path: {first_img_path}")
    print(f"First label: {df.iloc[0]['soil_type']}")
    # You could add more preprocessing logic here if needed
    return df

if __name__ == "__main__":
    preprocessing()

