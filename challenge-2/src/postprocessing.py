"""

Author: Annam.ai IIT Ropar
Team Name: ARiES
Team Members: Aditya Pratap Singh Takuli, Piyush Gupta, Lay Gupta, Antik Sen, Satarupa Mishra
Leaderboard Rank: 18

"""
def postprocessing():
    import pandas as pd
    import numpy as np
    import os
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import load_model

    # Paths
    TEST_DIR = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test"
    TEST_IDS_CSV = "/kaggle/input/soil-classification-part-2/soil_competition-2025/test_ids.csv"
    MODEL_PATH = "/kaggle/working/final_model.h5"  # Assuming model saved previously
    IMG_SIZE = (224, 224)

    # Load test data
    test_ids = pd.read_csv(TEST_IDS_CSV)
    test_images = []
    for img_id in test_ids["image_id"]:
        img_path = os.path.join(TEST_DIR, img_id)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        test_images.append(img_array)
    test_images = np.array(test_images)

    # Load trained model
    model = load_model(MODEL_PATH)

    # Predict
    preds = model.predict(test_images)
    test_ids["label"] = (preds > 0.5).astype(int)

    # Save submission
    submission = test_ids[["image_id", "label"]]
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission saved as submission.csv")

    return 0
