import pandas as pd

# Mapping from numeric IDs back to soil type labels
id2label = {0: "Alluvial soil", 1: "Black Soil", 2: "Clay soil", 3: "Red soil"}

def map_predictions_to_labels(preds):
    """
    Convert an array or list of numeric prediction indices to corresponding soil type labels.
   
    Args:
        preds (list or np.array): Numeric predictions from the model.
       
    Returns:
        list: Corresponding soil type string labels.
    """
    return [id2label[p] for p in preds]

def save_submission(test_df, preds, output_path="submission.csv"):
    """
    Create a submission DataFrame and save it as a CSV file.
   
    Args:
        test_df (pd.DataFrame): DataFrame containing the test data identifiers.
        preds (list or np.array): Predicted labels (numeric).
        output_path (str): Path to save the submission CSV.
    """
    print(f"Saving submission to {output_path} ...")
    submission_df = test_df.copy()
    submission_df['soil_type'] = map_predictions_to_labels(preds)
    submission_df.to_csv(output_path, index=False)
    print("Submission file saved successfully.")

def postprocessing(preds, test_df):
    """
    Complete postprocessing routine: mapping predictions and saving submission.
   
    Args:
        preds (list or np.array): Numeric model predictions.
        test_df (pd.DataFrame): Test dataframe with image IDs.
       
    Returns:
        pd.DataFrame: The submission dataframe.
    """
    labels = map_predictions_to_labels(preds)
    submission_df = test_df.copy()
    submission_df['soil_type'] = labels
    print("Postprocessing completed.")
    return submission_df

if __name__ == "__main__":
    # Example usage for testing (can be removed in actual use)
    dummy_preds = [0, 1, 2, 3, 0]
    dummy_test_df = pd.DataFrame({'image_id': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']})
    submission = postprocessing(dummy_preds, dummy_test_df)
    print(submission)
    save_submission(dummy_test_df, dummy_preds)
