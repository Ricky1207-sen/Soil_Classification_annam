# Soil_Classification_annam


# Soil Image Classification Challenge – Solution README

## Overview

This repository contains my submission for the **Soil Image Classification Challenge** organized by **Annam.ai at IIT Ropar**. The task involves building a machine learning model to classify soil images into one of four categories:

- Alluvial Soil  
- Black Soil  
- Clay Soil  
- Red Soil  

The competition aims to promote balanced classification performance by maximizing the **minimum F1-score** across all classes.

---

## Model Used

I used the **ConvNeXt-Tiny** model, a modern convolutional architecture that has shown state-of-the-art performance on image classification tasks. It combines hierarchical convolutional structures with design insights from transformers, offering both accuracy and efficiency.

---

## Workflow Summary

### 1. **Data Preprocessing**

- **Resizing**: All images were resized to `224x224` pixels to match the input requirements of ConvNeXt-Tiny.
- **Normalization**: Pixel values were normalized using ImageNet statistics.
- **Augmentation** (during training):
  - Random Horizontal Flip
  - Random Rotation
  - Color Jitter
  - Random Crop with Padding
- These augmentations help improve generalization, especially for soil types with subtle differences.

### 2. **Model Architecture**

- **Base Model**: `ConvNeXt-Tiny` from `timm` library, pretrained on ImageNet.
- **Modifications**:
  - Replaced the classifier head with a fully connected layer for 4 output classes.
  - Applied dropout before the final classifier for regularization.

### 3. **Training Setup**

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW` with weight decay
- **Scheduler**: Cosine Annealing with warm restarts
- **Batch Size**: 32  
- **Epochs**: 30 (with early stopping based on validation F1-min score)

### 4. **Evaluation Strategy**

- The model's performance was evaluated using the **minimum F1-score** across all classes on the validation set.
- Validation set was stratified to maintain class distribution.

---

## Results

On the validation set:

- **Minimum F1-score**: 1.000
- Confusion matrix and per-class F1-scores are plotted in the notebook to visualize performance across soil types.

---

## File Structure

- `convinexttiny.ipynb`: Main notebook containing all code for:
  - Data loading and preprocessing
  - Model architecture
  - Training and evaluation
  - Prediction generation

- `submission.csv`: Final prediction file in the required format.

---

## Reproducibility

To reproduce the results:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebook:
   ```bash
   jupyter notebook convinexttiny.ipynb
   ```

3. Ensure the dataset is placed in the correct directory as referenced in the notebook.

---

## Conclusion

This solution leverages modern convolutional architectures to effectively classify soil images. Emphasis was placed on:

- Balanced performance across all soil types
- Proper data augmentation and preprocessing
- Clear code structure and modular implementation

Please refer to the code comments for additional insights on decisions and tuning strategies.



THE SECOND CHALLENGE:


#  Soil Image Classification Challenge

##  Overview

The **Soil Image Classification Challenge** is a machine learning competition organized by **Annam.ai at IIT Ropar**. The objective is to build a classifier that determines whether an input image contains **soil** or **non-soil** content using visual features.

**Deadline**: May 25, 2025, 11:59 PM IST  
**Metric**: F1 Score (binary classification)

---

##  Dataset Description

###  Objective
Predict whether an image is a soil image (`label = 1`) or not (`label = 0`).

###  Structure

- **Train Set**  
  - Images located in `train/`  
  - CSV with columns: `image_id`, `soil_type`  
  - All soil images → labeled as `1`

- **Test Set**  
  - Only `image_id`s are provided  
  - Participants must submit predicted `label`s

- **Soil Types in Training**:  
  - Alluvial Soil  
  - Black Soil  
  - Clay Soil  
  - Red Soil

Note: For this binary classification, all above types are treated equally as “soil”.

---

##  Model Architecture

This solution uses **MobileNetV2**, a lightweight convolutional neural network pretrained on ImageNet.

### Architecture

- **Base**: MobileNetV2 (frozen layers)
- **Custom Head**:
  - `GlobalAveragePooling2D`
  - `Dense(64, relu)`
  - `Dense(1, sigmoid)` — for binary classification

### Training Strategy

1. **Data Preprocessing**:
   - Images resized to 224×224
   - Normalized to [0, 1]
   - Augmented using `ImageDataGenerator`

2. **Synthetic Negative Samples**:
   - Augmented training images used to create visually diverse fake “non-soil” examples
   - Labelled as `0`

3. **Training Phases**:
   - **Phase 1**: Train head layers with base model frozen
   - **Phase 2**: Fine-tune top 30 layers of MobileNetV2 with a reduced learning rate

4. **Loss Function**: Binary Crossentropy  
5. **Optimizer**: Adam  
6. **Callbacks**: EarlyStopping (monitored `val_loss`)

---

##  Evaluation Metric

- **F1 Score**: Balances Precision and Recall
- Encourages strong, **balanced classification performance**
- Final evaluation includes:
  - F1-score on private test set
  - Review of submitted code and documentation

---

##  Submission Format

Submit a `submission.csv` with columns:

```
image_id,label
IMG_001.jpg,1
IMG_002.jpg,0
...
```

Also include your:
- Final notebook/script with:
  - Preprocessing steps
  - Model training
  - Evaluation
  - Prediction generation
- Code must be clean, readable, and well-commented

---

##  Reproducibility

To reproduce the results:

```bash
pip install -r requirements.txt
python preprocess.py    # optional, if separate
python train.py
python predict.py
```

Model and prediction logic is implemented in [`mobilenetupdated.ipynb`](./mobilenetupdated.ipynb)

---

##  Results

- Achieved F1 Score = 1.000
- Efficient training due to lightweight MobileNet backbone
- Robust handling of imbalance via synthetic negatives

---

##  Tools & Libraries

- Python, NumPy, Pandas, Matplotlib
- TensorFlow / Keras
- Scikit-learn (F1-score)
- ImageDataGenerator for augmentation

---

##  Notes

- Avoid training on test data
- Ensure predictions are reproducible
- Submit early to avoid last-minute issues

---

##  Acknowledgements

Organized by **Annam.ai** in collaboration with **IIT Ropar**. This challenge fosters practical application of AI in agriculture and environmental sciences.
