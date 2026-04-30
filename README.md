# SCT_ML_3
# Cats vs Dogs Image Classification using Support Vector Machine (SVM)

This project implements a **Support Vector Machine (SVM)** model to classify images of **cats** and **dogs** using the Kaggle Dogs vs Cats dataset. The model uses **Histogram of Oriented Gradients (HOG)** for feature extraction and achieves reliable classification performance.

---

## Project Overview

Image classification is an important application of Machine Learning and Computer Vision. In this project:

- Images are loaded from dataset folders
- Images are resized and preprocessed
- HOG features are extracted from each image
- SVM classifier is trained using training data
- Model is tested on unseen images
- Performance is evaluated using multiple metrics
- Trained model is saved for future predictions

---

## Dataset

Dataset Used: **Dogs vs Cats Dataset (Kaggle)**

### Classes

- Cat
- Dog

### Folder Structure

```bash
Animal/
│── training_set/
│   ├── cats/
│   └── dogs/
│
│── test_set/
│   ├── cats/
│   └── dogs/
│
│── animal.py
│── README.md
│── svm_cat_dog.joblib
│── hog_visualization.png
│── confusion_matrix.png
│── precision_recall_chart.png
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Matplotlib
- Joblib
- TQDM

---

## Machine Learning Workflow

1. Load training and testing images  
2. Resize images to fixed dimensions  
3. Convert images to grayscale  
4. Extract HOG features  
5. Normalize features using StandardScaler  
6. Train SVM model with RBF Kernel  
7. Perform 5-Fold Cross Validation  
8. Predict on test dataset  
9. Evaluate results  
10. Save trained model  

---

## Model Performance

### Dataset Statistics

- Training Images: **4000**
  - 2000 Cats
  - 2000 Dogs

- Testing Images: **2023**
  - 1011 Cats
  - 1012 Dogs

### HOG Feature Vector Size

- **1764 Features**

### Cross Validation Accuracy

- **74.20% ± 1.48%**

### Test Accuracy

- **73.01%**

### Classification Report

```text
              precision    recall    f1-score   support

Cat              0.72       0.74       0.73      1011
Dog              0.74       0.72       0.73      1012

accuracy                               0.73      2023
macro avg         0.73       0.73       0.73     2023
weighted avg      0.73       0.73       0.73     2023
```

---

## Generated Output Files

- `svm_cat_dog.joblib` → Saved trained model  
- `hog_visualization.png` → HOG feature visualization  
- `confusion_matrix.png` → Confusion matrix heatmap  
- `precision_recall_chart.png` → Precision vs Recall graph  

---

## Installation

Install required libraries using pip:

```bash
pip install numpy opencv-python scikit-learn scikit-image matplotlib joblib tqdm
```

---

## How to Run

```bash
python animal.py
```

---

## Sample Output

```text
   SVM Cat vs Dog Classifier

[INFO] Loaded 4000 training images
[INFO] Loaded 2023 testing images

[INFO] Extracting HOG features...
[INFO] Training SVM model...

CV Accuracy: 74.20% ± 1.48%
Test Accuracy: 73.01%

[DONE] All steps complete!
```

---

## Why Support Vector Machine?

Support Vector Machine is effective for binary classification tasks like Cats vs Dogs because:

- Strong performance on medium datasets
- Works well with HOG features
- Good generalization ability
- Efficient memory usage
- Fast prediction after training

---
