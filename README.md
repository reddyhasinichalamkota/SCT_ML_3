# SCT_ML_3
# Cats vs Dogs Image Classification using Support Vector Machine (SVM)

This project implements a **Support Vector Machine (SVM)** model to classify images of **cats** and **dogs** using the Kaggle Dogs vs Cats dataset. The model uses **Histogram of Oriented Gradients (HOG)** for feature extraction and achieves reliable classification performance.

## Project Overview

Image classification is an important application of Machine Learning and Computer Vision. In this project:

- Images are preprocessed and resized
- HOG features are extracted from each image
- SVM classifier is trained using training data
- Model is tested on unseen images
- Performance is evaluated using multiple metrics

## Dataset

Dataset Used: **Dogs vs Cats Dataset (Kaggle)**

Classes:

- Cat
- Dog

Folder Structure:

```bash
training_set/
   cats/
   dogs/

test_set/
   cats/
   dogs/
