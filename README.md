# ml-models-scratch

# SMAI25 Machine Learning Assignments

This repository contains three machine learning projects completed as part of the **Smart Machines and Artificial Intelligence (SMAI)** coursework. Each project involves implementing and analyzing a different machine learning model on a real-world dataset.

---

## Contents

1. [Food Delivery Time Prediction (Linear Regression)](#1-food-delivery-time-prediction-linear-regression)
2. [KNN and ANN Search (CIFAR-10 CLIP Embeddings)](#2-knn-and-ann-search-cifar-10-clip-embeddings)
3. [Cryptocurrency Fraud Detection (Decision Tree)](#3-cryptocurrency-fraud-detection-decision-tree)
4. [SLIC Segmentation (K Means) ](#4-slic-segmentation-kmeans)
---

## 📖 Assignment Descriptions

### 1️⃣ Food Delivery Time Prediction (Linear Regression)
**Objective:**  
Predict the food delivery time in minutes based on input features such as distance, number of items, delivery area, and time of day.

**Approach:**  
- Implemented **Multiple Linear Regression** using `sklearn`.
- Applied **Ridge** and **Lasso Regularization** to improve generalization.
- Evaluated model performance using **R² Score**, **MAE**, and **RMSE**.

---

### 2️⃣ KNN and ANN Search (CIFAR-10 CLIP Embeddings)
**Objective:**  
Perform nearest neighbor search using K-Nearest Neighbors (KNN) and Approximate Nearest Neighbor (ANN) techniques on image embeddings.

**Approach:**  
- Generated image embeddings using **CLIP** for the CIFAR-10 dataset.
- Implemented **KNN Search** using `scikit-learn`.
- Performed **Approximate Nearest Neighbor (ANN) Search** using **FAISS**.
- Compared retrieval accuracy and query time for both methods.

---

### 3️⃣ Cryptocurrency Fraud Detection (Decision Tree)
**Objective:**  
Detect fraudulent cryptocurrency transactions using a Decision Tree classifier.

**Approach:**  
- Cleaned and preprocessed the provided crypto transaction dataset.
- Built a **Decision Tree Classifier** using `sklearn`.
- Tuned hyperparameters such as `max_depth` and `criterion`.
- Evaluated model accuracy, precision, recall, and confusion matrix.

---

## 4️⃣ SLIC Segmentation (K Means)

- Implemented SLIC superpixel segmentation from scratch for images with hyperparameter tuning and iterative visualization.
- Applied SLIC segmentation frame-by-frame on a video and reconstructed the segmented video.
- Optimized the video segmentation by leveraging temporal continuity to reduce iterations and improve processing speed.
