# Breast Cancer Classification Dashboard

This repository contains the implementation of various machine learning classification models to predict breast cancer based on cytological features. This project is part of the Machine Learning Assignment 2.

## Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early diagnosis significantly increases the chances of survival. The goal of this project is to implement and compare multiple machine learning algorithms to classify breast tumors as either **Malignant** (1) or **Benign** (0) based on numerical features derived from digitized images of fine needle aspirate (FNA) of a breast mass.

## Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Instances:** 569
- **Features:** 30 numerical features (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension).
- **Target:** Binary (0: Benign, 1: Malignant)

The dataset meets the assignment requirements:
- Minimum Feature Size: 30 (> 12)
- Minimum Instance Size: 569 (> 500)

## Models Used & Performance Comparison

All 6 models were implemented on the same dataset. The evaluation metrics recorded are as follows:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9561 | 0.9977 | 0.9459 | 0.9859 | 0.9655 | 0.9068 |
| Decision Tree | 0.9386 | 0.9369 | 0.9571 | 0.9437 | 0.9504 | 0.8701 |
| kNN | 0.9561 | 0.9959 | 0.9342 | 1.0000 | 0.9660 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| Random Forest | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| XGBoost | 0.9561 | 0.9908 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |

## Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performed exceptionally well with high AUC, showing that the classes are linearly separable to a large extent. |
| Decision Tree | Showed slightly lower performance compared to others, likely due to overfitting or the simplistic nature of a single tree. |
| kNN | Achieved perfect Recall (1.0), which is critical in medical diagnosis to ensure no malignant cases are missed. |
| Naive Bayes | **Best Performer** in this study, achieving the highest Accuracy and MCC. Its assumption of feature independence seems to work well here. |
| Random Forest | Highly robust and provided high precision and recall, as expected from an ensemble method. |
| XGBoost | Delivered strong performance, comparable to Logistic Regression and Random Forest, with good balance across all metrics. |

## Repository Structure
- `app.py`: Streamlit application for interactive model demonstration.
- `train_models.py`: Script to train and save all 6 models.
- `requirements.txt`: List of dependencies.
- `model/`: Directory containing saved `.joblib` model files.
- `test_data.csv`: A sample test dataset for use with the Streamlit app.

## How to Run Locally
1. Create a virtual environment: `python -m venv venv`
2. Activate venv: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run models training: `python train_models.py`
5. Launch Streamlit: `streamlit run app.py`