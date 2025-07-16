# 💳 Credit Card Fraud Detection Using Machine Learning

## 📌 Overview
This project aims to detect fraudulent credit card transactions using machine learning algorithms. The dataset, sourced from Kaggle, includes anonymized transaction data from European cardholders. The goal is to identify patterns that indicate potential fraud.

## 📁 Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (V1–V28 are anonymized features, plus `Time`, `Amount`, and `Class`)
- **Class Distribution**:
  - 0: Valid Transactions
  - 1: Fraudulent Transactions

## 🧰 Tools and Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📊 Exploratory Data Analysis
- Checked for missing values, duplicates
- Plotted distributions of anonymized features (V1–V28)
- Generated correlation heatmap
- Calculated class imbalance ratio

## 🧠 Model Building
- Logistic Regression
- Random Forest Classifier
- Evaluated using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC-AUC Score

📈 Results
The models are able to detect fraudulent transactions with high precision and recall, despite class imbalance.
Feature correlation insights helped reduce overfitting.
Successfully identified fraudulent transactions.
Achieved high recall and ROC-AUC on imbalanced data.

