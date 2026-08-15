# Dry Bean Classification

## Problem Statement

The aim of this project is to classify dry bean samples into their respective
bean varieties using machine learning classification algorithms.

The dataset contains measurements obtained from images of dry bean samples.
These measurements describe different physical and shape-related properties
of the beans. The task is to use these features to predict the bean class.

Five classification models were implemented and compared:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

## Dataset Description

The project uses the UCI Dry Bean dataset.

The dataset contains 13,611 observations and 16 input features. There are
seven different dry bean classes in the target variable.

The 16 features describe properties such as area, perimeter, major and minor
axis length, aspect ratio, eccentricity, convex area, equivalent diameter,
extent, solidity, roundness, compactness and shape factors.

The target variable is `Class`, which represents the type of dry bean.

## GitHub Repository

[GitHub Repository](https://github.com/2025ac05114/dry_bean_classification)

## Data Preparation

The dataset was inspected to understand its structure, data types, class
distribution, missing values, duplicate records and feature relationships.
Exploratory analysis included feature distributions, outlier visualizations
and a correlation matrix.

The data was divided into training and testing sets using a stratified split,
which keeps the class distribution similar in both sets.

Feature scaling was used for Logistic Regression and KNN. The scaling step was
included inside a pipeline for these models so the same preprocessing is
applied during prediction.

## Machine Learning Models

### Logistic Regression

A linear classification baseline for the multiclass Dry Bean problem.

### Decision Tree

A tree-based classifier that learns classification rules through feature
splits.

### K-Nearest Neighbors

A distance-based classifier. Feature scaling was applied because the model
uses distances between observations.

### Naive Bayes

Gaussian Naive Bayes was used because the input features are numerical.

### Random Forest

An ensemble of decision trees whose combined predictions are used for
classification.

## Model Comparison

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score
and Matthews Correlation Coefficient (MCC).

For this multiclass problem, AUC uses the one-vs-rest (`ovr`) strategy.
Precision, Recall and F1 use weighted averaging.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9448 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest | 0.9218 | 0.9929 | 0.9219 | 0.9218 | 0.9217 | 0.9054 |

## Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.9207, AUC of 0.9948, F1 score of 0.9209 and MCC of 0.9042. It ranked 2 for accuracy and 2 for F1 score among the five implemented models.

### Decision Tree

Decision Tree achieved an accuracy of 0.8917, AUC of 0.9448, F1 score of 0.8913 and MCC of 0.8691. It ranked 4 for accuracy and 4 for F1 score among the five implemented models.

### KNN

KNN achieved an accuracy of 0.9166, AUC of 0.9833, F1 score of 0.9168 and MCC of 0.8992. It ranked 3 for accuracy and 3 for F1 score among the five implemented models.

### Naive Bayes

Naive Bayes achieved an accuracy of 0.7639, AUC of 0.9672, F1 score of 0.7615 and MCC of 0.7154. It ranked 5 for accuracy and 5 for F1 score among the five implemented models.

### Random Forest

Random Forest achieved an accuracy of 0.9218, AUC of 0.9929, F1 score of 0.9217 and MCC of 0.9054. It ranked 1 for accuracy and 1 for F1 score among the five implemented models.

### Overall Winner

Using the mean of the six required evaluation metrics as the overall score
(the assignment does not prescribe a separate winner-selection rule),
**Random Forest** is the overall winner for this test set.

Its aggregate score across Accuracy, AUC, Precision, Recall, F1 and MCC is
**0.9309**.

## Project Structure

```text
dry_bean_classification/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application allows the evaluator to upload the supplied test CSV, select
one of the five trained models, view the six evaluation metrics, and inspect
the confusion matrix and classification report.

## Deployment

The Streamlit application has been deployed using Streamlit Community Cloud.

**Live App:** https://2025ac05114-dry-bean-classification-app-mh5iaz.streamlit.app/
