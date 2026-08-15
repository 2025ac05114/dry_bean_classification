
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

## Data Preparation

The dataset was first inspected to understand its structure, data types,
class distribution, missing values, duplicate records and feature
relationships.

Exploratory analysis was also performed using feature distributions,
outlier visualizations and a correlation matrix.

The data was divided into training and testing sets using a stratified split.
This keeps the class distribution similar in both sets.

Feature scaling was used for models where it is useful, particularly Logistic
Regression and KNN. The scaling step was included inside a pipeline for these
models so that the same preprocessing is applied during prediction.

## Machine Learning Models

Five classification models were implemented.

### Logistic Regression

Logistic Regression was used as a linear classification model and provided a
strong baseline for the dataset.

### Decision Tree

Decision Tree was used to learn classification rules by repeatedly splitting
the data based on feature values.

### K-Nearest Neighbors

KNN classifies an observation using the classes of its nearest training
observations. Feature scaling was applied because KNN is based on distances.

### Naive Bayes

Gaussian Naive Bayes was used because the input features are numerical.

### Random Forest

Random Forest combines multiple decision trees and uses their combined
predictions for classification.

## Model Comparison

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score
and Matthews Correlation Coefficient (MCC).

For multiclass evaluation, AUC was calculated using the one-vs-rest approach.
Precision, Recall and F1 Score were calculated using weighted averaging.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| KNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest | 0.9203 | 0.9927 | 0.9205 | 0.9203 | 0.9203 | 0.9036 |

## Observations

### Logistic Regression

Logistic Regression gave the best result among the five models. It achieved
92.07% accuracy, with an AUC of 0.9948 and an F1 score of 0.9209.

It also had the highest MCC value of 0.9042. Overall, it performed slightly
better than Random Forest and KNN on the test data.

### Decision Tree

Decision Tree achieved an accuracy of 89.20%, which was lower than Logistic
Regression, Random Forest and KNN.

Its F1 score was 0.8916 and its MCC was 0.8696. The model still performed
reasonably well, but its results were not as strong as the other tree-based
ensemble model, Random Forest.

### KNN

KNN achieved 91.66% accuracy and an F1 score of 0.9168. Its performance was
close to Logistic Regression and Random Forest.

The AUC of 0.9833 also shows that KNN was able to separate the bean classes
well. Since KNN uses distances between observations, feature scaling was
important for this model.

### Naive Bayes

Naive Bayes had the lowest overall performance of the five models. Its
accuracy was 76.39%, with an F1 score of 0.7615 and an MCC of 0.7154.

Its AUC was 0.9672, which was still relatively high, but its accuracy,
precision, recall and F1 score were clearly lower than the other models.

### Random Forest

Random Forest was very close to Logistic Regression. It achieved 92.03%
accuracy, an AUC of 0.9927 and an F1 score of 0.9203.

It was the second-best model across the reported metrics, but its scores were
slightly lower than Logistic Regression.

### Overall Winner

Logistic Regression was selected as the overall winner because it achieved
the highest value for all six reported evaluation metrics: Accuracy, AUC,
Precision, Recall, F1 and MCC.

The difference between Logistic Regression and Random Forest was very small,
so Random Forest was also a strong performer. However, based on the actual
test results, Logistic Regression had the best overall performance.

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
## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt