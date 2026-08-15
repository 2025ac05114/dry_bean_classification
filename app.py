import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("Dry Bean Classification")
st.write("Compare five machine learning classification models.")

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl")
}

model_name = st.selectbox("Select a model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write("Rows:", data.shape[0])
    st.write("Columns:", data.shape[1])
    st.dataframe(data.head())

    if "Class" not in data.columns:
        st.error("The uploaded CSV must contain a 'Class' column.")
        st.stop()

    X_uploaded = data.drop(columns=["Class"])
    y_uploaded = data["Class"]

    # Read the feature names from the fitted model rather than hard-coding
    # dataset column names. This keeps the app aligned with the exact
    # training schema saved inside the model pipeline/estimator.
    if not hasattr(model, "feature_names_in_"):
        st.error("The selected saved model does not contain feature-name metadata.")
        st.stop()

    expected_features = list(model.feature_names_in_)
    uploaded_features = list(X_uploaded.columns)

    if uploaded_features != expected_features:
        st.error("Uploaded CSV does not match the feature schema expected by the selected model.")
        st.write("Expected feature columns:", expected_features)
        st.write("Uploaded feature columns:", uploaded_features)
        st.stop()

    predictions = model.predict(X_uploaded)
    probabilities = model.predict_proba(X_uploaded)

    accuracy = accuracy_score(y_uploaded, predictions)
    auc = roc_auc_score(
        y_uploaded,
        probabilities,
        multi_class="ovr",
        labels=model.classes_
    )
    precision = precision_score(y_uploaded, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_uploaded, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_uploaded, predictions, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_uploaded, predictions)

    st.subheader("Predictions")
    st.dataframe(pd.DataFrame({
        "Actual": y_uploaded,
        "Predicted": predictions
    }).head(20))

    st.subheader("Evaluation Metrics")
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "Score": [accuracy, auc, precision, recall, f1, mcc]
    })
    st.dataframe(metrics)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_uploaded, predictions, labels=model.classes_)
    st.dataframe(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))

    st.subheader("Classification Report")
    report = classification_report(
        y_uploaded,
        predictions,
        labels=model.classes_,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose())
