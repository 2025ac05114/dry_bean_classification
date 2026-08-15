
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

    expected_features = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRatio",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "Roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4"
    ]

    if "Class" not in data.columns:
        st.error("The uploaded CSV must contain a 'Class' column.")
        st.stop()

    X = data.drop(columns=["Class"])
    y = data["Class"]

    if list(X.columns) != expected_features:
        st.error("Uploaded CSV does not have the expected feature columns.")
        st.stop()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, probabilities, multi_class="ovr")
    precision = precision_score(y, predictions, average="weighted")
    recall = recall_score(y, predictions, average="weighted")
    f1 = f1_score(y, predictions, average="weighted")
    mcc = matthews_corrcoef(y, predictions)

    st.subheader("Evaluation Metrics")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "Score": [accuracy, auc, precision, recall, f1, mcc]
    })

    st.dataframe(metrics)

    st.subheader("Predictions")

    prediction_data = pd.DataFrame({
        "Actual": y,
        "Predicted": predictions
    })

    st.dataframe(prediction_data.head(20))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions, labels=model.classes_)

    st.dataframe(
        pd.DataFrame(
            cm,
            index=model.classes_,
            columns=model.classes_
        )
    )

    st.subheader("Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True
    )

    st.dataframe(pd.DataFrame(report).transpose())
