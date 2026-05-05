"""Streamlit app for loan-approval prediction.

Loads the Decision Tree and SVM pipelines trained by
``notebooks/loan_approval.ipynb`` and serves an interactive form for
predictions. For the Decision Tree, the rule path that led to the
predicted leaf is also displayed (model interpretability).

Run from the project root::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make `app/` importable regardless of where streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.features import clean_and_engineer, CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_artifacts():
    models = {
        "Decision Tree": joblib.load(MODELS_DIR / "dt_pipeline.joblib"),
        "SVM":           joblib.load(MODELS_DIR / "svm_pipeline.joblib"),
    }
    impute = joblib.load(MODELS_DIR / "impute_values.joblib")
    try:
        thresholds = joblib.load(MODELS_DIR / "thresholds.joblib")
    except FileNotFoundError:
        thresholds = {name: 0.5 for name in models}
    return models, impute, thresholds


def explain_decision_path(pipeline, x_row: pd.DataFrame) -> list[str]:
    """Return human-readable conditions along the predicted leaf."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feat_names = prep.get_feature_names_out()
    x_trans = prep.transform(x_row)

    node_indicator = clf.decision_path(x_trans)
    leaf_id = clf.apply(x_trans)[0]
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    nodes = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    rules: list[str] = []
    for node in nodes:
        if node == leaf_id:
            break
        fname = feat_names[feature[node]]
        thr = threshold[node]
        val = x_trans[0, feature[node]]
        op = "<=" if val <= thr else ">"
        rules.append(f"`{fname}` = {val:.3f}  ({op} {thr:.3f})")
    return rules


def main() -> None:
    st.title("🏦 Loan Approval Predictor")
    st.write(
        "Predict whether a loan application will be **approved** or **rejected**, "
        "using a Decision Tree (interpretable) or an SVM (strong baseline). "
        "Trained on the Analytics Vidhya loan-prediction dataset."
    )

    try:
        models, impute_values, thresholds = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Trained models not found in `models/`. Run "
            "`python scripts/train.py` (or the notebook) first."
        )
        st.stop()

    st.sidebar.header("Model")
    model_name = st.sidebar.radio("Choose model", list(models.keys()))
    pipeline = models[model_name]
    default_thr = float(thresholds.get(model_name, 0.5))
    threshold = st.sidebar.slider(
        "Decision threshold (probability of approval)",
        min_value=0.10, max_value=0.90, value=default_thr, step=0.025,
        help="Lower = approve more applicants (higher recall). Default is the F1-optimal value found during training.",
    )

    st.subheader("Applicant details")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    with col2:
        applicant_income = st.number_input("Applicant income (monthly)", min_value=0,
                                           value=5000, step=100)
        coapplicant_income = st.number_input("Co-applicant income (monthly)",
                                             min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan amount (in thousands)", min_value=1,
                                      value=128, step=1)
        loan_term = st.selectbox("Loan term (months)",
                                 [360, 180, 120, 84, 60, 36, 12], index=0)
        credit_history = st.selectbox(
            "Credit history (1 = good, 0 = bad)", [1, 0])

    if st.button("Predict", type="primary"):
        raw_row = pd.DataFrame([{
            "Loan_ID": "APP_0001",
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }])

        cleaned, _ = clean_and_engineer(raw_row, impute_values=impute_values)
        X = cleaned[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

        proba = pipeline.predict_proba(X)[0]
        pred = int(proba[1] >= threshold)

        approved = pred == 1
        st.markdown("---")
        if approved:
            st.success(f"✅ **Approved** — P(approval) = {proba[1]:.1%}  (threshold {threshold:.2f})")
        else:
            st.error(f"❌ **Rejected** — P(approval) = {proba[1]:.1%}  (threshold {threshold:.2f})")

        st.write(
            f"Model: **{model_name}**  \n"
            f"P(Approved) = {proba[1]:.3f}  |  P(Rejected) = {proba[0]:.3f}"
        )

        if model_name == "Decision Tree":
            st.subheader("Why? — Decision path to this prediction")
            rules = explain_decision_path(pipeline, X)
            if rules:
                for r in rules:
                    st.markdown("- " + r)
            else:
                st.info("Tree returned the prediction at the root.")
        else:
            st.subheader("Engineered feature snapshot")
            st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
