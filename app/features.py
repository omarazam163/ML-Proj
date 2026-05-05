"""Shared data cleaning and feature engineering for the loan-approval project.

Used by both the training notebook and the Streamlit app so that inputs are
processed identically at train time and inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns expected in the raw Kaggle dataset (excluding Loan_ID and target).
RAW_CATEGORICAL = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
RAW_NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Final feature lists after engineering (used to build the ColumnTransformer).
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Credit_History",  # treated as categorical (0/1)
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "TotalIncome",
    "IncomePerLoanAmount",
    "LoanAmount_log",
    "TotalIncome_log",
    "EMI",
    "BalanceIncome",
    "DebtToIncome",
    "ApplicantIncome_log",
]


def clean_and_engineer(
    df: pd.DataFrame,
    impute_values: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Clean raw loan data and add engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with the original Kaggle columns.
    impute_values : dict | None
        Pre-computed imputation values (mode for categoricals, median for
        numerics). When None, they are computed from ``df`` and returned so the
        same values can be reused at inference time.

    Returns
    -------
    (cleaned_df, impute_values) : tuple[pd.DataFrame, dict]
    """
    df = df.copy()

    # Drop ID column if present.
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Normalize the "3+" dependents value to integer 3.
    if "Dependents" in df.columns:
        df["Dependents"] = (
            df["Dependents"].astype(str).str.replace("+", "", regex=False)
        )
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    # Compute imputation values if not supplied.
    if impute_values is None:
        impute_values = {}
        for col in ["Gender", "Married", "Self_Employed", "Credit_History",
                    "Dependents", "Loan_Amount_Term"]:
            if col in df.columns and df[col].notna().any():
                impute_values[col] = df[col].mode(dropna=True).iloc[0]
        for col in ["LoanAmount", "ApplicantIncome", "CoapplicantIncome"]:
            if col in df.columns and df[col].notna().any():
                impute_values[col] = float(df[col].median())

    # Apply imputation.
    for col, val in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Cast types after imputation.
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(int).astype(str)
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].astype(int).astype(str)

    # Engineered features.
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    # Avoid divide-by-zero; LoanAmount has been imputed but guard anyway.
    df["IncomePerLoanAmount"] = df["TotalIncome"] / df["LoanAmount"].replace(0, np.nan)
    df["IncomePerLoanAmount"] = df["IncomePerLoanAmount"].fillna(
        df["IncomePerLoanAmount"].median() if df["IncomePerLoanAmount"].notna().any() else 0
    )
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
    df["ApplicantIncome_log"] = np.log1p(df["ApplicantIncome"])

    # EMI = monthly installment (loan amount is in thousands).
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000.0) / term
    df["EMI"] = df["EMI"].fillna(df["EMI"].median() if df["EMI"].notna().any() else 0)

    # BalanceIncome = monthly income left after paying EMI.
    df["BalanceIncome"] = df["TotalIncome"] - df["EMI"]

    # Debt-to-income ratio (lower is safer).
    df["DebtToIncome"] = df["EMI"] / df["TotalIncome"].replace(0, np.nan)
    df["DebtToIncome"] = df["DebtToIncome"].fillna(
        df["DebtToIncome"].median() if df["DebtToIncome"].notna().any() else 0
    )

    # Map target if present.
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0}).astype(int)

    return df, impute_values
