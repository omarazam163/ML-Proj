"""Retrain Decision Tree and SVM on the upgraded feature set and save the
best pipelines.

Run from project root::

    .venv\\Scripts\\python.exe scripts\\train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.features import (  # noqa: E402
    clean_and_engineer,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def tune(name: str, pipe: Pipeline, grid: dict, X, y) -> Pipeline:
    search = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
    search.fit(X, y)
    print(f"  best CV F1: {search.best_score_:.4f}  params: {search.best_params_}")
    return search.best_estimator_


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "threshold": threshold,
    }


def best_threshold(model, X_val, y_val) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 25):
        pred = (proba >= t).astype(int)
        f = f1_score(y_val, pred)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    df, impute_values = clean_and_engineer(raw)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df["Loan_Status"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train={X_tr.shape}  test={X_te.shape}")

    pre = build_preprocessor

    print("\n[Decision Tree] tuning...")
    dt = tune(
        "dt",
        Pipeline([("prep", pre()), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        {
            "clf__max_depth": [3, 4, 5, 6, 8, None],
            "clf__min_samples_split": [2, 5, 10, 20],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        },
        X_tr,
        y_tr,
    )

    print("\n[SVM] tuning...")
    svm = tune(
        "svm",
        Pipeline([("prep", pre()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        {
            "clf__C": [0.1, 1, 3, 10, 30],
            "clf__kernel": ["linear", "rbf"],
            "clf__gamma": ["scale", "auto"],
            "clf__class_weight": [None, "balanced"],
        },
        X_tr,
        y_tr,
    )

    models = {"Decision Tree": dt, "SVM": svm}

    print("\n=== Default-threshold (0.5) test metrics ===")
    rows = []
    for name, m in models.items():
        rows.append({"model": name, **evaluate(m, X_te, y_te)})
    summary = pd.DataFrame(rows).set_index("model").round(4)
    print(summary.to_string())

    # Tune threshold per model on the test set's first 50% (held-out validation) for robustness:
    # split the test set to derive thresholds without training-set leakage.
    X_val, X_final, y_val, y_final = train_test_split(
        X_te, y_te, test_size=0.5, stratify=y_te, random_state=RANDOM_STATE
    )

    print("\n=== With per-model tuned threshold (val 50% / final 50% of test) ===")
    rows = []
    thresholds = {}
    for name, m in models.items():
        t = best_threshold(m, X_val, y_val)
        thresholds[name] = t
        rows.append({"model": name, **evaluate(m, X_final, y_final, threshold=t)})
    tuned_summary = pd.DataFrame(rows).set_index("model").round(4)
    print(tuned_summary.to_string())

    # Persist artifacts.
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(dt, out / "dt_pipeline.joblib")
    joblib.dump(svm, out / "svm_pipeline.joblib")
    joblib.dump(impute_values, out / "impute_values.joblib")
    joblib.dump(thresholds, out / "thresholds.joblib")
    print(f"\nSaved 2 pipelines + impute_values + thresholds to {out}")


if __name__ == "__main__":
    main()
