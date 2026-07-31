"""
train_model.py — Train classification models for customer churn prediction.

Models trained:
1. Logistic Regression (with class_weight='balanced')
2. Random Forest (with class_weight='balanced')
3. XGBoost (with scale_pos_weight for class imbalance)

The best model (by ROC-AUC) is saved to models/best_model.pkl.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_processed_data(filepath: str = None):
    """Load the processed (encoded) dataset and split into X, y."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")
    df = pd.read_csv(filepath)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    print(f"Loaded processed data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Churn distribution: {y.value_counts().to_dict()}  ({y.mean()*100:.1f}% churn rate)")
    return X, y


def compute_scale_pos_weight(y):
    """Compute XGBoost scale_pos_weight for class imbalance."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return n_neg / n_pos


def build_models(scale_pos_weight: float):
    """
    Build the three classification models.
    All models handle class imbalance:
    - Logistic Regression & Random Forest: class_weight='balanced'
    - XGBoost: scale_pos_weight
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        ),
    }
    return models


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all models and return them with their ROC-AUC scores.
    Returns: dict of {model_name: (fitted_model, roc_auc_score)}
    """
    from sklearn.metrics import roc_auc_score

    spw = compute_scale_pos_weight(y_train)
    print(f"Class imbalance — scale_pos_weight: {spw:.2f}")

    models = build_models(spw)
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Get prediction probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        auc = roc_auc_score(y_test, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")
        results[name] = (model, auc)

    return results


def save_best_model(results: dict, feature_names: list, save_dir: str = None):
    """Save the best model (by ROC-AUC) along with feature names to models/best_model.pkl."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc = results[best_name]

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pkl")

    # Save model along with metadata for the dashboard
    model_artifact = {
        "model": best_model,
        "model_name": best_name,
        "roc_auc": best_auc,
        "feature_names": feature_names,
    }
    joblib.dump(model_artifact, save_path)
    print(f"\nBest model: {best_name} (ROC-AUC: {best_auc:.4f})")
    print(f"Saved to {save_path}")

    return best_name, best_auc


def run_training(data_path: str = None):
    """
    Full training pipeline:
    1. Load processed data
    2. Split 80/20 stratified
    3. Train 3 models
    4. Save best model
    Returns: X_train, X_test, y_train, y_test, results dict
    """
    X, y = load_processed_data(data_path)

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    results = train_all_models(X_train, y_train, X_test, y_test)
    save_best_model(results, list(X.columns))

    return X_train, X_test, y_train, y_test, results


if __name__ == "__main__":
    run_training()
