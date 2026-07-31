"""
evaluate.py — Model Evaluation for Customer Churn Prediction

Generates:
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for all 3 models
- Model comparison table (saved to Project_Report.md)
- ROC curves for all models (saved as PNG)
- Feature importance bar chart (saved as PNG)
- Confusion matrix for the best model (saved as PNG)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# Set consistent plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics for a single model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def evaluate_all_models(results: dict, X_test, y_test):
    """
    Evaluate all trained models on the test set.
    Returns a DataFrame with metrics for each model.
    """
    metrics_list = []

    for name, (model, _) in results.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = name
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    df_metrics = df_metrics.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    return df_metrics


def print_comparison_table(df_metrics: pd.DataFrame):
    """Print the model comparison table in a readable format."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    for _, row in df_metrics.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Accuracy:  {row['Accuracy']:.4f}")
        print(f"  Precision: {row['Precision']:.4f}")
        print(f"  Recall:    {row['Recall']:.4f}")
        print(f"  F1 Score:  {row['F1 Score']:.4f}")
        print(f"  ROC-AUC:   {row['ROC-AUC']:.4f}")
    print("\n" + "=" * 70)
    best = df_metrics.iloc[0]
    print(f"\nBest Model: {best['Model']} (ROC-AUC: {best['ROC-AUC']:.4f})")


def save_comparison_table_md(df_metrics: pd.DataFrame, report_path: str):
    """Save the comparison table as a markdown table for the project report."""
    md_table = "\n| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n"
    md_table += "|-------|----------|-----------|--------|----------|----------|\n"
    for _, row in df_metrics.iterrows():
        md_table += (
            f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['F1 Score']:.4f} | {row['ROC-AUC']:.4f} |\n"
        )
    return md_table


def plot_roc_curves(results: dict, X_test, y_test, save_path: str):
    """Plot ROC curves for all 3 models on one chart and save as PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Logistic Regression": "#2196F3", "Random Forest": "#4CAF50", "XGBoost": "#FF5722"}

    for name, (model, _) in results.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = colors.get(name, "#9C27B0")
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {save_path}")


def plot_feature_importance(results: dict, feature_names: list, save_path: str, top_n: int = 15):
    """
    Plot feature importance from Random Forest or XGBoost.
    Shows the top N most important features.
    """
    # Prefer XGBoost, fall back to Random Forest
    for model_name in ["XGBoost", "Random Forest"]:
        if model_name in results:
            model = results[model_name][0]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                break
    else:
        print("No tree-based model found for feature importance.")
        return

    # Create importance DataFrame and sort
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color="#5C6BC0", edgecolor="white")

    # Highlight top 3
    for i, bar in enumerate(bars):
        if i >= len(bars) - 3:
            bar.set_color("#FF5722")

    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances ({model_name})")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {save_path}")


def plot_confusion_matrix(results: dict, X_test, y_test, save_path: str):
    """Plot confusion matrix for the best model (first in sorted results)."""
    # Find best model by AUC
    best_name = max(results, key=lambda k: results[k][1])
    model = results[best_name][0]
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {best_name}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    # Also print it
    print(f"\nConfusion Matrix ({best_name}):")
    print(f"                  Predicted No Churn  Predicted Churn")
    print(f"  Actual No Churn        {cm[0][0]:>5}            {cm[0][1]:>5}")
    print(f"  Actual Churn           {cm[1][0]:>5}            {cm[1][1]:>5}")


def run_evaluation(X_test, y_test, results: dict, feature_names: list, figures_dir: str = None, report_path: str = None):
    """
    Run the full evaluation pipeline:
    1. Compute metrics for all models
    2. Print and save the comparison table
    3. Plot ROC curves
    4. Plot feature importance
    5. Plot confusion matrix for best model
    Returns: df_metrics, md_table_string
    """
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    if report_path is None:
        report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "Project_Report.md")

    # Compute metrics
    df_metrics = evaluate_all_models(results, X_test, y_test)
    print_comparison_table(df_metrics)

    # Save comparison table as markdown
    md_table = save_comparison_table_md(df_metrics, report_path)

    # Plot ROC curves
    plot_roc_curves(results, X_test, y_test, os.path.join(figures_dir, "roc_curves.png"))

    # Plot feature importance
    plot_feature_importance(results, feature_names, os.path.join(figures_dir, "feature_importance.png"))

    # Plot confusion matrix for best model
    plot_confusion_matrix(results, X_test, y_test, os.path.join(figures_dir, "confusion_matrix.png"))

    return df_metrics, md_table


if __name__ == "__main__":
    # When run standalone, execute the full pipeline
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_prep import run_pipeline
    from train_model import run_training

    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)
    run_pipeline()

    print("\n" + "=" * 60)
    print("STEP 2: Model Training")
    print("=" * 60)
    X_train, X_test, y_train, y_test, results = run_training()

    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)
    feature_names = list(X_test.columns)
    df_metrics, md_table = run_evaluation(X_test, y_test, results, feature_names)
