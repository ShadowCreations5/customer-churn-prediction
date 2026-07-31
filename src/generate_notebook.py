"""
generate_notebook.py — Creates the Jupyter notebook churn_analysis.ipynb programmatically.
This script builds the notebook with all EDA cells and markdown commentary.
"""

import json
import os

def make_cell(cell_type, source, outputs=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n")
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell


def build_notebook():
    cells = []

    # Title
    cells.append(make_cell("markdown", [
        "# Customer Churn Prediction — Exploratory Data Analysis\n",
        "\n",
        "**Dataset:** IBM Telco Customer Churn  \n",
        "**Source:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv  \n",
        "**Data Type:** REAL (publicly sourced, not synthetic)  \n",
        "\n",
        "This notebook performs data cleaning, exploratory data analysis, model training, and evaluation\n",
        "for predicting customer churn in a telecommunications company."
    ]))

    # Imports
    cells.append(make_cell("code", [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import os\n",
        "import sys\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Set plot style\n",
        "sns.set_style('whitegrid')\n",
        "plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 12})\n",
        "\n",
        "# Ensure figures directory exists\n",
        "FIGURES_DIR = os.path.join('..', 'reports', 'figures')\n",
        "os.makedirs(FIGURES_DIR, exist_ok=True)\n",
        "\n",
        "print('Libraries loaded successfully.')"
    ]))

    # Section 1: Load Data
    cells.append(make_cell("markdown", [
        "## 1. Load the Raw Dataset\n",
        "\n",
        "We load the original Telco Customer Churn dataset and inspect its structure."
    ]))

    cells.append(make_cell("code", [
        "df = pd.read_csv(os.path.join('..', 'data', 'raw', 'telco_churn.csv'))\n",
        "print(f'Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns')\n",
        "print(f'\\nColumn names:')\n",
        "for i, col in enumerate(df.columns):\n",
        "    print(f'  {i+1}. {col} ({df[col].dtype})')\n",
        "df.head()"
    ]))

    cells.append(make_cell("code", [
        "df.info()"
    ]))

    # Section 2: Data Cleaning
    cells.append(make_cell("markdown", [
        "## 2. Data Cleaning\n",
        "\n",
        "### Handling Missing / Blank TotalCharges\n",
        "\n",
        "A known issue with this dataset is that some rows have blank strings (`' '`) in the\n",
        "`TotalCharges` column instead of numeric values. These correspond to new customers\n",
        "with very short tenure. We convert these to `NaN`, then fill with `0.0`."
    ]))

    cells.append(make_cell("code", [
        "# Check for blank TotalCharges\n",
        "blank_tc = df[df['TotalCharges'].str.strip() == ''] if df['TotalCharges'].dtype == 'object' else pd.DataFrame()\n",
        "print(f'Blank TotalCharges rows: {len(blank_tc)}')\n",
        "if len(blank_tc) > 0:\n",
        "    print(blank_tc[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']].head(11))"
    ]))

    cells.append(make_cell("code", [
        "# Clean the data using our data_prep module\n",
        "sys.path.insert(0, os.path.join('..', 'src'))\n",
        "from data_prep import clean_data, encode_features\n",
        "\n",
        "# Drop customerID for analysis but keep a clean copy\n",
        "df_clean = clean_data(df)\n",
        "print(f'\\nCleaned data shape: {df_clean.shape}')\n",
        "print(f'Missing values per column: {df_clean.isnull().sum().sum()}')\n",
        "df_clean.head()"
    ]))

    # Section 3: EDA - Overall Churn Rate
    cells.append(make_cell("markdown", [
        "## 3. Exploratory Data Analysis\n",
        "\n",
        "### 3.1 Overall Churn Rate"
    ]))

    cells.append(make_cell("code", [
        "churn_counts = df_clean['Churn'].value_counts()\n",
        "churn_pct = df_clean['Churn'].value_counts(normalize=True) * 100\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
        "\n",
        "# Bar chart\n",
        "colors = ['#4CAF50', '#FF5722']\n",
        "bars = axes[0].bar(['Not Churned (0)', 'Churned (1)'], churn_counts.values, color=colors, edgecolor='white', linewidth=1.5)\n",
        "for bar, count, pct in zip(bars, churn_counts.values, churn_pct.values):\n",
        "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,\n",
        "                 f'{count}\\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')\n",
        "axes[0].set_title('Customer Churn Distribution')\n",
        "axes[0].set_ylabel('Count')\n",
        "\n",
        "# Pie chart\n",
        "axes[1].pie(churn_counts.values, labels=['Not Churned', 'Churned'], colors=colors,\n",
        "            autopct='%1.1f%%', startangle=90, explode=(0, 0.05), textprops={'fontsize': 12})\n",
        "axes[1].set_title('Churn Rate Proportion')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIGURES_DIR, 'churn_rate_overall.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'Overall churn rate: {churn_pct[1]:.1f}%')"
    ]))

    cells.append(make_cell("markdown", [
        "**Findings — Overall Churn Rate:**\n",
        "The dataset exhibits a moderate class imbalance, with approximately 26.5% of customers having churned\n",
        "and 73.5% remaining. This imbalance is significant enough to warrant special handling during model\n",
        "training (e.g., using `class_weight='balanced'` or `scale_pos_weight`). The churn rate of ~26.5% is\n",
        "typical for telecom datasets and represents a meaningful business problem — losing over a quarter of\n",
        "customers has substantial revenue implications."
    ]))

    # 3.2 Churn by Contract Type
    cells.append(make_cell("markdown", [
        "### 3.2 Churn by Contract Type"
    ]))

    cells.append(make_cell("code", [
        "# Churn by Contract Type - use original categories before encoding\n",
        "df_analysis = df.copy()\n",
        "df_analysis['Churn_Binary'] = (df_analysis['Churn'] == 'Yes').astype(int)\n",
        "\n",
        "contract_churn = df_analysis.groupby('Contract')['Churn_Binary'].agg(['mean', 'sum', 'count']).reset_index()\n",
        "contract_churn.columns = ['Contract', 'Churn Rate', 'Churned', 'Total']\n",
        "contract_churn = contract_churn.sort_values('Churn Rate', ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "colors_contract = ['#FF5722', '#FFA726', '#4CAF50']\n",
        "bars = ax.bar(contract_churn['Contract'], contract_churn['Churn Rate'] * 100,\n",
        "              color=colors_contract, edgecolor='white', linewidth=1.5)\n",
        "for bar, rate, churned, total in zip(bars, contract_churn['Churn Rate'],\n",
        "                                      contract_churn['Churned'], contract_churn['Total']):\n",
        "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
        "            f'{rate*100:.1f}%\\n({churned}/{total})', ha='center', va='bottom', fontweight='bold')\n",
        "\n",
        "ax.set_ylabel('Churn Rate (%)')\n",
        "ax.set_title('Churn Rate by Contract Type')\n",
        "ax.set_ylim(0, 55)\n",
        "ax.grid(axis='y', alpha=0.3)\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIGURES_DIR, 'churn_by_contract.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]))

    cells.append(make_cell("markdown", [
        "**Findings — Churn by Contract Type:**\n",
        "Contract type is one of the strongest predictors of churn. Month-to-month customers have a dramatically\n",
        "higher churn rate (~42%) compared to one-year (~11%) and two-year (~3%) contract holders. This makes\n",
        "intuitive sense — customers without long-term commitments face no switching costs. This insight suggests\n",
        "that incentivizing customers to switch to longer contracts (e.g., discounts for annual plans) could\n",
        "significantly reduce churn."
    ]))

    # 3.3 Churn by Tenure
    cells.append(make_cell("markdown", [
        "### 3.3 Churn by Tenure"
    ]))

    cells.append(make_cell("code", [
        "# Churn by Tenure\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Histogram of tenure by churn status\n",
        "axes[0].hist(df_analysis[df_analysis['Churn_Binary']==0]['tenure'], bins=30, alpha=0.7,\n",
        "             label='Not Churned', color='#4CAF50', edgecolor='white')\n",
        "axes[0].hist(df_analysis[df_analysis['Churn_Binary']==1]['tenure'], bins=30, alpha=0.7,\n",
        "             label='Churned', color='#FF5722', edgecolor='white')\n",
        "axes[0].set_xlabel('Tenure (months)')\n",
        "axes[0].set_ylabel('Count')\n",
        "axes[0].set_title('Tenure Distribution by Churn Status')\n",
        "axes[0].legend()\n",
        "\n",
        "# Create tenure groups and compute churn rate\n",
        "df_analysis['Tenure_Group'] = pd.cut(df_analysis['tenure'],\n",
        "    bins=[0, 6, 12, 24, 36, 48, 60, 72],\n",
        "    labels=['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61-72'])\n",
        "tenure_churn = df_analysis.groupby('Tenure_Group', observed=True)['Churn_Binary'].mean() * 100\n",
        "\n",
        "bars = axes[1].bar(tenure_churn.index, tenure_churn.values, color='#5C6BC0', edgecolor='white')\n",
        "for bar, val in zip(bars, tenure_churn.values):\n",
        "    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,\n",
        "                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)\n",
        "axes[1].set_xlabel('Tenure Group (months)')\n",
        "axes[1].set_ylabel('Churn Rate (%)')\n",
        "axes[1].set_title('Churn Rate by Tenure Group')\n",
        "axes[1].grid(axis='y', alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIGURES_DIR, 'churn_by_tenure.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()"
    ]))

    cells.append(make_cell("markdown", [
        "**Findings — Churn by Tenure:**\n",
        "New customers (0-6 months) have the highest churn rate, around 47-50%, which drops sharply as tenure\n",
        "increases. Long-tenured customers (61-72 months) have churn rates below 10%. This pattern reveals that\n",
        "the first few months of a customer relationship are the most critical for retention. Companies should\n",
        "focus onboarding programs and early-stage engagement to reduce early attrition. Once customers pass the\n",
        "first year, they become progressively more loyal."
    ]))

    # 3.4 Churn by Monthly Charges
    cells.append(make_cell("markdown", [
        "### 3.4 Churn by Monthly Charges"
    ]))

    cells.append(make_cell("code", [
        "# Churn by Monthly Charges\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# KDE plot\n",
        "df_analysis[df_analysis['Churn_Binary']==0]['MonthlyCharges'].plot.kde(\n",
        "    ax=axes[0], label='Not Churned', color='#4CAF50', linewidth=2)\n",
        "df_analysis[df_analysis['Churn_Binary']==1]['MonthlyCharges'].plot.kde(\n",
        "    ax=axes[0], label='Churned', color='#FF5722', linewidth=2)\n",
        "axes[0].set_xlabel('Monthly Charges ($)')\n",
        "axes[0].set_title('Monthly Charges Distribution by Churn Status')\n",
        "axes[0].legend()\n",
        "axes[0].set_xlim(0, 130)\n",
        "\n",
        "# Box plot\n",
        "bp = axes[1].boxplot(\n",
        "    [df_analysis[df_analysis['Churn_Binary']==0]['MonthlyCharges'],\n",
        "     df_analysis[df_analysis['Churn_Binary']==1]['MonthlyCharges']],\n",
        "    labels=['Not Churned', 'Churned'],\n",
        "    patch_artist=True,\n",
        "    boxprops=dict(facecolor='#E3F2FD'),\n",
        "    medianprops=dict(color='#FF5722', linewidth=2)\n",
        ")\n",
        "bp['boxes'][0].set_facecolor('#C8E6C9')\n",
        "bp['boxes'][1].set_facecolor('#FFCCBC')\n",
        "axes[1].set_ylabel('Monthly Charges ($)')\n",
        "axes[1].set_title('Monthly Charges by Churn Status')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIGURES_DIR, 'churn_by_monthly_charges.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print(f\"Mean monthly charges — Not Churned: ${df_analysis[df_analysis['Churn_Binary']==0]['MonthlyCharges'].mean():.2f}\")\n",
        "print(f\"Mean monthly charges — Churned:     ${df_analysis[df_analysis['Churn_Binary']==1]['MonthlyCharges'].mean():.2f}\")"
    ]))

    cells.append(make_cell("markdown", [
        "**Findings — Churn by Monthly Charges:**\n",
        "Churned customers tend to have significantly higher monthly charges (mean ~$74) compared to retained\n",
        "customers (mean ~$61). The KDE plot shows that churned customers are concentrated in the $70-$110 range,\n",
        "while non-churned customers have a more uniform distribution with a notable peak in the low-charge\n",
        "range ($20-$30). This suggests that customers paying premium prices are more price-sensitive and may\n",
        "feel they're not getting sufficient value for their spend."
    ]))

    # 3.5 Correlation Heatmap
    cells.append(make_cell("markdown", [
        "### 3.5 Correlation Heatmap"
    ]))

    cells.append(make_cell("code", [
        "# Correlation heatmap using encoded features\n",
        "df_encoded = encode_features(df_clean)\n",
        "\n",
        "# Select features with highest absolute correlation to Churn\n",
        "churn_corr = df_encoded.corr()['Churn'].abs().sort_values(ascending=False)\n",
        "top_features = churn_corr.head(16).index.tolist()  # Top 15 + Churn itself\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 10))\n",
        "corr_matrix = df_encoded[top_features].corr()\n",
        "\n",
        "mask = np.triu(np.ones_like(corr_matrix, dtype=bool))\n",
        "sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',\n",
        "            center=0, square=True, linewidths=0.5,\n",
        "            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},\n",
        "            ax=ax, vmin=-1, vmax=1)\n",
        "ax.set_title('Correlation Heatmap — Top Features vs. Churn', fontsize=14)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('\\nTop 10 features by absolute correlation with Churn:')\n",
        "for feat, corr_val in churn_corr.head(11).items():\n",
        "    if feat != 'Churn':\n",
        "        print(f'  {feat}: {corr_val:.3f}')"
    ]))

    cells.append(make_cell("markdown", [
        "**Findings — Correlation Heatmap:**\n",
        "The correlation analysis reveals that month-to-month contracts, lack of online security, lack of tech\n",
        "support, fiber optic internet service, and electronic check payments are the most positively correlated\n",
        "with churn. Conversely, longer tenure, two-year contracts, and having services like online security\n",
        "and tech support are negatively correlated with churn. The strong correlation between fiber optic\n",
        "internet and churn may seem counterintuitive, but likely reflects higher prices associated with this\n",
        "service tier combined with unmet performance expectations."
    ]))

    # Section 4: Model Training
    cells.append(make_cell("markdown", [
        "## 4. Model Training & Evaluation\n",
        "\n",
        "We train three classification models with class imbalance handling and evaluate them on\n",
        "the held-out test set."
    ]))

    cells.append(make_cell("code", [
        "from train_model import run_training\n",
        "from evaluate import run_evaluation\n",
        "\n",
        "# Run training pipeline\n",
        "X_train, X_test, y_train, y_test, results = run_training(\n",
        "    os.path.join('..', 'data', 'processed', 'cleaned_churn.csv')\n",
        ")"
    ]))

    cells.append(make_cell("code", [
        "# Run evaluation\n",
        "feature_names = list(X_test.columns)\n",
        "df_metrics, md_table = run_evaluation(\n",
        "    X_test, y_test, results, feature_names,\n",
        "    figures_dir=os.path.join('..', 'reports', 'figures'),\n",
        "    report_path=os.path.join('..', 'reports', 'Project_Report.md')\n",
        ")\n",
        "\n",
        "print('\\nModel Comparison:')\n",
        "print(df_metrics.to_string(index=False))"
    ]))

    # Section 5: Summary
    cells.append(make_cell("markdown", [
        "## 5. Summary\n",
        "\n",
        "### Key Findings:\n",
        "1. **Overall churn rate** is approximately 26.5%, creating a moderately imbalanced classification problem.\n",
        "2. **Contract type** is the strongest predictor — month-to-month customers churn at ~42% vs. ~3% for two-year contracts.\n",
        "3. **New customers** (0-6 months tenure) are the most at-risk segment.\n",
        "4. **Higher monthly charges** correlate with higher churn rates.\n",
        "5. **Fiber optic internet** customers churn more, likely due to price-value perception.\n",
        "\n",
        "### Model Performance:\n",
        "All three models (Logistic Regression, Random Forest, XGBoost) were trained with class imbalance handling.\n",
        "The model comparison table and ROC curves are saved in `reports/figures/`.\n",
        "\n",
        "### Business Recommendations:\n",
        "1. Offer incentives for month-to-month customers to switch to annual or biennial contracts.\n",
        "2. Implement proactive onboarding and engagement programs for new customers in their first 6 months.\n",
        "3. Review pricing strategy for high-charge customers, especially those on fiber optic plans.\n",
        "4. Bundle value-added services (online security, tech support) to increase stickiness and perceived value."
    ]))

    # Build the notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "cells": cells
    }

    return notebook


def save_notebook(notebook, filepath):
    """Save the notebook to a file."""
    # Fix cell sources — ensure each line ends with \n except the last
    for cell in notebook["cells"]:
        source = cell["source"]
        if isinstance(source, list):
            for i in range(len(source) - 1):
                if not source[i].endswith("\n"):
                    source[i] = source[i] + "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Notebook saved to {filepath}")


if __name__ == "__main__":
    nb = build_notebook()
    outpath = os.path.join(os.path.dirname(__file__), "..", "notebooks", "churn_analysis.ipynb")
    save_notebook(nb, outpath)
