# Customer Churn Prediction

Predict which telecom customers are likely to churn using machine learning classification models, with comprehensive EDA, model comparison, and an interactive Streamlit dashboard.

📓 **[View Full EDA Notebook](https://github.com/ShadowCreations5/customer-churn-prediction/blob/main/notebooks/churn_analysis.ipynb)** &nbsp;|&nbsp; 📊 **[View Project Report](https://github.com/ShadowCreations5/customer-churn-prediction/blob/main/reports/Project_Report.md)**

---

## 📋 Project Overview

This project analyzes the IBM Telco Customer Churn dataset to identify key factors that drive customer attrition and builds predictive models to flag at-risk customers. The analysis includes exploratory data analysis (EDA), feature engineering, training three classification models (Logistic Regression, Random Forest, XGBoost), and deploying an interactive prediction dashboard.

**Dataset:** IBM Telco Customer Churn — Real, publicly sourced data  
**Source URL:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv  
**Records:** 7,043 customers | **Features:** 21

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11+ | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosted classifier |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |
| Jupyter | Notebook-based analysis |

---

## 📁 Folder Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/telco_churn.csv              # Original dataset
│   └── processed/cleaned_churn.csv      # Cleaned & encoded data
├── notebooks/
│   └── churn_analysis.ipynb             # Full EDA & analysis notebook
├── src/
│   ├── data_prep.py                     # Data cleaning & encoding
│   ├── train_model.py                   # Model training pipeline
│   ├── evaluate.py                      # Evaluation & visualization
│   ├── run_all.py                       # Master pipeline script
│   └── generate_notebook.py             # Notebook generator
├── dashboard/
│   └── app.py                           # Streamlit prediction app
├── reports/
│   ├── figures/                         # All saved charts (.png)
│   └── Project_Report.md               # Written project report
├── models/
│   └── best_model.pkl                   # Saved best model
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 How to Run

### Run the full pipeline (data prep → training → evaluation)
```bash
python src/run_all.py
```
This will:
- Clean and preprocess the raw data
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate all models and save comparison charts
- Generate the project report
- Save the best model to `models/best_model.pkl`

### Run the Jupyter Notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard will open in your browser. Use the sidebar to input customer details and click "Predict Churn" to see the churn probability.

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|----------|
| **Logistic Regression** | 0.7402 | 0.5069 | 0.7861 | 0.6164 | **0.8414** |
| XGBoost | 0.7559 | 0.5282 | 0.7513 | 0.6203 | 0.8312 |
| Random Forest | 0.7637 | 0.5463 | 0.6471 | 0.5924 | 0.8261 |

> **Best Model:** Logistic Regression with ROC-AUC of **0.8414** — saved to `models/best_model.pkl`.

### Key Findings
- **26.5% overall churn rate** — moderate class imbalance addressed with balanced weights
- **Month-to-month contracts** have ~42% churn rate vs. ~3% for two-year contracts
- **New customers** (0-6 months) are the highest-risk segment
- **Higher monthly charges** correlate with higher churn probability

For detailed results, see [Project Report](reports/Project_Report.md).

---

## 📸 Sample Outputs

After running the pipeline, charts are saved to `reports/figures/`:
- `churn_rate_overall.png` — Overall churn distribution
- `churn_by_contract.png` — Churn rate by contract type
- `churn_by_tenure.png` — Churn rate by tenure groups
- `churn_by_monthly_charges.png` — Charge distributions by churn status
- `correlation_heatmap.png` — Feature correlation heatmap
- `roc_curves.png` — ROC curves for all 3 models
- `feature_importance.png` — Top feature importances
- `confusion_matrix.png` — Best model confusion matrix

---

## 📄 License

This project uses publicly available data from IBM's Telco Customer Churn dataset. The code is provided for educational purposes.

---

## 👤 Author

Sharon — Data Science Internship Project
