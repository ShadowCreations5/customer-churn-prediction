"""
app.py — Streamlit Dashboard for Customer Churn Prediction

This dashboard allows users to:
1. Enter customer details via input widgets
2. Predict churn probability with a button click
3. View overall churn rate from the dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")


@st.cache_resource
def load_model():
    """Load the saved best model artifact."""
    artifact = joblib.load(MODEL_PATH)
    return artifact


@st.cache_data
def load_raw_data():
    """Load the raw data for summary statistics."""
    return pd.read_csv(DATA_PATH)


def build_input_features(artifact):
    """Build feature input vector from user widgets, matching training features."""
    feature_names = artifact["feature_names"]

    st.sidebar.header("📋 Customer Details")

    # Core numeric inputs
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=0.5)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges), step=10.0)

    # Categorical inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    # Build raw dict matching original columns (before encoding)
    raw_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Create a DataFrame and one-hot encode to match training features
    input_df = pd.DataFrame([raw_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Ensure all expected features exist (fill missing with 0)
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns to match training data exactly
    input_encoded = input_encoded[feature_names]

    # Convert all to numeric
    for col in input_encoded.columns:
        input_encoded[col] = pd.to_numeric(input_encoded[col], errors="coerce").fillna(0)

    return input_encoded


def main():
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown(
        "Predict whether a telecom customer is likely to churn based on their profile. "
        "Adjust the customer details in the sidebar, then click **Predict Churn**."
    )

    # Load model and data
    try:
        artifact = load_model()
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run the training pipeline first (`python src/train_model.py`).")
        return

    model = artifact["model"]
    model_name = artifact["model_name"]
    model_auc = artifact["roc_auc"]

    st.markdown(f"**Model in use:** {model_name} &nbsp;|&nbsp; **ROC-AUC:** {model_auc:.4f}")
    st.markdown("---")

    # Build input features from sidebar widgets
    input_features = build_input_features(artifact)

    # Prediction section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔮 Churn Prediction")

        if st.button("🚀 Predict Churn", use_container_width=True):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_features)[0][1]
            else:
                proba = model.predict(input_features)[0]

            churn_pct = proba * 100

            # Color indicator based on risk level
            if churn_pct >= 70:
                color = "#FF4444"
                risk = "🔴 HIGH RISK"
                emoji = "⚠️"
            elif churn_pct >= 40:
                color = "#FFA500"
                risk = "🟡 MEDIUM RISK"
                emoji = "⚡"
            else:
                color = "#4CAF50"
                risk = "🟢 LOW RISK"
                emoji = "✅"

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}44); 
                            border: 2px solid {color}; border-radius: 16px; padding: 30px; 
                            text-align: center; margin: 20px 0;">
                    <h1 style="color: {color}; margin: 0; font-size: 3em;">{emoji} {churn_pct:.1f}%</h1>
                    <h3 style="color: {color}; margin: 10px 0 0 0;">{risk}</h3>
                    <p style="margin: 10px 0 0 0; font-size: 1.1em;">Likelihood of customer churning</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if churn_pct >= 50:
                st.warning("💡 **Recommendation:** Consider offering this customer a loyalty discount, "
                           "contract upgrade incentive, or proactive tech support outreach.")
            else:
                st.success("✅ This customer appears to be at low risk of churning. Continue monitoring.")

    with col2:
        st.subheader("📈 Dataset Overview — Churn Distribution")

        try:
            df_raw = load_raw_data()
            churn_counts = df_raw["Churn"].value_counts()
            total = len(df_raw)
            churn_rate = (churn_counts.get("Yes", 0) / total) * 100

            fig, ax = plt.subplots(figsize=(6, 4))
            colors_chart = ["#4CAF50", "#FF5722"]
            bars = ax.bar(["Not Churned", "Churned"],
                          [churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
                          color=colors_chart, edgecolor="white", linewidth=1.5)

            for bar, count in zip(bars, [churn_counts.get("No", 0), churn_counts.get("Yes", 0)]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        f"{count}\n({count/total*100:.1f}%)",
                        ha="center", va="bottom", fontweight="bold", fontsize=11)

            ax.set_ylabel("Number of Customers")
            ax.set_title(f"Overall Churn Rate: {churn_rate:.1f}%")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info(f"Could not load raw data for overview: {e}")

    st.markdown("---")
    st.caption("Built with Streamlit • Model trained on the IBM Telco Customer Churn dataset")


if __name__ == "__main__":
    main()
