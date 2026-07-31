"""
data_prep.py — Data Cleaning & Preprocessing for Telco Customer Churn Dataset

This module handles:
- Loading raw data
- Handling missing / blank TotalCharges values
- Converting data types
- Encoding categorical variables (one-hot encoding)
- Saving the cleaned dataset to data/processed/cleaned_churn.csv
"""

import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv")
    df = pd.read_csv(filepath)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco churn dataset:
    - Drop customerID (not a predictive feature)
    - Fix blank TotalCharges values (some rows have ' ' instead of a number)
    - Convert TotalCharges to numeric
    - Encode the target variable Churn as 0/1
    - Convert SeniorCitizen to categorical string for consistency
    """
    df = df.copy()

    # Drop customerID — it's an identifier, not a feature
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle blank TotalCharges: replace blank strings with NaN, then convert to float
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0 (these are new customers with tenure=0 or 1)
    n_missing = df["TotalCharges"].isna().sum()
    print(f"TotalCharges: found {n_missing} missing/blank values — filling with 0.0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Encode target variable: 'Yes' -> 1, 'No' -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Convert SeniorCitizen from 0/1 int to 'No'/'Yes' string for uniform encoding later
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    print(f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns.
    Returns a DataFrame with only numeric columns ready for modeling.
    """
    df = df.copy()

    # Identify categorical columns (object dtype)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}")

    # One-hot encode, dropping the first level to avoid multicollinearity
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure all columns are numeric
    for col in df_encoded.columns:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors="coerce")

    print(f"Encoded data: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")
    return df_encoded


def run_pipeline(raw_path: str = None, save_path: str = None):
    """
    Run the full data preparation pipeline:
    1. Load raw data
    2. Clean data
    3. Encode features
    4. Save to processed CSV
    Returns the cleaned & encoded DataFrame.
    """
    if raw_path is None:
        raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv")
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")

    df = load_raw_data(raw_path)
    df_clean = clean_data(df)
    df_encoded = encode_features(df_clean)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_encoded.to_csv(save_path, index=False)
    print(f"Saved processed data to {save_path}")

    return df_encoded


if __name__ == "__main__":
    run_pipeline()
