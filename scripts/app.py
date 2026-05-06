
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "telco" / "telco_pipeline.pkl"

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("Telco Customer Churn Prediction")

st.write(
    "Upload a Telco customer CSV file and the model will predict churn probability."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def clean_telco_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"], errors="ignore")
    df = df.drop(columns=["Churn"], errors="ignore")
    return df

model = load_model()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Input Data")
    st.dataframe(df.head())

    clean_df = clean_telco_input(df)

    st.subheader("Cleaned Data")
    st.dataframe(clean_df.head())

    proba = model.predict_proba(clean_df)[:, 1]
    result_df = df.copy()
    result_df["churn_probability"] = proba
    result_df["predicted_churn"] = (proba >= 0.5).astype(int)

    st.subheader("Predictions")
    st.dataframe(result_df.head(20))

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="telco_churn_predictions.csv",
        mime="text/csv",
    )