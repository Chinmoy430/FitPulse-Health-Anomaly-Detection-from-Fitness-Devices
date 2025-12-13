import streamlit as st
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="FitPulse Health Anomaly Detection",
    layout="wide"
)

st.title("FitPulse Health Anomaly Detection")
st.subheader("Milestone 1 – Cleaned Fitness Dataset Visualization")

DATA_PATH = "data/cleaned_dataset.csv"

# Check file exists
if not os.path.exists(DATA_PATH):
    st.error("❌ cleaned_dataset.csv not found in data folder")
else:
    df = pd.read_csv(DATA_PATH)

    st.markdown("## 📊 Dataset Preview")
    st.dataframe(df.head(20))

    st.markdown("## ℹ️ Dataset Summary")
    st.write(df.describe())

    st.markdown("## 🧱 Dataset Columns")
    st.write(df.columns.tolist())
