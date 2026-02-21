import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import TiDEModel

# --- FIX FOR PYTORCH weights_only ERROR ---
import torch
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# ------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Crude Oil Forecaster", layout="centered")
st.title("🛢️ Crude Oil Production Forecaster")
st.write("Hybrid TiDE + SVR model to forecast future crude oil production.")

# --- Load Models & Artifacts ---
@st.cache_resource
def load_artifacts():
    tide_model = TiDEModel.load("tide_final.pt")
    svr_corrector = joblib.load("tide_svr_corrector.pkl")
    scaler_target = joblib.load("tide_target_scaler.pkl")
    scaler_covs = joblib.load("tide_cov_scaler.pkl")
    return tide_model, svr_corrector, scaler_target, scaler_covs

try:
    tide_model, svr_corrector, scaler_target, scaler_covs = load_artifacts()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models. Make sure all .pt and .pkl files are in the same folder as app.py. Details: {e}")
    st.stop()

# --- File Upload ---
st.subheader("Upload Recent Data")
st.write("Upload an Excel file containing `Production`, `Lag1`, `Lag2`, and `oil price` columns to predict the next month.")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("**Data Preview:**")
    st.dataframe(df.tail())
    
    if st.button("Generate Forecast"):
        with st.spinner("Running Hybrid Model..."):
            try:
                # 1. Prepare Data
                df['Time'] = pd.date_range(start="2000-01-01", periods=len(df), freq='MS')
                series = TimeSeries.from_dataframe(df, time_col='Time', value_cols='Production')
                covariates = TimeSeries.from_dataframe(df, time_col='Time', value_cols=['Lag1', 'Lag2', 'oil price'])
                
                # 2. Scale Data
                series_scaled = scaler_target.transform(series)
                covs_scaled = scaler_covs.transform(covariates)
                
                # 3. TiDE Prediction (Predict next 1 month)
                tide_pred_scaled = tide_model.predict(n=1, series=series_scaled, past_covariates=covs_scaled)
                tide_pred = scaler_target.inverse_transform(tide_pred_scaled)
                tide_pred_val = tide_pred.values().flatten()[0]
                
                # 4. SVR Error Correction
                last_covs = df[['Lag1', 'Lag2', 'oil price']].iloc[-1].values
                svr_features = np.append(last_covs, tide_pred_val).reshape(1, -1)
                
                correction = svr_corrector.predict(svr_features)[0]
                hybrid_pred = tide_pred_val + correction
                
                st.success(f"### 📈 Predicted Next Month Production: {hybrid_pred:.4f}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
