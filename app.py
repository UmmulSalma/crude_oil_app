import streamlit as st
import pandas as pd
import numpy as np
import joblib
from darts import TimeSeries
from darts.models import TiDEModel

st.set_page_config(page_title="Crude Oil Forecaster", layout="centered")
st.title("🛢️ Crude Oil Production Forecaster")

@st.cache_resource
def load_artifacts():
    tide_model = TiDEModel.load("tide_final.pt")
    svr_corrector = joblib.load("tide_svr_corrector.pkl")
    scaler_target = joblib.load("tide_target_scaler.pkl")
    scaler_covs = joblib.load("tide_cov_scaler.pkl")
    return tide_model, svr_corrector, scaler_target, scaler_covs

tide_model, svr_corrector, scaler_target, scaler_covs = load_artifacts()

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df.tail())
    
    if st.button("Generate Forecast"):
        # Prepare Data
        df['Time'] = pd.date_range(start="2000-01-01", periods=len(df), freq='MS')
        series = TimeSeries.from_dataframe(df, time_col='Time', value_cols='Production')
        covariates = TimeSeries.from_dataframe(df, time_col='Time', value_cols=['Lag1', 'Lag2', 'oil price'])
        
        # Scale Data
        series_scaled = scaler_target.transform(series)
        covs_scaled = scaler_covs.transform(covariates)
        
        # TiDE Prediction
        tide_pred_scaled = tide_model.predict(n=1, series=series_scaled, past_covariates=covs_scaled)
        tide_pred = scaler_target.inverse_transform(tide_pred_scaled)
        tide_pred_val = tide_pred.values().flatten()[0]
        
        # SVR Error Correction
        last_covs = df[['Lag1', 'Lag2', 'oil price']].iloc[-1].values
        svr_features = np.append(last_covs, tide_pred_val).reshape(1, -1)
        
        correction = svr_corrector.predict(svr_features)[0]
        hybrid_pred = tide_pred_val + correction
        
        st.success(f"Predicted Next Month Production: {hybrid_pred:.4f}")