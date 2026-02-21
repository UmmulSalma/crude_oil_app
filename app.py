import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from darts import TimeSeries
from darts.models import TiDEModel

# --- FIX FOR PYTORCH weights_only ERROR ---
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
# ------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Crude Oil Forecaster", layout="centered")
st.title("🛢️ Crude Oil Production Forecaster")
st.write("Enter the latest monthly data to forecast the following month.")

# --- Load Models & Artifacts ---
@st.cache_resource
def load_artifacts():
    tide_model = TiDEModel.load("tide_final.pt")
    svr_corrector = joblib.load("tide_svr_corrector.pkl")
    scaler_target = joblib.load("tide_target_scaler.pkl")
    scaler_covs = joblib.load("tide_cov_scaler.pkl")
    return tide_model, svr_corrector, scaler_target, scaler_covs

# --- Load Background History ---
@st.cache_data
def load_history():
    # Automatically loads the background data so the user doesn't have to upload it!
    return pd.read_excel("historical_data.xlsx")

try:
    tide_model, svr_corrector, scaler_target, scaler_covs = load_artifacts()
    df_history = load_history()
except Exception as e:
    st.error(f"Error loading files. Ensure your models and 'historical_data.xlsx' are in the folder. Details: {e}")
    st.stop()

# --- User Input Section ---
st.subheader("Input Current Month's Data")
st.write("Please enter the values for the **latest** month below to predict the **next** month's production.")

# Create a 2-column layout for the input fields
col1, col2 = st.columns(2)

with col1:
    # Pre-filling with the last known values from the history for convenience
    prod_input = st.number_input("Current Production", value=float(df_history['Production'].iloc[-1]))
    lag1_input = st.number_input("Lag 1", value=float(df_history['Lag1'].iloc[-1]))

with col2:
    lag2_input = st.number_input("Lag 2", value=float(df_history['Lag2'].iloc[-1]))
    oil_price_input = st.number_input("Oil Price", value=float(df_history['oil price'].iloc[-1]))

# --- Forecasting ---
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Running Hybrid Model..."):
        try:
            # 1. Append user input to the historical data
            new_row = pd.DataFrame({
                'Production': [prod_input],
                'Lag1': [lag1_input],
                'Lag2': [lag2_input],
                'oil price': [oil_price_input]
            })
            df = pd.concat([df_history, new_row], ignore_index=True)
            
            # 2. Prepare Data for Darts
            df['Time'] = pd.date_range(start="2000-01-01", periods=len(df), freq='MS')
            series = TimeSeries.from_dataframe(df, time_col='Time', value_cols='Production')
            covariates = TimeSeries.from_dataframe(df, time_col='Time', value_cols=['Lag1', 'Lag2', 'oil price'])
            
            # 3. Scale Data
            series_scaled = scaler_target.transform(series)
            covs_scaled = scaler_covs.transform(covariates)
            
            # 4. TiDE Prediction (Predict next 1 month)
            tide_pred_scaled = tide_model.predict(n=1, series=series_scaled, past_covariates=covs_scaled)
            tide_pred = scaler_target.inverse_transform(tide_pred_scaled)
            tide_pred_val = tide_pred.values().flatten()[0]
            
            # 5. SVR Error Correction
            last_covs = np.array([lag1_input, lag2_input, oil_price_input])
            svr_features = np.append(last_covs, tide_pred_val).reshape(1, -1)
            
            correction = svr_corrector.predict(svr_features)[0]
            hybrid_pred = tide_pred_val + correction
            
            st.success(f"### 📈 Predicted Next Month Production: {hybrid_pred:.4f}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
