
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Advanced Budget & Financial Forecasting Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/enriched_financial_data.csv", parse_dates=["date"])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
min_date = df["date"].min().date()
max_date = df["date"].max().date()

# Slider expects date objects
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY"
)

# Apply date filter using .dt.date
filtered_df = df[
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1])
]

# Plot economic indicators
st.subheader("📉 Economic Indicators Over Time")
indicators = ["Inflation Rate (%)", "Exchange Rate (NGN/USD)", "GDP Growth Rate (%)", "Interest Rate (%)", "Oil Price (USD/barrel)"]
selected_indicator = st.selectbox("Select Economic Indicator:", indicators)
fig = px.line(filtered_df, x="date", y=selected_indicator, markers=True, title=f"{selected_indicator} Over Time")
st.plotly_chart(fig, use_container_width=True)

# Forecasting section
def run_forecast(target_column, periods=3):
    data = df[["date", target_column]].rename(columns={"date": "ds", target_column: "y"})
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)

    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Evaluation
    y_true = data["y"]
    y_pred = model.predict(data)["yhat"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"**MAE:** {mae:,.2f} | **RMSE:** {rmse:,.2f} | **R²:** {r2:.2f}")

    # Anomaly detection
    residuals = y_true - y_pred
    threshold = st.slider(f"Anomaly Sensitivity ({target_column})", 0.5, 5.0, 2.0, step=0.1)
    std_dev = np.std(residuals)
    anomalies = data[np.abs(residuals) > threshold * std_dev]
    st.dataframe(anomalies.rename(columns={"ds": "Date", "y": target_column}))
    return forecast

st.subheader("📈 Forecast: Profit Before Tax (PBT)")
run_forecast("Profit Before Tax (PBT)")

st.subheader("📈 Forecast: Disbursement (NGN)")
run_forecast("Disbursement (NGN)")

st.subheader("📈 Forecast: Job Creation Statistics")
run_forecast("Job Creation Statistics")

# Correlation heatmap

st.subheader("📈 Forecast: MSME Capacity Building")
run_forecast("MSME Capacity Building")

st.subheader("📌 Correlation Heatmap")
corr = df.select_dtypes(include='number').corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)
