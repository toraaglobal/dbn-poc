import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from datetime import date
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import base64

# Load data
df = pd.read_csv("./data/enriched_financial_data.csv", parse_dates=["date"])

# Title
st.title("📊 Advanced Budget & Financial Forecasting Dashboard")

# Sidebar options
st.sidebar.title("Options")

# Prophet tuning
st.sidebar.markdown("### 🔧 Prophet Model Settings")
seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=0)
changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05)

# Date filtering
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.slider(
    "Select Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY"
)
filtered_df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

# Select variable to forecast
target_variable = st.selectbox(
    "📈 Select a variable to forecast:",
    [
        "Profit Before Tax (PBT)",
        "Disbursement (NGN)",
        "Job Creation Statistics",
        "MSME Capacity Building"
    ]
)

# Display raw data
st.subheader("📅 Filtered Data")
st.dataframe(filtered_df)

# Forecast
st.subheader(f"📈 Forecasting: {target_variable}")
data = filtered_df[["date", target_variable]].rename(columns={"date": "ds", target_variable: "y"})

model = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
model.fit(data)

future = model.make_future_dataframe(periods=5, freq="Y")
forecast = model.predict(future)

fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Residuals
st.subheader("📉 Residuals and Uncertainty")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"] - forecast["yhat_lower"],
                          name="Prediction Interval Width"))
st.plotly_chart(fig2)

# PDF export
if st.sidebar.button("📄 Export Forecast Report to PDF"):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        for variable in ["Profit Before Tax (PBT)", "Disbursement (NGN)", "Job Creation Statistics", "MSME Capacity Building"]:
            st.write(f"Exporting {variable}...")
            sub_df = filtered_df[["date", variable]].rename(columns={"date": "ds", variable: "y"})
            mod = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
            mod.fit(sub_df)
            future = mod.make_future_dataframe(periods=5, freq="Y")
            forecast = mod.predict(future)
            fig = mod.plot(forecast)
            fig.suptitle(f"Forecast for {variable}")
            pdf.savefig(fig)
            plt.close(fig)

    buffer.seek(0)
    b64_pdf = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="forecast_report.pdf">📥 Download Forecast Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)
