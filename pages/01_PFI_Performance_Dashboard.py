import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="PFI Performance & Impact Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("./data/simulated_msmeloans_data.csv", parse_dates=["EffectiveDate", "DateAdded"])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
selected_pfi = st.sidebar.multiselect("Select PFI(s)", options=df["CusID"].unique(), default=df["CusID"].unique())
selected_sector = st.sidebar.multiselect("Select Sector(s)", options=df["Sector"].unique(), default=df["Sector"].unique())

df_filtered = df[df["CusID"].isin(selected_pfi) & df["Sector"].isin(selected_sector)]

st.title("📊 PFI Performance & Impact Dashboard")

# Summary KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Disbursed Amount", f"₦{df_filtered['AmountGranted'].sum():,.2f}")
col2.metric("Average Interest Rate", f"{df_filtered['Rate'].mean():.2f}%")
col3.metric("Average Tenor (Days)", f"{df_filtered['TenorInDays'].mean():.0f}")

st.markdown("---")

# Inclusion Metrics
st.subheader("👩🏾‍🤝‍👨🏼 Inclusion Metrics")
gender_dist = df_filtered["Gender"].value_counts().reset_index()
gender_dist.columns = ["Gender", "count"]
age_group = df_filtered["age_group_id"].value_counts().reset_index()
age_group.columns = ["YouthStatus", "Count"]

col4, col5 = st.columns(2)
with col4:
    fig_gender = px.pie(gender_dist, names="Gender", values="count", title="Gender Distribution")
    st.plotly_chart(fig_gender, use_container_width=True)

with col5:
    fig_age = px.pie(age_group, names="YouthStatus", values="Count", title="Youth Inclusion")
    st.plotly_chart(fig_age, use_container_width=True)

# Green Energy Loans
st.subheader("🌱 Green Energy Financing")
green_df = df_filtered["green_energy/energy_efficiency"].value_counts().reset_index()
green_df.columns = ["GreenEnergy", "Count"]
fig_green = px.bar(green_df, x="GreenEnergy", y="Count", color="GreenEnergy", title="Green Loans Distribution")
st.plotly_chart(fig_green, use_container_width=True)

# Time Series Trends
st.subheader("📈 Disbursement Trends Over Time")
df_ts = df_filtered.copy()
df_ts["Month"] = df_ts["EffectiveDate"].dt.to_period("M").astype(str)
month_summary = df_ts.groupby("Month")["AmountGranted"].sum().reset_index()
fig_ts = px.line(month_summary, x="Month", y="AmountGranted", title="Monthly Disbursement Volume")
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ for DBN and regulators by Tajudeen Abdulazeez")
