import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.set_page_config(page_title="MSME Loan Dashboard", layout="wide")

# @st.cache_data
def load_data():
    # df = pd.read_csv("./data/simulated_msmeloans_data.csv")
    np.random.seed(42)
    n_samples = 100
    simulated_df = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'State': np.random.choice(['Lagos', 'Abuja', 'Kano', 'Rivers'], n_samples),
        'Sector': np.random.choice(['Agriculture', 'Manufacturing', 'Trade and Commerce', 'ICT'], n_samples),
        'AmountGranted': np.random.randint(100_000, 10_000_000, n_samples),
        'Tenor': np.random.randint(1, 60, n_samples),
        'Rate': np.round(np.random.uniform(5, 25, n_samples), 2),
        'TenorType': 'M',
        'ScheduleType': np.random.choice(['Monthly', 'Quarterly'], n_samples),
        'StartUp': np.random.choice(['Yes', 'No'], n_samples),
        'FirstTimeAccessToCredit': np.random.choice(['Yes', 'No'], n_samples),
        'Age': np.random.randint(21, 65, n_samples),
        'MSMEAnnualTurnover': np.random.randint(500_000, 20_000_000, n_samples),
        'NumberOfEmployees': np.random.randint(1, 200, n_samples),
        'green_energy/energy_efficiency': np.random.choice(['Yes', 'No'], n_samples),
        'InterestRate': np.round(np.random.uniform(5, 30, n_samples), 2),
        'AmountRepaid': np.random.randint(50_000, 9_000_000, n_samples),
        # 'AmountOutstanding': lambda x: x['AmountGranted'] - x['AmountRepaid'],
    })

    # Create a DefaultRisk label based on a simple heuristic
    simulated_df['DefaultRisk'] = (
        (simulated_df['AmountGranted'] > 5_000_000).astype(int) +
        (simulated_df['Rate'] > 15).astype(int) +
        (simulated_df['StartUp'] == 'Yes').astype(int)
    )
    # Mark as default risk if 2 or more risk factors are true
    simulated_df['DefaultRisk'] = (simulated_df['DefaultRisk'] >= 2).astype(int)

    return simulated_df

@st.cache_resource
def train_model(data):
    df = data.copy()
    df = df.dropna()

    features = [
        'Gender', 'Sector', 'State', 'StartUp', 'Tenor', 'TenorType',
        'AmountGranted', 'Rate', 'MSMEAnnualTurnover',
        'Age', 'NumberOfEmployees', 'green_energy/energy_efficiency'
    ]

    X = df[features]
    y = df['DefaultRisk']

    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Load data
st.title("📊 MSME Loan Analytics & Default Prediction")
df = load_data()
model = train_model(df)

# Sidebar filters
st.sidebar.header("🔍 Filter MSMEs")
sector = st.sidebar.selectbox("Select Sector", options=["All"] + sorted(df['Sector'].unique().tolist()))
state = st.sidebar.selectbox("Select State", options=["All"] + sorted(df['State'].unique().tolist()))
gender = st.sidebar.selectbox("Select Gender", options=["All"] + sorted(df['Gender'].unique().tolist()))

# Filter data
filtered_df = df.copy()
if sector != "All":
    filtered_df = filtered_df[filtered_df['Sector'] == sector]
if state != "All":
    filtered_df = filtered_df[filtered_df['State'] == state]
if gender != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == gender]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Amount Granted", f"₦{filtered_df['AmountGranted'].sum():,.0f}")
col2.metric("Avg. Interest Rate", f"{filtered_df['InterestRate'].mean():.2f}%")
col3.metric("Total MSMEs", f"{len(filtered_df)}")

# Visualizations
st.subheader("📈 Loan Distribution Insights")
col4, col5 = st.columns(2)
with col4:
    st.plotly_chart(px.histogram(filtered_df, x='Sector', color='Gender', barmode='group', title="Loans by Sector"), use_container_width=True)
with col5:
    st.plotly_chart(px.histogram(filtered_df, x='State', color='StartUp', barmode='group', title="Loans by State"), use_container_width=True)

# Prediction form
st.subheader("🤖 Predict Loan Default Risk")
st.markdown("Enter MSME details to predict risk of default")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)
    gender = col1.selectbox("Gender", ["Male", "Female"])
    sector = col2.selectbox("Sector", sorted(df['Sector'].unique()))
    state = col3.selectbox("State", sorted(df['State'].unique()))
    
    startup = col1.selectbox("Is Startup?", ["Yes", "No"])
    tenor = col2.slider("Loan Tenor (days)", 30, 1095, 365)
    amount = col3.number_input("Amount Granted (₦)", min_value=10000, max_value=50000000, value=500000)
    
    rate = col1.slider("Interest Rate (%)", 5.0, 30.0, 15.0)
    turnover = col2.number_input("Annual Turnover (₦)", min_value=100000.0, value=5000000.0)
    age = col3.slider("Business Age (years)", 1, 30, 5)
    employees = col1.slider("Number of Employees", 1, 100, 5)
    green = col2.selectbox("Green Energy MSME?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Sector': sector,
            'State': state,
            'Startup': startup,
            'TenorInDays': tenor,
            'AmountGranted': amount,
            'InterestRate': rate,
            'MSMEAnnualTurnover': turnover,
            'Age': age,
            'NumberOfEmployees': employees,
            'GreenEnergy': green
        }])

        for col in input_df.select_dtypes(include='object').columns:
            input_df[col] = LabelEncoder().fit_transform(input_df[col])

        pred = model.predict_proba(input_df)[0][1]
        st.success(f"Predicted Risk of Default: {pred:.2%}")

# Data Table
st.subheader("🧾 MSME Data Table")
st.dataframe(filtered_df, use_container_width=True)
