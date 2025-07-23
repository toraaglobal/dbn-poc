import pandas as pd
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
from io import BytesIO
import xlsxwriter

def load_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    return df

def forecast_revenue(df, horizon=12, return_model=False):
    model_df = df[['date', 'revenue']].rename(columns={'date': 'ds', 'revenue': 'y'})
    model = Prophet()
    model.fit(model_df)
    future = model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'date'})
    if return_model:
        return forecast_df, model, model_df
    return forecast_df

def forecast_costs(df, horizon=12, return_model=False):
    if 'cost' not in df.columns:
        df['cost'] = df['revenue'] * 0.6  # simulate cost as 60% of revenue
    model_df = df[['date', 'cost']].rename(columns={'date': 'ds', 'cost': 'y'})
    model = Prophet()
    model.fit(model_df)
    future = model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'date'})
    if return_model:
        return forecast_df, model, model_df
    return forecast_df

def inflation_adjusted_budget(revenue_df, cost_df, original_df):
    merged = revenue_df.merge(cost_df, on='date', suffixes=('_revenue', '_cost'))
    cpi = original_df[['date', 'cpi']].set_index('date').resample('MS').mean().ffill()
    merged = merged.set_index('date').join(cpi).reset_index()
    merged['real_revenue'] = merged['yhat_revenue'] / (1 + merged['cpi'] / 100)
    merged['real_cost'] = merged['yhat_cost'] / (1 + merged['cpi'] / 100)
    merged['real_net_budget'] = merged['real_revenue'] - merged['real_cost']
    return merged[['date', 'real_revenue', 'real_cost', 'real_net_budget']]

def evaluate_forecast_model(model, train_df):
    forecast_train = model.predict(train_df[['ds']])
    y_true = train_df['y'].values
    y_pred = forecast_train['yhat'].values
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

def plot_residuals(model, train_df):
    forecast_train = model.predict(train_df[['ds']])
    residuals = train_df['y'] - forecast_train['yhat']
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df['ds'], residuals, marker='o', linestyle='-', label='Residuals')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title("Residual Plot")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    return fig

def cross_validation_report(model, horizon="180 days"):
    df_cv = cross_validation(model, horizon=horizon)
    df_p = performance_metrics(df_cv)
    return df_p


def detect_anomalies(model, train_df, threshold=2.0):
    forecast_train = model.predict(train_df[['ds']])
    residuals = train_df['y'] - forecast_train['yhat']
    std = np.std(residuals)
    mean = np.mean(residuals)
    anomalies = train_df[(np.abs(residuals - mean) > threshold * std)]
    anomalies = anomalies.copy()
    anomalies['residual'] = residuals[anomalies.index]
    return anomalies



def export_excel_report(data, revenue_df, cost_df, adj_budget, metrics_rev, metrics_cost):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    data.to_excel(writer, sheet_name='Input Data', index=False)
    revenue_df.to_excel(writer, sheet_name='Revenue Forecast', index=False)
    cost_df.to_excel(writer, sheet_name='Cost Forecast', index=False)
    adj_budget.to_excel(writer, sheet_name='Inflation Adjusted', index=False)
    pd.DataFrame([metrics_rev]).to_excel(writer, sheet_name='Revenue Metrics', index=False)
    pd.DataFrame([metrics_cost]).to_excel(writer, sheet_name='Cost Metrics', index=False)

    # writer.save()
    output.seek(0)
    return output