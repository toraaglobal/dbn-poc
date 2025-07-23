import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from forecast import (
    load_data,
    forecast_revenue,
    forecast_costs,
    inflation_adjusted_budget,
    evaluate_forecast_model,
    plot_residuals,
    cross_validation_report,
    detect_anomalies,
    export_excel_report
)

import io
import base64

st.set_page_config(page_title="Advanced Financial Forecasting Dashboard", layout="wide")
st.title("📊 Advanced Budget & Financial Forecasting Dashboard")

# Upload economic data
uploaded_file = st.file_uploader("Upload CSV of economic variables:", type="csv")
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Economic Variables", data.head())

    # Correlation Heatmap
    with st.expander("📌 Correlation Heatmap"):
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        corr = numeric_data.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        buf_corr = io.BytesIO()
        fig_corr.savefig(buf_corr, format="png")
        st.download_button("Download Correlation Heatmap", data=buf_corr.getvalue(), file_name="correlation_heatmap.png")

    # Scatter Matrix
    with st.expander("📌 Pairwise Scatter Plot Matrix"):
        selected_cols = st.multiselect("Select numeric variables to compare:", numeric_data.columns.tolist(), default=numeric_data.columns.tolist())
        if selected_cols:
            sns.set(style="ticks")
            fig_scat = sns.pairplot(data[selected_cols])
            st.pyplot(fig_scat)

            buf_scat = io.BytesIO()
            fig_scat.savefig(buf_scat, format="png")
            st.download_button("Download Scatter Plot Matrix", data=buf_scat.getvalue(), file_name="scatter_matrix.png")

    forecast_horizon = st.slider("Select forecast horizon (months):", 1, 24, 12)

    with st.expander("Revenue Forecast"):
        revenue_df, revenue_model, revenue_train = forecast_revenue(data, horizon=forecast_horizon, return_model=True)
        st.line_chart(revenue_df.set_index("date")["yhat"])
        st.download_button("Download Revenue Forecast", revenue_df.to_csv(index=False), file_name="revenue_forecast.csv")

        st.subheader("Forecast Components")
        fig1 = revenue_model.plot_components(revenue_model.predict(revenue_model.make_future_dataframe(periods=forecast_horizon, freq='MS')))
        st.pyplot(fig1)

        buf_rev = io.BytesIO()
        fig1.savefig(buf_rev, format="png")
        st.download_button("Download Revenue Components Plot", data=buf_rev.getvalue(), file_name="revenue_components.png")

        st.subheader("Forecast Evaluation")
        metrics_rev = evaluate_forecast_model(revenue_model, revenue_train)
        st.write(metrics_rev)

        st.subheader("Residual Plot")
        fig_resid = plot_residuals(revenue_model, revenue_train)
        st.pyplot(fig_resid)

        st.subheader("Cross-Validation Report")
        cv_df_rev = cross_validation_report(revenue_model)
        st.write(cv_df_rev)

        st.subheader("🔍 Anomaly Detection")
        threshold_rev = st.slider("Revenue Anomaly Sensitivity (Std Dev Threshold):", 0.5, 5.0, 2.0, step=0.1)
        anomalies_rev = detect_anomalies(revenue_model, revenue_train, threshold=threshold_rev)
        st.write(anomalies_rev)

    with st.expander("Cost Forecast"):
        cost_df, cost_model, cost_train = forecast_costs(data, horizon=forecast_horizon, return_model=True)
        st.line_chart(cost_df.set_index("date")["yhat"])
        st.download_button("Download Cost Forecast", cost_df.to_csv(index=False), file_name="cost_forecast.csv")

        st.subheader("Forecast Components")
        fig2 = cost_model.plot_components(cost_model.predict(cost_model.make_future_dataframe(periods=forecast_horizon, freq='MS')))
        st.pyplot(fig2)

        buf_cost = io.BytesIO()
        fig2.savefig(buf_cost, format="png")
        st.download_button("Download Cost Components Plot", data=buf_cost.getvalue(), file_name="cost_components.png")

        st.subheader("Forecast Evaluation")
        metrics_cost = evaluate_forecast_model(cost_model, cost_train)
        st.write(metrics_cost)

        st.subheader("Residual Plot")
        fig_resid_cost = plot_residuals(cost_model, cost_train)
        st.pyplot(fig_resid_cost)

        st.subheader("Cross-Validation Report")
        cv_df_cost = cross_validation_report(cost_model)
        st.write(cv_df_cost)

        st.subheader("🔍 Anomaly Detection")
        threshold_cost = st.slider("Cost Anomaly Sensitivity (Std Dev Threshold):", 0.5, 5.0, 2.0, step=0.1)
        anomalies_cost = detect_anomalies(cost_model, cost_train, threshold=threshold_cost)
        st.write(anomalies_cost)

    with st.expander("Inflation-Adjusted Budget"):
        adj_budget = inflation_adjusted_budget(revenue_df, cost_df, data)
        st.write("### Inflation Adjusted Budget")
        st.dataframe(adj_budget)
        st.download_button("Download Inflation-Adjusted Budget", adj_budget.to_csv(index=False), file_name="inflation_adjusted_budget.csv")

    with st.expander("📤 Export Excel Report"):
        if st.button("Generate and Download Excel Report"):
            output_excel = export_excel_report(data, revenue_df, cost_df, adj_budget, metrics_rev, metrics_cost)
            st.download_button(
                label="📥 Download Excel Report",
                data=output_excel.getvalue(),
                file_name="financial_forecast_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )