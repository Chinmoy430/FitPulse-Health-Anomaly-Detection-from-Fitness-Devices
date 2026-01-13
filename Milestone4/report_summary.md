
# Milestone 4: Dashboard for Insights (FitPulse Health Anomaly Detection)

## Objective
The objective of Milestone 4 is to develop an interactive Streamlit dashboard that provides
health anomaly insights from fitness device data. The dashboard integrates the complete workflow
from earlier milestones, including preprocessing, trend modeling, anomaly detection, interactive visualization,
and report generation.

---

## Dashboard Workflow
The dashboard performs the following workflow:

1. **Data Upload**
   - Users can upload fitness data files (CSV/JSON) through the Streamlit dashboard executed in Google Colab.

2. **Data Integration**
   - The uploaded data is processed and used to run the pipeline developed in previous milestones.
   - The dashboard supports dynamic execution of anomaly detection.

3. **Trend Modeling and Anomaly Detection**
   - Prophet is used to model expected trends for selected metrics such as heart rate, steps, and sleep.
   - Residuals are calculated as:
     - residual = actual value − predicted value
   - Anomalies are detected using a threshold-based rule:
     - Anomaly if |residual| > 2 × standard deviation of residuals

4. **Interactive Visualization**
   - Metric-wise visualizations are displayed with anomaly markers highlighted.
   - Filtering options allow:
     - User-wise selection
     - Metric-wise selection (Heart Rate, Steps, Sleep)
     - Date-wise filtering for focused insights

5. **Report Generation**
   - The dashboard generates a downloadable anomaly summary report in CSV format.
   - The report includes:
     - Metric name
     - Timestamp/date
     - Actual value
     - Predicted value
     - Residual value
     - Anomaly label

---

## Tools and Libraries Used
- Python
- Streamlit
- Pandas
- NumPy
- Facebook Prophet
- Matplotlib / Plotly (for visualizations)
- Google Colaboratory
- Ngrok (to run Streamlit dashboard in Colab)
- Font Awesome (for icons)
- CSS for styling
---

## Key Insights from the Dashboard
- Heart rate trends show deviations which can indicate possible health anomalies.
- Sleep pattern visualization highlights abnormal sleep segments when residual deviations are high.
- Step count analysis helps observe changes in activity behavior and detect unusual patterns.
- The anomaly report provides metric-level and timestamp-level insights for easy tracking and validation.

---

## Screenshot References
Screenshots are saved inside the `screenshots/` folder:
- `dashboard_ui.png` : Streamlit dashboard UI with filters and interactive anomaly visualizations
- `report_download.png` : Downloadable anomaly report generation output

