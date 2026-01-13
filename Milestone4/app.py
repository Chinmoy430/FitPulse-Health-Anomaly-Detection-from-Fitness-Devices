import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet

# Page Configuration with improved theme
st.set_page_config(
    page_title="FitPulse Dashboard",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF9A3D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2C3E50;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498DB;
    }
    .sub-header i {
        margin-right: 10px;
        color: #3498DB;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .info-box i {
        font-size: 1.2rem;
        margin-right: 8px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3498DB;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    .metric-card i {
        font-size: 1.5rem;
        margin-bottom: 10px;
        color: #3498DB;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3498DB, #2ECC71);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
    .stSelectbox, .stDateInput {
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }
    .success-box {
        background: linear-gradient(135deg, #2ECC71, #27AE60);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    .success-box i {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .warning-box {
        background: linear-gradient(135deg, #FF9A3D, #FF6B6B);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    .warning-box i {
        font-size: 1.5rem;
        margin-right: 10px;
    }
    .icon-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .icon-container i {
        font-size: 1.2rem;
        margin-right: 10px;
        color: #3498DB;
        width: 25px;
    }
    .status-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3498DB, #2ECC71);
        color: white;
        margin-right: 10px;
    }
    .footer-icons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 10px;
    }
    .footer-icons i {
        font-size: 1.2rem;
        color: #3498DB;
        transition: all 0.3s;
    }
    .footer-icons i:hover {
        color: #FF4B4B;
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<h1 class="main-header"><i class="fas fa-heartbeat"></i> FitPulse Health Anomaly Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #7F8C8D; font-size: 1.1rem; margin-bottom: 2rem;'>
    <div class="icon-container">
        <i class="fas fa-upload"></i>
        <span>Upload your fitness data to detect anomalies and gain insights into your health patterns.</span>
    </div>
    <div class="icon-container">
        <i class="fas fa-chart-line"></i>
        <span>Get visual analytics and downloadable reports for better health monitoring.</span>
    </div>
</div>
""", unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="sub-header"><i class="fas fa-file-upload"></i> Data Upload</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your cleaned fitness dataset in CSV format"
    )
with col2:
    st.markdown("""
    <div class="info-box">
        <h4 style='margin:0;'><i class="fas fa-clipboard-list"></i> Data Requirements</h4>
        <p style='margin:0.5rem 0;'><i class="fas fa-check-circle"></i> CSV format required</p>
        <p style='margin:0.5rem 0;'><i class="fas fa-check-circle"></i> Must include 'timestamp' column</p>
        <p style='margin:0.5rem 0;'><i class="fas fa-check-circle"></i> Must include 'Id' column for users</p>
        <p style='margin:0.5rem 0;'><i class="fas fa-check-circle"></i> Supports heart_rate, steps, sleep metrics</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        <h3 style='margin:0; text-align: center;'><i class="fas fa-file-excel"></i> No File Uploaded</h3>
        <p style='text-align: center; margin: 0.5rem 0;'>Please upload a CSV file to begin analysis</p>
        <div style='text-align: center; margin-top: 1rem;'>
            <i class="fas fa-arrow-up" style='font-size: 2rem; opacity: 0.7;'></i>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load Data
try:
    df = pd.read_csv(uploaded_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    st.markdown('<div class="success-box"><i class="fas fa-check-circle"></i> Dataset loaded successfully!</div>', unsafe_allow_html=True)
    
    # Display data preview with metrics
    st.markdown('<div class="sub-header"><i class="fas fa-chart-bar"></i> Data Preview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <i class="fas fa-database"></i>
            <div style='font-size: 0.9rem; color: #7F8C8D;'>Total Records</div>
            <div style='font-size: 2rem; font-weight: 700; color: #3498DB;'>{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <i class="fas fa-users"></i>
            <div style='font-size: 0.9rem; color: #7F8C8D;'>Unique Users</div>
            <div style='font-size: 2rem; font-weight: 700; color: #2ECC71;'>{df['Id'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        st.markdown(f"""
        <div class="metric-card">
            <i class="fas fa-calendar-alt"></i>
            <div style='font-size: 0.9rem; color: #7F8C8D;'>Date Range</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: #E74C3C;'>{date_range}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("View Data Sample", expanded=False):
        st.markdown('<div style="margin-bottom: 10px;"><i class="fas fa-eye"></i> <strong>Preview of your data:</strong></div>', unsafe_allow_html=True)
        st.dataframe(df.head(10).style.background_gradient(subset=['heart_rate', 'steps', 'sleep'], cmap='YlOrRd'))
        
except Exception as e:
    st.error(f"<i class='fas fa-exclamation-triangle'></i> Error loading file: {str(e)}", unsafe_allow_html=True)
    st.stop()

# Analysis Configuration
st.markdown('<div class="sub-header"><i class="fas fa-cogs"></i> Analysis Configuration</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    user_ids = df["Id"].unique()
    selected_user = st.selectbox(
        "👤 Select User",
        user_ids,
        help="Choose which user's data to analyze"
    )
    
with col2:
    metrics = ["heart_rate", "steps", "sleep"]
    metric_icons = {
        "heart_rate": "❤️",
        "steps": "👣",
        "sleep": "😴"
    }
    selected_metric = st.selectbox(
        "📈 Select Metric",
        metrics,
        format_func=lambda x: f"{metric_icons[x]} {x.replace('_', ' ').title()}",
        help="Choose the health metric to analyze for anomalies"
    )
    
with col3:
    user_df = df[df["Id"] == selected_user].copy()
    min_date = user_df["timestamp"].min().date()
    max_date = user_df["timestamp"].max().date()
    date_range = st.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select the date range for analysis"
    )

# Process selected date range
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    user_df = user_df[(user_df["timestamp"].dt.date >= start_date) & (user_df["timestamp"].dt.date <= end_date)]

# Prepare data for Prophet
data = user_df[["timestamp", selected_metric]].dropna()
data = data.rename(columns={"timestamp": "ds", selected_metric: "y"})
data = data.set_index("ds").resample("D").mean().reset_index()

if len(data) < 10:
    st.markdown("""
    <div class="warning-box">
        <i class="fas fa-exclamation-triangle"></i> Not enough data points for reliable anomaly detection.
        Please select a user with at least 10 days of data.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run Prophet Model
st.markdown('<div class="sub-header"><i class="fas fa-search"></i> Running Anomaly Detection</div>', unsafe_allow_html=True)

with st.spinner('Training Prophet model and detecting anomalies...'):
    progress_bar = st.progress(0)
    
    # Simulate progress with icon
    progress_col1, progress_col2 = st.columns([1, 5])
    with progress_col1:
        st.markdown('<div style="text-align: center;"><i class="fas fa-cogs fa-spin" style="font-size: 2rem; color: #3498DB;"></i></div>', unsafe_allow_html=True)
    
    with progress_col2:
        # Simulate progress
        for i in range(100):
            progress_bar.progress(i + 1)
    
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    
    merged = data.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    merged["residual"] = merged["y"] - merged["yhat"]
    threshold = 2 * merged["residual"].std()
    merged["anomaly"] = (merged["residual"].abs() > threshold)
    
    progress_bar.empty()
    st.markdown('<div style="text-align: center; margin-top: 10px;"><i class="fas fa-check-circle" style="color: #2ECC71; font-size: 1.5rem;"></i> Model training complete!</div>', unsafe_allow_html=True)

# Display Results
st.markdown('<div class="success-box"><i class="fas fa-flag-checkered"></i> Anomaly detection completed successfully!</div>', unsafe_allow_html=True)

# Display metrics about anomalies
anomaly_count = merged["anomaly"].sum()
total_points = len(merged)
anomaly_percentage = (anomaly_count / total_points * 100) if total_points > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-chart-line"></i>
        <div style='font-size: 0.9rem; color: #7F8C8D;'>Total Data Points</div>
        <div style='font-size: 2rem; font-weight: 700; color: #3498DB;'>{total_points:,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    icon_color = "#E74C3C" if anomaly_count > 0 else "#2ECC71"
    icon_class = "fas fa-exclamation-circle" if anomaly_count > 0 else "fas fa-check-circle"
    st.markdown(f"""
    <div class="metric-card">
        <i class="{icon_class}" style="color: {icon_color};"></i>
        <div style='font-size: 0.9rem; color: #7F8C8D;'>Anomalies Detected</div>
        <div style='font-size: 2rem; font-weight: 700; color: {icon_color}'>
            {anomaly_count:,}
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    percent_icon = "fas fa-exclamation-triangle" if anomaly_percentage > 5 else "fas fa-thumbs-up"
    percent_color = "#F39C12" if anomaly_percentage > 5 else "#2ECC71"
    st.markdown(f"""
    <div class="metric-card">
        <i class="{percent_icon}" style="color: {percent_color};"></i>
        <div style='font-size: 0.9rem; color: #7F8C8D;'>Anomaly Percentage</div>
        <div style='font-size: 2rem; font-weight: 700; color: {percent_color}'>
            {anomaly_percentage:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# Visualization
st.markdown(f'<div class="sub-header"><i class="fas fa-chart-area"></i> {selected_metric.replace("_", " ").title()} Trend Analysis</div>', unsafe_allow_html=True)

# Format y-axis label based on metric
y_label = ""
if selected_metric == "heart_rate":
    y_label = "Heart Rate (BPM)"
elif selected_metric == "steps":
    y_label = "Steps Count"
elif selected_metric == "sleep":
    y_label = "Sleep Duration (Hours)"

fig = px.line(
    merged, 
    x="ds", 
    y="y", 
    title=f"{selected_metric.replace('_', ' ').title()} Trend with Anomaly Detection",
    labels={"ds": "Date", "y": y_label}
)

# Add prediction line
fig.add_scatter(
    x=merged["ds"], 
    y=merged["yhat"], 
    mode="lines", 
    name="<i class='fas fa-project-diagram'></i> Predicted",
    line=dict(color="#2ECC71", dash="dash")
)

# Add anomalies
anoms = merged[merged["anomaly"] == True]
if not anoms.empty:
    fig.add_scatter(
        x=anoms["ds"], 
        y=anoms["y"], 
        mode="markers", 
        name="<i class='fas fa-exclamation-triangle'></i> Anomaly",
        marker=dict(color="#E74C3C", size=12, symbol="diamond", line=dict(width=2, color="white"))
    )

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
    height=500,
    font=dict(family="Arial, sans-serif", size=12),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    )
)

# Update x-axis with black color and better visibility
fig.update_xaxes(
    title_text="Date",
    title_font=dict(size=14, color="black", family="Arial, sans-serif"),
    tickfont=dict(size=12, color="black", family="Arial, sans-serif"),
    gridcolor="#e0e0e0",
    gridwidth=1,
    showgrid=True,
    linecolor="black",
    linewidth=2,
    mirror=True,
    tickformat="%b %d, %Y",  # Format: Apr 17, 2016
    tickangle=0,
    tickmode="auto",
    nticks=8
)

# Update y-axis with black color and better visibility
fig.update_yaxes(
    title_text=y_label,
    title_font=dict(size=14, color="black", family="Arial, sans-serif"),
    tickfont=dict(size=12, color="black", family="Arial, sans-serif"),
    gridcolor="#e0e0e0",
    gridwidth=1,
    showgrid=True,
    linecolor="black",
    linewidth=2,
    mirror=True,
    tickmode="linear",
    dtick=5,  # Adjust based on data range
    range=[merged["y"].min() - 5, merged["y"].max() + 5]  # Add some padding
)

# Improve hover template
fig.update_traces(
    hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Value:</b> %{y:.2f}<extra></extra>"
)

st.plotly_chart(fig, use_container_width=True)

# Anomaly Report Section
st.markdown('<div class="sub-header"><i class="fas fa-file-alt"></i> Anomaly Report</div>', unsafe_allow_html=True)

report = merged[merged["anomaly"] == True][["ds", "y", "yhat", "residual"]].copy()
report = report.round(2)
report["metric"] = selected_metric.replace("_", " ").title()
report["user_id"] = selected_user
report = report.rename(columns={
    "ds": "Date",
    "y": "Actual Value",
    "yhat": "Predicted Value",
    "residual": "Deviation",
    "metric": "Metric",
    "user_id": "User ID"
})

if not report.empty:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div style="margin-bottom: 10px;"><i class="fas fa-table"></i> <strong>Detected Anomalies:</strong></div>', unsafe_allow_html=True)
        st.dataframe(
            report.style.background_gradient(
                subset=['Deviation'],
                cmap='Reds'
            ).format({
                'Actual Value': '{:.1f}',
                'Predicted Value': '{:.1f}',
                'Deviation': '{:.1f}'
            })
        )
    with col2:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;'>
            <h4 style='color: #2C3E50; margin-bottom: 1rem;'>
                <i class="fas fa-download"></i> Export Options
            </h4>
            <p style='color: #7F8C8D; font-size: 0.9rem;'>
                <i class="fas fa-file-csv" style="margin-right: 5px;"></i>
                Download the complete anomaly report for further analysis.
            </p>
            <p style='color: #7F8C8D; font-size: 0.9rem;'>
                <i class="fas fa-info-circle" style="margin-right: 5px;"></i>
                Includes all detected anomalies with details.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        csv = report.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name=f"anomaly_report_{selected_user}_{selected_metric}.csv",
            mime="text/csv",
            help="Download the anomaly report as CSV"
        )
else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #2ECC71;'>
            <i class="fas fa-trophy"></i> No Anomalies Detected!
        </h3>
        <p style='color: #7F8C8D;'>
            <i class="fas fa-check-circle" style="color: #2ECC71; margin-right: 5px;"></i>
            The selected data shows normal patterns within expected ranges.
        </p>
        <p style='color: #7F8C8D;'>
            <i class="fas fa-heartbeat" style="color: #2ECC71; margin-right: 5px;"></i>
            This indicates consistent and healthy metrics for the chosen period.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style='margin: 2rem 0; border: none; border-top: 2px solid #f0f0f0;'>
<div style='text-align: center; color: #7F8C8D; font-size: 0.9rem; padding: 1rem;'>
    <div class="status-icon">
        <i class="fas fa-heartbeat"></i>
    </div>
    <p><strong>FitPulse Health Analytics</strong> | Health Monitoring Dashboard v1.0</p>
    <div class="footer-icons">
        <a href="#" title="Health Insights"><i class="fas fa-heartbeat"></i></a>
        <a href="#" title="Data Analytics"><i class="fas fa-chart-line"></i></a>
        <a href="#" title="Anomaly Detection"><i class="fas fa-search"></i></a>
        <a href="#" title="Export Reports"><i class="fas fa-file-export"></i></a>
        <a href="#" title="User Support"><i class="fas fa-headset"></i></a>
    </div>
    <p style='font-size: 0.8rem; margin-top: 15px;'>
        <i class="fas fa-shield-alt" style="margin-right: 5px;"></i>
        For healthcare professionals and fitness enthusiasts | Data privacy ensured
    </p>
</div>
""", unsafe_allow_html=True)
