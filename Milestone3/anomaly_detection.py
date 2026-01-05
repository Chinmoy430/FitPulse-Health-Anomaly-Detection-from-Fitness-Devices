import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("/content/drive/MyDrive/FitPulse Health Anomaly Detection from Fitness Devices/Milestone2/data/cleaned_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

# Select at least 5 users
user_ids = df["Id"].unique()[:5]

print("Selected User IDs:", user_ids)

# ANOMALY IDENTIFICATION (RESIDUAL ANALYSIS)


all_hr_results = []

for uid in user_ids:
    user_df = df[df["Id"] == uid]

    # -------- Heart Rate Data --------
    hr = user_df[["timestamp", "heart_rate"]].dropna()
    hr = hr.rename(columns={"timestamp": "ds", "heart_rate": "y"})
    hr = hr.set_index("ds").resample("D").mean().reset_index()

    if len(hr) < 10:
        continue

    # Prophet Model
    model = Prophet(daily_seasonality=True)
    model.fit(hr)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Residuals
    merged = hr.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    merged["residual"] = merged["y"] - merged["yhat"]
    merged["user_id"] = uid

    all_hr_results.append(merged)

# Combine all users
hr_results = pd.concat(all_hr_results, ignore_index=True)

# Threshold-based anomaly detection (Standard Deviation)

threshold_hr = 2 * hr_results["residual"].std()
hr_results["anomaly"] = abs(hr_results["residual"]) > threshold_hr

#ANOMALY LABELING

hr_results["label"] = hr_results["anomaly"].apply(
    lambda x: "Anomalous" if x else "Normal"
)

print("Sample labeled data:")
hr_results.head()

#VISUALIZATION OF HEART RATE ANOMALIES

plt.figure(figsize=(12,6))

for uid in user_ids:
    user_data = hr_results[hr_results["user_id"] == uid]
    plt.plot(user_data["ds"], user_data["y"], label=f"User {uid}")

    anomalies = user_data[user_data["anomaly"]]
    plt.scatter(anomalies["ds"], anomalies["y"], color="red")

plt.title("Heart Rate Time-Series with Anomalies (5 Users)")
plt.xlabel("Date")
plt.ylabel("Heart Rate")
plt.legend()
plt.tight_layout()

plt.savefig("visualizations/heart_rate_anomalies.png")
plt.show()

# ============================================================
# SLEEP ANOMALY DETECTION 

all_sleep_results = []

for uid in user_ids:
    user_df = df[df["Id"] == uid]

    # -------- Sleep Data --------
    sleep = user_df[["timestamp", "sleep"]].dropna()
    sleep = sleep.rename(columns={"timestamp": "ds", "sleep": "y"})
    sleep = sleep.set_index("ds").resample("D").mean().reset_index()

    if len(sleep) < 10:
        continue

    model = Prophet(daily_seasonality=True)
    model.fit(sleep)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    merged = sleep.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    merged["residual"] = merged["y"] - merged["yhat"]
    merged["user_id"] = uid

    all_sleep_results.append(merged)

sleep_results = pd.concat(all_sleep_results, ignore_index=True)

# Threshold-based detection
threshold_sleep = 2 * sleep_results["residual"].std()
sleep_results["anomaly"] = abs(sleep_results["residual"]) > threshold_sleep

# ============================================================
# VISUALIZATION OF SLEEP ANOMALIES
# ============================================================

plt.figure(figsize=(12,6))

for uid in user_ids:
    user_data = sleep_results[sleep_results["user_id"] == uid]
    plt.plot(user_data["ds"], user_data["y"], label=f"User {uid}")

    anomalies = user_data[user_data["anomaly"]]
    plt.scatter(anomalies["ds"], anomalies["y"], color="red")

plt.title("Sleep Pattern Visualization with Anomalies (5 Users)")
plt.xlabel("Date")
plt.ylabel("Sleep")
plt.legend()
plt.tight_layout()

plt.savefig("visualizations/sleep_anomalies.png")
plt.show()
print("Milestone 3 anomaly detection completed successfully.")
print("Screenshots saved in 'visualizations/' folder.")