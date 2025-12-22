import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

# Use same 5 users as feature extraction
user_ids = df["Id"].unique()[:5]
df = df[df["Id"].isin(user_ids)]

# ---------- PROPHET (per user, per metric) ----------
for uid in user_ids:
    print("\nUser:", uid)
    user_df = df[df["Id"] == uid]

    for metric in ["heart_rate", "steps", "sleep"]:
        temp = user_df[["timestamp", metric]].dropna()
        temp = temp.rename(columns={"timestamp": "ds", metric: "y"})
        temp = temp.set_index("ds").resample("D").mean().reset_index()

        model = Prophet(daily_seasonality=True)
        model.fit(temp)

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        model.plot(forecast)
        plt.title(f"{metric.capitalize()} Trend - User {uid}")
        plt.show()

        merged = temp.merge(forecast[["ds", "yhat"]], on="ds", how="left")
        merged["residual"] = merged["y"] - merged["yhat"]
        threshold = 2 * merged["residual"].std()
        merged["anomaly"] = abs(merged["residual"]) > threshold

        print("Anomalies:")
        print(merged[merged["anomaly"]])

# ---------- CLUSTERING (5 users, no error) ----------
features = pd.read_csv("data/selected_features.csv", index_col=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="viridis", s=100)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("User Behavioral Clusters")
plt.show()
