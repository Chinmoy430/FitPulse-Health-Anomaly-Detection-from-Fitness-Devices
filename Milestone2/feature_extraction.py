import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.feature_selection import VarianceThreshold

# Load cleaned dataset
df = pd.read_csv("data/cleaned_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Use FIRST 5 USERS (as in notebook)
user_ids = df["Id"].unique()[:5]
df = df[df["Id"].isin(user_ids)]

# Limit rows per user to keep balance
df = df.groupby("Id").head(600)

# Prepare TSFresh input (Heart Rate)
ts_data = df[["Id", "timestamp", "heart_rate"]].dropna()
ts_data.columns = ["id", "time", "value"]

# Extract features
features = extract_features(
    ts_data,
    column_id="id",
    column_sort="time",
    default_fc_parameters=MinimalFCParameters(),
    disable_progressbar=False
)

print("Extracted features shape:", features.shape)

# Feature selection (same as notebook)
if features.shape[0] > 1:
    selector = VarianceThreshold(threshold=0.01)
    selected_array = selector.fit_transform(features)
    selected_features = pd.DataFrame(
        selected_array,
        index=features.index,
        columns=features.columns[selector.get_support()]
    )
else:
    selected_features = features.copy()

# Save outputs
features.to_csv("data/tsfresh_features.csv")
selected_features.to_csv("data/selected_features.csv")

print("Final selected feature shape:", selected_features.shape)
