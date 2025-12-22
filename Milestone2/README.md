
FitPulse Health Anomaly Detection from Fitness Devices
Milestone 2: Feature Extraction and Modeling

Objective of Milestone 2

The objective of this milestone is to derive meaningful insights from the preprocessed
fitness tracker data by performing feature extraction, trend modeling, anomaly detection, 
and behavioral pattern analysis. This milestone lays the foundation for identifying 
abnormal fitness behavior in later stages of the project.

Dataset Description
This milestone uses the cleaned and time-aligned dataset generated in Milestone 1.

Dataset Used:
- cleaned_dataset.csv

Metrics Included:
- Heart Rate
- Steps
- Sleep

The dataset contains multiple users’ fitness data normalized to a consistent time scale.

Steps Performed

1. Feature Extraction
- Selected five users from the dataset.
- Applied TSFresh to automatically extract statistical time-series features.
- Extracted features include mean, standard deviation, variance, RMS, minimum, and maximum values.

2. Feature Selection
- Applied variance thresholding to remove low-variance and less informative features.
- Retained only relevant features for modeling.
- Saved the final feature set as selected_features.csv.

3. Trend Modeling
- Applied Facebook Prophet to model temporal trends and seasonality.
- Modeled trends separately for heart rate, steps, and sleep.
- Generated forecasts with confidence intervals for each metric.

4. Anomaly Detection
- Computed residuals between observed values and Prophet predictions.
- Identified unusual behavior using the rule:
  absolute residual greater than 2 times the standard deviation.
- Days crossing this threshold were flagged as anomalies.
- In some cases, no anomalies were detected, indicating stable behavior.

5. Clustering Behavioral Patterns
- Used KMeans clustering to group users based on extracted features.
- Standardized features using StandardScaler.
- Applied PCA for dimensionality reduction and visualization.
- Identified clusters representing normal and atypical behavior patterns.

Visualization and Deployment
All feature extraction results, trend modeling outputs, anomaly detection tables,
and clustering visualizations are presented through an interactive Streamlit 
application deployed on Hugging Face Spaces.

Live Application Link:
https://huggingface.co/spaces/Chinmoy02/FitPulse2

The application provides:
- Cleaned dataset preview
- Extracted feature summary
- Prophet trend visualizations for heart rate, steps, and sleep
- Residual-based anomaly detection
- Behavioral clustering visualization

Tools and Libraries Used
- Python
- Pandas
- NumPy
- TSFresh
- Facebook Prophet
- Scikit-learn
- Matplotlib
- Streamlit
- Google Colab
- Hugging Face Spaces

Key Observations
- TSFresh effectively extracted meaningful time-series features.
- Feature selection improved modeling efficiency by removing redundant features.
- Prophet successfully captured temporal patterns and seasonality.
- Residual-based analysis highlighted unusual behavior in certain periods.
- Clustering grouped users with similar fitness behavior patterns.
