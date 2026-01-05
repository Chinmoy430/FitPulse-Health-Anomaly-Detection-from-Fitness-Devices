
# Milestone 3: Anomaly Detection and Visualization

## Objective
The objective of Milestone 3 is to identify, label, and visualize anomalies in 
fitness time-series data using model residuals and clustering insights. 
This milestone builds upon the preprocessing and modeling work from previous milestones
to clearly distinguish normal and abnormal behavioral patterns.

## Dataset Used
The analysis is performed on the preprocessed fitness dataset generated in Milestone 1.  
The dataset contains the following key columns:
- Id (User identifier)
- timestamp
- heart_rate
- steps
- sleep

Five users were selected for anomaly analysis to ensure clarity and computational efficiency.

## Steps Followed

### 1. Residual-Based Anomaly Identification
- Facebook Prophet was used to model expected temporal behavior for heart rate and sleep data.
- Residuals were calculated as the difference between actual values and predicted values (actual − predicted).
- Residual analysis helped identify deviations from normal behavioral patterns.

### 2. Threshold-Based Anomaly Detection
- A statistical threshold was applied using two times the standard deviation of residuals.
- Data points exceeding this threshold were considered anomalous.
- This approach helps detect sudden spikes or drops in fitness metrics.

### 3. Cluster-Based Anomaly Reference
- Behavioral clustering results obtained in Milestone 2 (using KMeans and PCA) were referenced.
- Outlier clusters were treated as indicators of atypical behavior, supporting anomaly identification.

### 4. Anomaly Labeling
- Each data point was clearly labeled as either:
  - Normal
  - Anomalous
- This labeling ensures clarity between normal and abnormal observations in the dataset.

### 5. Visualization of Anomalies
- Time-series visualizations were created for heart rate and sleep data.
- Anomalies were highlighted using red markers over the time-series plots.
- Visual outputs were saved as images for reporting and evaluation purposes.

## Tools and Libraries Used
- Python
- Pandas
- Facebook Prophet
- Matplotlib
- Google Colaboratory

## Key Insights
- Standard deviations in heart rate were successfully detected using residual analysis.
- Irregular sleep patterns were identified and clearly visualized.
- Residual-based detection combined with clustering insights provides a robust method for identifying abnormal behavior.
- Visual representations make anomaly patterns easy to interpret and validate.

## Visual Outputs
The following screenshots are included in the `visualizations` folder:
- `heart_rate_anomalies.png`: Heart rate time-series with anomalies highlighted
- `sleep_anomalies.png`: Sleep pattern visualization showing abnormal segments

