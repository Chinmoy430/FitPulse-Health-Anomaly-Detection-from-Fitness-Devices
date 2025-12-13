
# FitPulse Health Anomaly Detection from Fitness Devices

## Milestone 1: Data Collection and Preprocessing

## Objective
The objective of this milestone is to collect fitness tracker data from wearable devices, preprocess it by handling missing values and normalizing timestamps to UTC, align all metrics to a consistent 1-minute interval, and generate a clean, consolidated dataset ready for analysis and anomaly detection.

## Dataset Source
The dataset used in this project is the Fitbit Fitness Tracker Dataset from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/jahanzaibqamar/fitbit-fitness-tracker-data

## Files Used
- heartrate_seconds_merged.csv (Heart rate data)
- minuteStepsNarrow_merged.csv (Steps data)
- sleepDay_merged.csv (Sleep logs)

## Steps Performed
1. Created the required project folder structure.
2. Uploaded fitness data CSV files into the data directory.
3. Read and validated datasets using Pandas.
4. Converted all timestamp columns into Pandas datetime format and normalized them to UTC.
5. Handled missing and null values using simple and appropriate strategies.
6. Resampled heart rate data to a consistent 1-minute interval.
7. Aligned steps and sleep data to the same time scale.
8. Merged heart rate, steps, and sleep data into a single consolidated dataset.
9. Saved the final cleaned dataset as cleaned_dataset.csv.

## Output
The final cleaned and time-aligned dataset is available at:
Milestone1/data/cleaned_dataset.csv

## Visualization
A Streamlit application was created to visualize the cleaned dataset.

## Live Demo
https://huggingface.co/spaces/Chinmoy02/FitPulse

## Tools Used
- Python
- Pandas
- NumPy
- Streamlit
- Google Colab
- GitHub
- Hugging Face Spaces
