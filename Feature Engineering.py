import pandas as pd

# Load the processed data
data = pd.read_excel(r"C:\Users\tusha\Downloads\AnomaData.xlsx")

# Convert 'time' to datetime
data['time'] = pd.to_datetime(data['time'])

# Feature engineering example: Extracting hour from the timestamp
data['hour'] = data['time'].dt.hour

# Drop the original 'time' column
data = data.drop(columns=['time'])

# Save the updated data with engineered features
data.to_csv(r'C:\Users\tusha\Downloads\AnomaData_featured.csv', index=False)
