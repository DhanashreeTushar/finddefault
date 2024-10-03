import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/anomaly_detection_model.pkl')

# Load new data
new_data = pd.read_csv('data/new_data.csv')

# Ensure the data is preprocessed the same way
# Example: Feature engineering, scaling, etc.
new_data['hour'] = pd.to_datetime(new_data['time']).dt.hour
new_data = new_data.drop(columns=['time'])

# Predict anomalies
predictions = model.predict(new_data)

# Output predictions
print(predictions)