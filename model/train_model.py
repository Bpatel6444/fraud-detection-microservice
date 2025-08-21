# model/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib  # For saving the model
import os # Import the os module

# 1. Generate a synthetic dataset for demonstration
np.random.seed(42)
n_samples = 2000

# Create legitimate transactions (cluster around lower amounts, usual hours)
legit_amounts = np.random.normal(50, 20, n_samples // 2)
legit_hours = np.random.normal(14, 6, n_samples // 2)

# Create fraudulent transactions (cluster around higher amounts, unusual hours)
fraud_amounts = np.random.normal(210, 50, n_samples // 2)
fraud_hours = np.random.normal(2, 4, n_samples // 2)

# Combine the data
amounts = np.concatenate([legit_amounts, fraud_amounts])
hours = np.concatenate([legit_hours, fraud_hours])
data = np.column_stack((amounts, hours))
# Label: first half legit (0), second half fraud (1) – for evaluation only
labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# 2. Train the Model
# Isolation Forest is great for anomaly detection like fraud.
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(data)

# 3. Save the model as a pickle file
model_dir = "model" # Define the directory name
os.makedirs(model_dir, exist_ok=True) # Create the directory if it doesn't exist
joblib.dump(model, os.path.join(model_dir, "IsolationForestFraudDetector.pkl")) # Join the directory and filename
print("Model trained and saved as 'model/IsolationForestFraudDetector.pkl'")