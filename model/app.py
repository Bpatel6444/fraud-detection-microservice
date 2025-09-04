from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import math

app = Flask(__name__)
CORS(app)

# Set up logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Load the model and metadata
try:
    model = joblib.load('model/best_fraud_model.pkl')
    features = joblib.load('model/features.pkl')
    metadata = joblib.load('model/metadata.pkl')
    app.logger.info("Model and metadata loaded successfully")
    app.logger.info(f"Model type: {metadata.get('model_type', 'unknown')}")
    app.logger.info(f"Test PR-AUC: {metadata.get('test_pr_auc', 'unknown')}")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    model = None
    features = None
    metadata = {}

@app.route('/health', methods=['GET'])
def health_check():
    status = "OK" if model is not None else "ERROR"
    return jsonify({
        "status": status, 
        "model_loaded": model is not None,
        "model_type": metadata.get('model_type', 'unknown'),
        "test_pr_auc": metadata.get('test_pr_auc', 'unknown')
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.info(f"Received prediction request: {data}")
        
        # Extract basic features
        amount = float(data['amount'])
        timestamp_str = data.get('timestamp')
        
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.utcnow()
        
        # Create a feature dictionary with all possible features
        feature_dict = {}
        
        # Add PCA features if provided (V1-V28)
        for i in range(1, 29):
            v_key = f"V{i}"
            feature_dict[v_key] = data.get(v_key, 0.0)  # Default to 0 if not provided
        
        # Add engineered features
        feature_dict['log_amount'] = math.log1p(amount)
        
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        feature_dict['hour_sin'] = math.sin(2 * math.pi * hour / 24.0)
        feature_dict['hour_cos'] = math.cos(2 * math.pi * hour / 24.0)
        feature_dict['dow_sin'] = math.sin(2 * math.pi * day_of_week / 7.0)
        feature_dict['dow_cos'] = math.cos(2 * math.pi * day_of_week / 7.0)
        feature_dict['is_weekend'] = 1 if day_of_week in [5, 6] else 0
        
        # Rolling statistics (simplified - in production would use real transaction history)
        feature_dict['amount_roll_mean_10'] = amount * 0.9  # Placeholder
        feature_dict['amount_roll_std_10'] = amount * 0.1   # Placeholder
        
        # Create feature vector in the correct order
        feature_vector = [feature_dict[feature] for feature in features]
        
        # Make prediction
        prediction = model.predict([feature_vector])
        probability = model.predict_proba([feature_vector])[0][1]  # probability of class 1 (fraud)
        
        # Apply threshold (using F1-optimal threshold from training)
        threshold = metadata.get('f1_threshold', 0.5)
        is_fraud = bool(probability >= threshold)
        
        response = {
            "fraudProbability": float(probability),
            "isFraud": is_fraud,
            "threshold": float(threshold),
            "message": "FRAUD_DETECTED" if is_fraud else "LEGITIMATE",
            "modelType": metadata.get('model_type', 'unknown'),
            "featuresUsed": features
        }
        
        app.logger.info(f"Prediction result: {response}")
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)