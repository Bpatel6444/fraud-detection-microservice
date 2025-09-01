from flask import Flask, request, jsonify
from flask_cors import CORS  # Important for cross-origin requests
import joblib  # To load your .pkl model
import numpy as np

app = Flask(__name__)
CORS(app)  # This allows your Java app to call this Python app

# Load the pre-trained Isolation Forest model
model = joblib.load('IsolationForestFraudDetector.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "OK"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        amount = float(data['amount'])
        hour_of_day = int(data['hourOfDay'])

        features = np.array([[amount, hour_of_day]])
        prediction = model.predict(features)
        score = model.decision_function(features)

        # Explicit conversion of NumPy types to native Python types
        is_fraud = bool(prediction[0] == -1)          # numpy.bool_ -> bool
        fraud_prob = float(1 - (score[0] + 1) / 2)    # numpy.float64 -> float

        response = {
            "fraudProbability": fraud_prob,
            "isFraud": is_fraud,
            "message": "FRAUD_DETECTED" if is_fraud else "LEGITIMATE"
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
    
if __name__ == '__main__':
    # Run the Flask app. host='0.0.0.0' makes it available on all network interfaces.
    app.run(host='0.0.0.0', port=5000, debug=True)