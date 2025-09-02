# Base image
FROM python:3.11-slim

WORKDIR /app

# Copy Python files
COPY app.py .
COPY IsolationForestFraudDetector.pkl .

# Install dependencies
RUN pip install --no-cache-dir flask flask-cors scikit-learn joblib

# Expose port
EXPOSE 5000

# Run the Python app
CMD ["python", "app.py"]
