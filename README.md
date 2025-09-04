# 🛡️ Guardian - Fraud Detection System

Guardian is a real-time fraud detection system built using a **microservices architecture**.
It combines **Java Spring Boot** (for API + orchestration) and **Python Flask** (for ML predictions) to detect suspicious financial transactions.

---

## 📐 Architecture

### Java Service (Spring Boot)

* Provides REST API endpoints
* Handles request validation and business logic
* Forwards transactions to the Python ML service

### Python Service (Flask + ML)

* Hosts a trained **XGBoost model**
* Accepts transaction details and returns fraud predictions

**Communication:** Java → Python via REST calls
**Deployment:** Docker + Docker Compose

---

## ✨ Features

* Real-time fraud detection using **XGBoost**
* Features engineered from data:

  * PCA components (V1–V28)
  * Transaction amount (log-scaled)
  * Time-based features (hour of day, day of week, weekend flag)
  * Rolling statistics (mean/std placeholders)
* Dual detection pipeline:

  * Rule-based checks in Java
  * ML-based anomaly detection in Python
* Dockerized microservices for easy deployment
* Health checks (`/health`) and monitoring endpoints

---

## ⚙️ Prerequisites

Make sure you have installed:

* Java 17+
* Maven
* Python 3.8+
* pip
* Docker & Docker Compose (for containerized deployment)

---

## 🚀 Setup & Running Locally

### 1️⃣ Python ML Service

Install dependencies:

```bash
pip install flask flask-cors scikit-learn joblib pandas numpy xgboost lightgbm
```

Train the ML model:

```bash
python model/train_model.py
```

This will generate:

* `models/best_fraud_model.pkl`
* `models/features.pkl`
* `models/metadata.pkl`

Start the Flask app:

```bash
python model/app.py
```

👉 Runs at: `http://localhost:5000`

---

### 2️⃣ Java Service (Spring Boot)

Build the application:

```bash
mvn clean package
```

The JAR file will be created in:

```
target/fraud-detection-api-0.0.1-SNAPSHOT.jar
```

Run the JAR:

```bash
java -jar target/fraud-detection-api-0.0.1-SNAPSHOT.jar
```

👉 Runs at: `http://localhost:8080`

---

## 📡 API Usage

### ✅ Java Service Endpoints

**Rule-based check**

```http
GET /api/check?amount=150000&hourOfDay=14
```

Response:

```
YELLOW ALERT: Potential fraud due to high amount.
```

**ML-based check (via Python service)**

```http
POST /api/check
```

Body (JSON):

```json
{
  "amount": 1000.0,
  "timestamp": "2023-10-05T03:30:00Z",
  "transactionId": "txn_test",
  "merchantId": "mcht_abc",
  "customerId": "cust_xyz",
  "pcaFeatures": {
    "V1": 1.23,
    "V2": -0.45,
    "V3": 0.67
  }
}
```

Response (JSON):

```json
{
  "fraudProbability": 0.82,
  "isFraud": true,
  "message": "FRAUD_DETECTED",
  "threshold": 0.74,
  "modelType": "xgboost",
  "featuresUsed": ["V1","V2","V3",...,"log_amount","hour_sin","dow_cos"]
}
```

---

### ✅ Python Service Endpoints

**Health check**

```http
GET /health
```

Response:

```json
{"status": "OK"}
```

**Prediction**

```http
POST /predict
```

Body (JSON):

```json
{
  "amount": 10000.0,
  "timestamp": "2023-10-05T03:30:00Z",
  "transactionId": "txn_highrisk",
  "merchantId": "mcht_test",
  "customerId": "cust_test",
  "pcaFeatures": {
    "V1": 5.5,
    "V2": -6.2,
    "V3": 4.8
  }
}
```

Response (JSON):

```json
{
  "fraudProbability": 0.00008,
  "isFraud": false,
  "message": "LEGITIMATE",
  "threshold": 0.74,
  "modelType": "xgboost",
  "featuresUsed": [...]
}
```

---

## 🐳 Docker Deployment

Run both services together using Docker Compose:

```bash
docker-compose up --build
```

Access services:

* Java API → `http://localhost:8080`
* Python ML API → `http://localhost:5000`

---

## 📊 Model Training

The model is trained with the **Kaggle Credit Card Fraud Dataset (2013)**.

Features used:

* PCA components (V1–V28)
* Transaction amount (log transformed)
* Time-based features (hour, day of week, weekend)
* Rolling statistics (mean/std placeholders)

Saved models:

* `models/best_fraud_model.pkl`
* `models/features.pkl`
* `models/metadata.pkl`

⚠️ **Note:** This dataset is for research/benchmarking only, not production use.

---

## 🔮 Future Roadmap

Guardian is just the **first step**. Our vision is to evolve this into a **Next-Generation Financial AI Platform**. Planned enhancements include:

1. **Neural Fraud Detection Network** – combine XGBoost, deep learning, and graph-based anomaly detection for higher accuracy.
2. **Autonomous Compliance Engine** – real-time monitoring of financial regulations using NLP with explainable AI for auditability.
3. **Predictive Risk Intelligence** – forecasting systemic financial risks using macroeconomic indicators and sentiment analysis.
4. **Behavioral Biometrics & Quantum-Resistant Security** – continuous user authentication and future-proof encryption.
5. **Unified Cognitive Dashboard** – visual insights across fraud, compliance, and risk management.

We welcome contributions from:

* ML engineers (model improvement & explainability)
* Backend developers (scalable APIs, PaaS enablement)
* Fintech/domain experts (fraud patterns, compliance rules)
* DevOps engineers (monitoring, Kubernetes, cloud scaling)

👉 Check the [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Developed by **Bhargey**
Designed for **real-time fraud detection** demo with **ML + microservices** 🚀




