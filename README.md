# 🛡️ Guardian - Fraud Detection System

Guardian is a **real-time fraud detection system** built using a **microservices architecture**.  
It combines **Java Spring Boot** (for API + orchestration) and **Python Flask** (for ML predictions) to detect suspicious financial transactions.

---

## 📐 Architecture

- **Java Service (Spring Boot)**  
  - Provides REST API endpoints  
  - Handles request validation and business logic  
  - Forwards transactions to the Python ML service  

- **Python Service (Flask + ML)**  
  - Hosts a trained **Isolation Forest** model  
  - Accepts transaction details and returns fraud predictions  

- **Communication**: Java → Python via REST calls  
- **Deployment**: Docker + Docker Compose  

---

## ✨ Features

- Real-time fraud detection using **Isolation Forest**
- Enhanced features:
  - Transaction amount
  - Hour of day
  - Merchant category
  - Distance from home
  - Transaction count in last hour
- **Dual detection**:  
  - Rule-based checks in Java  
  - ML-based anomaly detection in Python
- Dockerized microservices for easy deployment
- Health checks (`/health`) and monitoring endpoints

---

## ⚙️ Prerequisites

Make sure you have installed:

- Java 17+
- Maven
- Python 3.8+
- pip
- Docker & Docker Compose (for containerized deployment)

---

## 🚀 Setup & Running Locally

### 1️⃣ Python ML Service

1. Install dependencies:
   ```bash
   pip install flask flask-cors scikit-learn joblib pandas numpy
````

2. Train the ML model:

   ```bash
   python model/enhanced_train_model.py
   ```

   This will generate:

   ```
   model/enhanced_isolation_forest.pkl
   ```

3. Start the Flask app:

   ```bash
   python model/app.py
   ```

👉 Runs at: `http://localhost:5000`

---

### 2️⃣ Java Service (Spring Boot)

1. Build the application:

   ```bash
   mvn clean package
   ```

   The JAR file will be created in:

   ```
   target/fraud-detection-api-0.0.1-SNAPSHOT.jar
   ```

2. Run the JAR:

   ```bash
   java -jar target/fraud-detection-api-0.0.1-SNAPSHOT.jar
   ```

👉 Runs at: `http://localhost:8080`

---

## 📡 API Usage

### Java Service Endpoints

#### ✅ Rule-based check

```http
GET /api/check?amount=150000&hourOfDay=14
```

**Response**:

```
YELLOW ALERT: Potential fraud due to high amount.
```

#### ✅ ML-based check

```http
POST /api/check
```

**Body (JSON):**

```json
{
  "amount": 1000.0,
  "hourOfDay": 14,
  "merchantCategory": 2,
  "distanceFromHome": 0.5,
  "transactionCountLastHour": 3
}
```

**Response (JSON):**

```json
{
  "fraudProbability": 0.78,
  "isFraud": true,
  "message": "FRAUD_DETECTED"
}
```

---

### Python Service Endpoints

#### ✅ Health check

```http
GET /health
```

Response:

```json
{"status": "OK"}
```

#### ✅ Prediction

```http
POST /predict
```

**Body (JSON):**

```json
{
  "amount": 1000.0,
  "hourOfDay": 14,
  "merchantCategory": 2,
  "distanceFromHome": 0.5,
  "transactionCountLastHour": 3
}
```

**Response:**

```json
{
  "fraudProbability": 0.23,
  "isFraud": false,
  "message": "LEGITIMATE"
}
```

---

## 🐳 Docker Deployment

You can run **both services** together using Docker Compose:

1. Build & start:

   ```bash
   docker-compose up --build
   ```

2. Access services:

   * Java API → `http://localhost:8080`
   * Python ML API → `http://localhost:5000`

---

## 📊 Model Training

The model is trained with **synthetic data** including:

* `amount`
* `merchantCategory` (0–4)
* `distanceFromHome`
* `transactionCountLastHour`
* `hourOfDay`

Trained using:

```bash
python model/enhanced_train_model.py
```

Saved model:

```
model/enhanced_isolation_forest.pkl
```

---

## 🔮 Future Enhancements

* Use real-world transaction datasets (e.g., Kaggle Credit Card Fraud Dataset)
* Add user behavioral profiling
* Circuit breaker + retries for service communication
* Authentication & security (JWT, API keys)
* Monitoring & observability with Prometheus/Grafana

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

Developed by **Bhargey**
Designed for **real-time fraud detection demo with ML + microservices** 🚀





