# 🐱🐶 Cats vs Dogs Classifier API

A production-ready FastAPI application for classifying images as **Cat** or **Dog**, built as part of an MLOps pipeline with CI and Docker support.

---

##  Project Overview

This project provides:

- ✅ CNN-based binary image classification (Cat vs Dog)
- ✅ FastAPI REST API
- ✅ Health check endpoint
- ✅ Dockerized deployment
- ✅ GitHub Actions CI pipeline
- ✅ Pytest-based automated testing
- ✅ Structured logging
- ✅ Production-style model loading

---

## 🏗️ Tech Stack

- Python 3.11
- FastAPI
- PyTorch
- Uvicorn
- Pytest
- Docker
- GitHub Actions

---

## 📂 Project Structure

.
├── app.py # FastAPI application
├── model.py # CNN architecture
├── preprocess.py # Image preprocessing logic
├── model.pt # Trained model weights
├── test_api.py # API unit tests
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
└── .github/workflows/
└── ci.yml # CI pipeline


---

## ⚙️ Installation (Local Setup)

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the API
```bash
uvicorn app:app --reload
```

API will be available at:
http://127.0.0.1:8000


## 🔎 API Endpoints

### 🟢 Health Check

GET /health
```
{
  "status": "healthy"
}
```

### 🐾 Prediction
POST /predict

Upload an image file (cat or dog).

Response Example:
```
{
  "probability": 0.8732,
  "label": "dog",
  "latency_sec": 0.0142
}
```

### 🧪 Running Tests
This project uses FastAPI TestClient for API testing.
```bash
pytest
```
CI automatically runs tests on every push to master.

### 🐳 Docker Usage
Build Docker Image
```bash
docker build -t catsdogs-api .
```

Run Container
```bash
docker run -p 8000:8000 catsdogs-api
```

### 🔁 CI Pipeline
GitHub Actions automatically:

* Sets up Python 3.11
* Installs dependencies
* Runs pytest
* Builds Docker image

Workflow file:
.github/workflows/ci.yml

### 📈 Logging
Structured logging is implemented for:

* Server startup
* Health checks
* Predictions
* Errors
* Latency tracking

Example log:
```
2026-02-21 10:21:33 | INFO | catsdogs-api | Prediction successful | Label: dog | Probability: 0.8732 | Latency: 0.0142 sec
```

### 👨‍💻 Author
Mohan K R (2024aa05419@wilp.bits-pilani.ac.in)
Shreyas T S (2024aa05418@wilp.bits-pilani.ac.in)
Sathwik H R (2024aa05903@wilp.bits-pilani.ac.in)