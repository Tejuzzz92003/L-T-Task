# Task 15 — End-to-End Deep Learning Production Deployment

## Project
**Image Classification System using PyTorch + Flask + Streamlit + Docker + Kubernetes**

This project follows the Task 15 instructions:
1. Develop and train a deep learning model.
2. Build a Flask API.
3. Create a Streamlit frontend.
4. Containerize the application with Docker.
5. Orchestrate the services with Docker Compose.
6. Kubernetes manifests are also included for deployment practice.

The model classifies CIFAR-10 images into 10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## 1. Software needed

Install:
- Python 3.10 or 3.11
- VS Code
- Docker Desktop (for container deployment)

You do NOT need to install Flask or Streamlit manually if you use the requirements file.

---

## 2. Open the project

Extract the ZIP.

Open the extracted folder in VS Code.

Open VS Code Terminal:
**Terminal → New Terminal**

---

## 3. Create virtual environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\activate
```

If PowerShell blocks activation, use:

```powershell
venv\Scripts\activate.bat
```

You should see `(venv)` in the terminal.

---

## 4. Install packages

```powershell
pip install -r requirements.txt
```

---

## 5. Train the deep learning model

Run:

```powershell
python train.py
```

The script automatically downloads the CIFAR-10 dataset through torchvision and trains a CNN.

For a faster demo, the default configuration uses a smaller training subset and 3 epochs.
For a stronger model, edit the settings at the top of `train.py`.

After successful training you should get:

```text
models/cifar10_cnn.pth
models/metadata.json
artifacts/training_history.json
artifacts/training_curves.png
```

---

## 6. Test the Flask API

Run:

```powershell
python api.py
```

You should see Flask running on:

```text
http://127.0.0.1:5000
```

Open this in the browser:

```text
http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 7. Test prediction

Keep Flask running.

Open a second VS Code terminal and run:

```powershell
python test_api.py
```

This creates a small test image and sends it to the Flask API.

You can also use Postman:

**POST**
`http://127.0.0.1:5000/predict`

Body → form-data:

Key: `image`

Type: File

Select an image.

---

## 8. Run Streamlit frontend

Open another terminal:

```powershell
.\venv\Scripts\activate
streamlit run streamlit_app.py
```

The browser normally opens at:

```text
http://localhost:8501
```

Upload an image and click **Predict Image**.

---

# Docker deployment

## 9. Build the API image

```powershell
docker build -t task15-api -f Dockerfile.api .
```

## 10. Build the Streamlit image

```powershell
docker build -t task15-frontend -f Dockerfile.frontend .
```

---

## 11. Run the complete application with Docker Compose

First make sure the trained model exists:

```text
models/cifar10_cnn.pth
```

Then:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:8501
```

The frontend calls the Flask API internally.

Stop the services with:

```powershell
docker compose down
```

---

# Kubernetes deployment

The `k8s` folder contains:
- namespace
- API deployment
- API service
- frontend deployment
- frontend service

For Minikube:

```powershell
minikube start
minikube image build -t task15-api:latest -f Dockerfile.api .
minikube image build -t task15-frontend:latest -f Dockerfile.frontend .
kubectl apply -f k8s/
kubectl get pods -n task15
kubectl get services -n task15
```

Then expose the frontend:

```powershell
minikube service task15-frontend -n task15
```

---

# Project architecture

```text
                 User
                   |
                   v
          Streamlit Frontend
              Port 8501
                   |
                   v
             Flask REST API
              Port 5000
                   |
                   v
          PyTorch CNN Model
                   |
                   v
             Prediction
```

Docker Compose orchestrates:

```text
task15-frontend  <---->  task15-api
```

Kubernetes orchestrates the same services using Deployments and Services.

---

# API endpoints

## Health

GET `/health`

## Model information

GET `/model-info`

## Prediction

POST `/predict`

Form-data:

```text
image = <image file>
```

Example response:

```json
{
  "success": true,
  "prediction": "cat",
  "confidence": 0.87,
  "top_3": [
    {"class": "cat", "confidence": 0.87},
    {"class": "dog", "confidence": 0.06},
    {"class": "frog", "confidence": 0.02}
  ]
}
```

---

# Deliverables for Task 15

- `train.py` — deep learning model training
- `models/cifar10_cnn.pth` — generated trained model
- `api.py` — Flask API
- `streamlit_app.py` — Streamlit frontend
- `Dockerfile.api` — API container
- `Dockerfile.frontend` — frontend container
- `docker-compose.yml` — local orchestration
- `k8s/` — Kubernetes deployment files
- `test_api.py` — API test
- `artifacts/` — training history/curve after training
- `report.md` — ready project report structure

---

# Important

The first training run needs internet access because CIFAR-10 is downloaded automatically.

If Windows shows a firewall prompt, allow Python/Docker as appropriate.

This is a student-friendly production-style project. The model is intentionally small so it can be trained on a normal laptop.
