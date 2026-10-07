# Task 15: End-to-End Deep Learning Production Deployment

## 1. Title
**End-to-End Deep Learning Production Deployment — Image Classification System**

## 2. Objective
To design, develop, containerize, orchestrate and deploy a complete deep learning application following industry-standard MLOps practices.

## 3. Problem Statement
The project builds an image classification system that predicts one of ten CIFAR-10 classes from an uploaded image.

## 4. Technology Stack
- Python
- PyTorch
- Torchvision
- Flask REST API
- Streamlit
- Docker
- Docker Compose
- Kubernetes

## 5. Model
A custom Convolutional Neural Network (CNN) is trained on CIFAR-10.

Architecture:
1. Convolution layer
2. ReLU activation
3. Max pooling
4. Convolution layer
5. ReLU activation
6. Max pooling
7. Convolution layer
8. ReLU activation
9. Max pooling
10. Fully connected layers
11. Softmax probabilities during inference

## 6. Dataset
CIFAR-10 contains 10 classes:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## 7. Training
The `train.py` file:
- downloads CIFAR-10
- preprocesses images
- trains the CNN
- evaluates validation/test accuracy
- saves the model
- saves training history
- generates a training curve

## 8. REST API
Flask exposes:
- GET `/`
- GET `/health`
- GET `/model-info`
- POST `/predict`

The `/predict` endpoint accepts an image and returns the top-3 predictions.

## 9. Frontend
Streamlit provides:
- image upload
- prediction button
- predicted class
- confidence
- top-3 results

## 10. Containerization
Two Docker images are created:
- API container
- Streamlit frontend container

## 11. Orchestration
Docker Compose runs the API and frontend together.

Kubernetes manifests are also provided for deployment using Deployments and Services.

## 12. MLOps Practices Demonstrated
- reproducible training script
- saved model artifact
- API-based model serving
- separate frontend and backend services
- health check endpoint
- containerization
- service orchestration
- Kubernetes deployment configuration
- configuration through environment variables

## 13. Testing
The project includes `test_api.py` for API smoke testing.

Recommended screenshots for submission:
1. Training running in terminal
2. Training accuracy/curve
3. Flask `/health` response
4. Streamlit application
5. Prediction result
6. `docker compose up` output
7. Docker containers
8. Kubernetes pods/services

## 14. Conclusion
The project demonstrates the complete lifecycle of a small deep learning application from model training to REST API serving, user interface, containerization and orchestration.
