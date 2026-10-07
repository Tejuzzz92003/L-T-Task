from pathlib import Path
import io
import json

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("models/cifar10_cnn.pth")
METADATA_PATH = Path("models/metadata.json")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CIFAR10CNN().to(DEVICE)
MODEL_LOADED = False

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    MODEL_LOADED = True

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


@app.get("/")
def home():
    return jsonify({
        "service": "Task 15 Image Classification API",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "endpoints": ["/health", "/model-info", "/predict"]
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy" if MODEL_LOADED else "model_missing",
        "model_loaded": MODEL_LOADED
    }), 200 if MODEL_LOADED else 503


@app.get("/model-info")
def model_info():
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return jsonify({
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
        "metadata": metadata
    })


@app.post("/predict")
def predict():
    if not MODEL_LOADED:
        return jsonify({
            "success": False,
            "error": "Model not found. Run: python train.py"
        }), 503

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please upload an image using form-data key 'image'."
        }), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]

        values, indices = torch.topk(probabilities, k=3)

        top_3 = [
            {
                "class": CLASS_NAMES[int(index)],
                "confidence": round(float(value), 4)
            }
            for value, index in zip(values, indices)
        ]

        return jsonify({
            "success": True,
            "prediction": top_3[0]["class"],
            "confidence": top_3[0]["confidence"],
            "top_3": top_3
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
