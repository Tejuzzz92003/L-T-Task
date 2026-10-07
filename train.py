from pathlib import Path
import json
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -----------------------------
# Student-friendly configuration
# -----------------------------
SEED = 42
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Set these to None to use the full CIFAR-10 dataset.
MAX_TRAIN_SAMPLES = 10000
MAX_TEST_SAMPLES = 2000

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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


def make_subset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2436, 0.2616)
        ),
    ])

    train_full = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform,
    )

    test_full = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataset = make_subset(train_full, MAX_TRAIN_SAMPLES)
    test_dataset = make_subset(test_full, MAX_TEST_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

    model_path = MODEL_DIR / "cifar10_cnn.pth"
    torch.save(model.state_dict(), model_path)

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "CIFAR10CNN",
                "framework": "PyTorch",
                "classes": CLASS_NAMES,
                "input_size": [32, 32],
                "epochs": EPOCHS,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "device": str(device),
            },
            f,
            indent=2,
        )

    with open(ARTIFACT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "training_curves.png", dpi=150)
    plt.close()

    print("\nTraining completed.")
    print(f"Model saved to: {model_path}")
    print(f"Training artifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
