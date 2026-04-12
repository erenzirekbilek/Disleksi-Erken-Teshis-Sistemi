import os
import glob
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpeechDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class SpectrogramCNN(nn.Module):
    def __init__(self, n_mfcc: int = 13, n_classes: int = 3):
        super(SpectrogramCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * (n_mfcc // 8) * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class SpeechLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super(SpeechLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        out = self.fc(hidden)
        return out


class SpeechTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super(SpeechTransformer, self).__init__()

        self.positional_encoding = nn.Parameter(torch.randn(1000, input_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.positional_encoding[:seq_len]
        x = self.transformer(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


class SpeechModelTrainer:
    def __init__(
        self,
        model_type: str = "cnn",
        n_classes: int = 3,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_type = model_type
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.model = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def build_model(self, input_shape: int):
        if self.model_type == "cnn":
            self.model = SpectrogramCNN(n_mfcc=input_shape, n_classes=self.n_classes)
        elif self.model_type == "lstm":
            self.model = SpeechLSTM(input_size=input_shape, n_classes=self.n_classes)
        elif self.model_type == "transformer":
            self.model = SpeechTransformer(
                input_size=input_shape, n_classes=self.n_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)
        logger.info(f"Built {self.model_type} model")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 5,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = criterion(outputs, y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    def evaluate(self, test_loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        results = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
            "precision": precision_score(all_labels, all_preds, average="macro"),
            "recall": recall_score(all_labels, all_preds, average="macro"),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

        if self.n_classes == 3:
            try:
                results["roc_auc"] = roc_auc_score(
                    all_labels, all_probs, multi_class="ovr"
                )
            except:
                results["roc_auc"] = None

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        return preds.cpu().numpy(), probs.cpu().numpy()

    def save_checkpoint(self, path: str):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "history": self.history}, path
        )
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})
        logger.info(f"Model loaded from {path}")


def train_speech_model(
    features_path: str,
    labels_path: str,
    model_type: str = "cnn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 32,
    epochs: int = 50,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)

    with open(features_path, "r") as f:
        features_data = json.load(f)

    if isinstance(features_data, list):
        features_list = features_data
    else:
        features_list = [features_data]

    X = np.array([f.get("mfcc_mean", [0] * 39) for f in features_list])
    X = np.nan_to_num(X, nan=0.0)

    labels_df = pd.read_csv(labels_path)
    y = labels_df["overall_risk"].map({"low": 0, "medium": 1, "high": 2}).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=42,
    )

    train_dataset = SpeechDataset(X_train, y_train)
    val_dataset = SpeechDataset(X_val, y_val)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X.shape[1]
    trainer = SpeechModelTrainer(model_type=model_type, n_classes=3)
    trainer.build_model(input_size)

    logger.info("Training speech model...")
    trainer.train(train_loader, val_loader, epochs=epochs, patience=5)

    logger.info("Evaluating model...")
    results = trainer.evaluate(test_loader)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"speech_{model_type}_model.pt")
    trainer.save_checkpoint(model_path)

    with open(os.path.join(output_dir, f"speech_{model_type}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Speech Model Training")
    parser.add_argument("--features", required=True, help="Features JSON file")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument(
        "--model-type", default="cnn", choices=["cnn", "lstm", "transformer"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_speech_model(
        features_path=args.features,
        labels_path=args.labels,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
