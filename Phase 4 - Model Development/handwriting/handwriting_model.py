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


class HandwritingDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: np.ndarray,
        transform=None,
        image_size: int = 224,
    ):
        self.image_paths = image_paths
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2

        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.conv(x))


class HandwritingCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 3):
        super(HandwritingCNN, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class HandwritingResNet(nn.Module):
    def __init__(self, n_classes: int = 3):
        super(HandwritingResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNetBlock, 64, 64, 2, 1)
        self.layer2 = self._make_layer(ResNetBlock, 64, 128, 2, 2)
        self.layer3 = self._make_layer(ResNetBlock, 128, 256, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class HandwritingEfficientNet(nn.Module):
    def __init__(self, n_classes: int = 3):
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            self.base_model = efficientnet_b0(
                weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.base_model.features[0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(num_features, n_classes)
            )
            self.uses_pretrained = True
        except:
            logger.warning("EfficientNet not available, using custom CNN")
            self.base_model = HandwritingCNN(in_channels=1, n_classes=n_classes)
            self.uses_pretrained = False

    def forward(self, x):
        return self.base_model(x)


class HandwritingViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        n_classes: int = 3,
        embed_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
    ):
        super(HandwritingViT, self).__init__()

        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        x = x[:, 0]

        return self.head(x)


class HandwritingModelTrainer:
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

    def build_model(self):
        if self.model_type == "cnn":
            self.model = HandwritingCNN(n_classes=self.n_classes)
        elif self.model_type == "resnet":
            self.model = HandwritingResNet(n_classes=self.n_classes)
        elif self.model_type == "efficientnet":
            self.model = HandwritingEfficientNet(n_classes=self.n_classes)
        elif self.model_type == "vit":
            self.model = HandwritingViT(n_classes=self.n_classes)
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

    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        import cv2

        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            images.append(img)

        images = torch.FloatTensor(np.array(images)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
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


def train_handwriting_model(
    image_dir: str,
    labels_path: str,
    model_type: str = "cnn",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 32,
    epochs: int = 50,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(image_dir, "*.jpeg")))

    labels_df = pd.read_csv(labels_path)

    image_names = [Path(f).stem for f in image_files]
    labels_df["image_name"] = labels_df["image_path"].apply(
        lambda x: Path(x).stem if pd.notna(x) else ""
    )

    matched_df = labels_df[labels_df["image_name"].isin(image_names)]

    matched_images = [
        f for f in image_files if Path(f).stem in matched_df["image_name"].values
    ]
    matched_labels = (
        matched_df.set_index("image_name")
        .loc[[Path(f).stem for f in matched_images], "overall_risk"]
        .map({"low": 0, "medium": 1, "high": 2})
        .values
    )

    X_train, X_test, y_train, y_test = train_test_split(
        matched_images,
        matched_labels,
        test_size=test_size,
        stratify=matched_labels,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=42,
    )

    train_dataset = HandwritingDataset(X_train, y_train)
    val_dataset = HandwritingDataset(X_val, y_val)
    test_dataset = HandwritingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    trainer = HandwritingModelTrainer(model_type=model_type, n_classes=3)
    trainer.build_model()

    logger.info("Training handwriting model...")
    trainer.train(train_loader, val_loader, epochs=epochs, patience=5)

    logger.info("Evaluating model...")
    results = trainer.evaluate(test_loader)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"handwriting_{model_type}_model.pt")
    trainer.save_checkpoint(model_path)

    with open(
        os.path.join(output_dir, f"handwriting_{model_type}_results.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    return trainer, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Handwriting Model Training")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument(
        "--model-type", default="cnn", choices=["cnn", "resnet", "efficientnet", "vit"]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_handwriting_model(
        image_dir=args.images,
        labels_path=args.labels,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
