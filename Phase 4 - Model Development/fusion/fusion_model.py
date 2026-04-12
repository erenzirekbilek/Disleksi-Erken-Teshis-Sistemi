import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FusionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super(FusionMLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.fc(x)


class FusionModel(nn.Module):
    def __init__(self, input_dim: int = 3, n_classes: int = 3):
        super(FusionModel, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        weighted = x * attn_weights
        return self.classifier(weighted)


class FusionTrainer:
    def __init__(
        self,
        model_type: str = "mlp",
        n_classes: int = 3,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_type = model_type
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.model = None
        self.sklearn_model = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def build_model(self, input_dim: int = 3):
        if self.model_type == "mlp":
            self.model = FusionMLP(input_dim=input_dim, n_classes=self.n_classes)
        elif self.model_type == "attention":
            self.model = FusionModel(input_dim=input_dim, n_classes=self.n_classes)
        elif self.model_type == "gradient_boosting":
            self.sklearn_model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
            return
        elif self.model_type == "random_forest":
            self.sklearn_model = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
            return
        elif self.model_type == "logistic_regression":
            self.sklearn_model = LogisticRegression(
                multi_class="multinomial", max_iter=1000, random_state=42
            )
            return
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)
        logger.info(f"Built {self.model_type} fusion model")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ):
        if self.sklearn_model is not None:
            X_train = []
            y_train = []
            for X, y in train_loader:
                X_train.append(X.numpy())
                y_train.append(y.numpy())
            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)

            self.sklearn_model.fit(X_train, y_train)
            logger.info("Sklearn model trained")
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
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

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint("best_fusion_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    def evaluate(self, test_loader: DataLoader) -> Dict:
        if self.sklearn_model is not None:
            X_test = []
            y_test = []
            for X, y in test_loader:
                X_test.append(X.numpy())
                y_test.append(y.numpy())
            X_test = np.vstack(X_test)
            y_test = np.concatenate(y_test)

            all_preds = self.sklearn_model.predict(X_test)
            all_probs = self.sklearn_model.predict_proba(X_test)
            all_labels = y_test
        else:
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

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.sklearn_model is not None:
            preds = self.sklearn_model.predict(features)
            probs = self.sklearn_model.predict_proba(features)
            return preds, probs

        self.model.eval()
        X_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        return preds.cpu().numpy(), probs.cpu().numpy()

    def save_checkpoint(self, path: str):
        if self.model is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "history": self.history,
                    "model_type": self.model_type,
                },
                path,
            )
        if self.sklearn_model is not None:
            import joblib

            joblib.dump(self.sklearn_model, path.replace(".pt", "_sklearn.joblib"))
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        if self.model is not None:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.history = checkpoint.get("history", {})
        if self.sklearn_model is not None:
            import joblib

            self.sklearn_model = joblib.load(path.replace(".pt", "_sklearn.joblib"))
        logger.info(f"Model loaded from {path}")


def load_modality_scores(
    speech_scores_path: str,
    handwriting_scores_path: str,
    text_scores_path: str,
    labels_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    with open(speech_scores_path, "r") as f:
        speech_data = json.load(f)
    with open(handwriting_scores_path, "r") as f:
        handwriting_data = json.load(f)
    with open(text_scores_path, "r") as f:
        text_data = json.load(f)

    labels_df = pd.read_csv(labels_path)

    speech_df = pd.DataFrame(speech_data)
    handwriting_df = pd.DataFrame(handwriting_data)
    text_df = pd.DataFrame(text_data)

    merged = labels_df.copy()
    merged = merged.merge(
        speech_df[["sample_id", "score"]].rename(columns={"score": "speech_score"}),
        on="sample_id",
        how="left",
    )
    merged = merged.merge(
        handwriting_df[["sample_id", "score"]].rename(
            columns={"score": "handwriting_score"}
        ),
        on="sample_id",
        how="left",
    )
    merged = merged.merge(
        text_df[["sample_id", "score"]].rename(columns={"score": "text_score"}),
        on="sample_id",
        how="left",
    )

    merged = merged.dropna(subset=["speech_score", "handwriting_score", "text_score"])

    X = merged[["speech_score", "handwriting_score", "text_score"]].values
    y = merged["overall_risk"].map({"low": 0, "medium": 1, "high": 2}).values

    return X, y


def train_fusion_model(
    speech_scores_path: str,
    handwriting_scores_path: str,
    text_scores_path: str,
    labels_path: str,
    model_type: str = "mlp",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 32,
    epochs: int = 100,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading modality scores...")
    X, y = load_modality_scores(
        speech_scores_path, handwriting_scores_path, text_scores_path, labels_path
    )
    logger.info(f"Loaded {len(X)} samples with features shape {X.shape}")

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

    train_dataset = FusionDataset(X_train, y_train)
    val_dataset = FusionDataset(X_val, y_val)
    test_dataset = FusionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X.shape[1]
    trainer = FusionTrainer(model_type=model_type, n_classes=3)
    trainer.build_model(input_dim)

    logger.info("Training fusion model...")
    trainer.train(train_loader, val_loader, epochs=epochs, patience=10)

    logger.info("Evaluating model...")
    results = trainer.evaluate(test_loader)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"fusion_{model_type}_model.pt")
    trainer.save_checkpoint(model_path)

    with open(os.path.join(output_dir, f"fusion_{model_type}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


def weighted_fusion(
    speech_scores: np.ndarray,
    handwriting_scores: np.ndarray,
    text_scores: np.ndarray,
    weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
) -> np.ndarray:
    w1, w2, w3 = weights
    combined = w1 * speech_scores + w2 * handwriting_scores + w3 * text_scores
    return combined


def cross_validate_fusion(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "mlp",
    n_splits: int = 5,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )

        train_dataset = FusionDataset(X_train, y_train)
        val_dataset = FusionDataset(X_val, y_val)
        test_dataset = FusionDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        trainer = FusionTrainer(model_type=model_type)
        trainer.build_model(input_dim=X.shape[1])
        trainer.train(train_loader, val_loader, epochs=50, patience=5)

        results = trainer.evaluate(test_loader)
        results["fold"] = fold + 1
        all_results.append(results)

        logger.info(
            f"Fold {fold + 1}: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}"
        )

    avg_results = {
        "accuracy": np.mean([r["accuracy"] for r in all_results]),
        "f1_macro": np.mean([r["f1_macro"] for r in all_results]),
        "f1_weighted": np.mean([r["f1_weighted"] for r in all_results]),
        "precision": np.mean([r["precision"] for r in all_results]),
        "recall": np.mean([r["recall"] for r in all_results]),
        "roc_auc": np.mean([r.get("roc_auc", 0) for r in all_results]),
    }

    with open(
        os.path.join(output_dir, f"fusion_{model_type}_cv_results.json"), "w"
    ) as f:
        json.dump(
            {"fold_results": all_results, "average_results": avg_results}, f, indent=2
        )

    logger.info(f"Cross-validation average: {avg_results}")
    return all_results, avg_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fusion Model Training")
    parser.add_argument("--speech-scores", required=True, help="Speech scores JSON")
    parser.add_argument(
        "--handwriting-scores", required=True, help="Handwriting scores JSON"
    )
    parser.add_argument("--text-scores", required=True, help="Text scores JSON")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument(
        "--model-type",
        default="mlp",
        choices=[
            "mlp",
            "attention",
            "gradient_boosting",
            "random_forest",
            "logistic_regression",
        ],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")

    args = parser.parse_args()

    X, y = load_modality_scores(
        args.speech_scores, args.handwriting_scores, args.text_scores, args.labels
    )

    if args.cv:
        cross_validate_fusion(X, y, args.model_type, output_dir=args.output)
    else:
        train_fusion_model(
            speech_scores_path=args.speech_scores,
            handwriting_scores_path=args.handwriting_scores,
            text_scores_path=args.text_scores,
            labels_path=args.labels,
            model_type=args.model_type,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output,
        )


if __name__ == "__main__":
    main()
