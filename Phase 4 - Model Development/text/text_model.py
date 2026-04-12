import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
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


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        n_classes: int = 3,
        freeze_bert: bool = False,
    ):
        super(BERTClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits


class DistilBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_classes: int = 3,
        freeze_layers: int = 0,
    ):
        super(DistilBERTClassifier, self).__init__()

        from transformers import DistilBertModel

        self.bert = DistilBertModel.from_pretrained(model_name)

        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)

        return logits


class TextModelTrainer:
    def __init__(
        self,
        model_type: str = "bert",
        model_name: str = "bert-base-uncased",
        n_classes: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def build_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.model_type == "bert":
            self.model = BERTClassifier(
                model_name=self.model_name, n_classes=self.n_classes
            )
        elif self.model_type == "distilbert":
            self.model = DistilBERTClassifier(
                model_name=self.model_name, n_classes=self.n_classes
            )
        elif self.model_type == "transformers":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.n_classes
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)
        logger.info(f"Built {self.model_type} model: {self.model_name}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        patience: int = 2,
    ):
        total_steps = len(train_loader) * epochs

        optimizer = AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

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
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
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

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

        return preds.cpu().numpy(), probs.cpu().numpy()

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
                "model_type": self.model_type,
                "model_name": self.model_name,
            },
            path,
        )
        self.tokenizer.save_pretrained(path.replace(".pt", "_tokenizer"))
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            path.replace(".pt", "_tokenizer")
        )
        logger.info(f"Model loaded from {path}")


def train_text_model(
    text_file: str,
    labels_path: str,
    model_type: str = "bert",
    model_name: str = "bert-base-uncased",
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 16,
    epochs: int = 5,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)

    with open(text_file, "r", encoding="utf-8") as f:
        texts = f.read().split("\n\n")
    texts = [t.strip() for t in texts if t.strip()]

    labels_df = pd.read_csv(labels_path)
    labels = labels_df["overall_risk"].map({"low": 0, "medium": 1, "high": 2}).values

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=42,
    )

    trainer = TextModelTrainer(model_type=model_type, model_name=model_name)
    trainer.build_model()

    train_dataset = TextDataset(X_train, y_train, trainer.tokenizer)
    val_dataset = TextDataset(X_val, y_val, trainer.tokenizer)
    test_dataset = TextDataset(X_test, y_test, trainer.tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logger.info("Training text model...")
    trainer.train(train_loader, val_loader, epochs=epochs, patience=2)

    logger.info("Evaluating model...")
    results = trainer.evaluate(test_loader)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"text_{model_type}_model.pt")
    trainer.save_checkpoint(model_path)

    with open(os.path.join(output_dir, f"text_{model_type}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text Model Training")
    parser.add_argument("--text", required=True, help="Text file")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument(
        "--model-type", default="bert", choices=["bert", "distilbert", "transformers"]
    )
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_text_model(
        text_file=args.text,
        labels_path=args.labels,
        model_type=args.model_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
