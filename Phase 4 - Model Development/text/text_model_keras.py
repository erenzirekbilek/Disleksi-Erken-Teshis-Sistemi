import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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


class TextDatasetKeras:
    def __init__(self, texts, labels, max_words: int = 10000, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(texts)

    def get_sequences(self):
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        return pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")

    def get_data(self):
        X = self.get_sequences()
        y = np.array(self.labels)
        return X, y


def build_bert_like_model(max_length: int = 512, vocab_size: int = 10000, n_classes: int = 3):
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    
    embedding = layers.Embedding(vocab_size, 128)(inputs)
    
    x = layers.Conv1D(256, 5, activation='relu', padding='same')(embedding)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="TextCNN_BERT_like")
    return model


def build_lstm_model(max_length: int = 512, vocab_size: int = 10000, n_classes: int = 3):
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    
    embedding = layers.Embedding(vocab_size, 128)(inputs)
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="TextBiLSTM")
    return model


def build_gru_model(max_length: int = 512, vocab_size: int = 10000, n_classes: int = 3):
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    
    embedding = layers.Embedding(vocab_size, 128)(inputs)
    
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.2))(embedding)
    x = layers.Bidirectional(layers.GRU(32, dropout=0.2))(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="TextBiGRU")
    return model


class KerasTextModelTrainer:
    def __init__(
        self,
        model_type: str = "cnn",
        max_words: int = 10000,
        max_length: int = 512,
        n_classes: int = 3,
        learning_rate: float = 1e-4,
    ):
        self.model_type = model_type
        self.max_words = max_words
        self.max_length = max_length
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.tokenizer = None

    def build_model(self):
        if self.model_type == "cnn":
            self.model = build_bert_like_model(self.max_length, self.max_words, self.n_classes)
        elif self.model_type == "lstm":
            self.model = build_lstm_model(self.max_length, self.max_words, self.n_classes)
        elif self.model_type == "gru":
            self.model = build_gru_model(self.max_length, self.max_words, self.n_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
        logger.info(f"Built Keras {self.model_type} model")

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 10, batch_size: int = 16):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def evaluate(self, X_test, y_test) -> Dict:
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        
        if self.n_classes == 3:
            try:
                results["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
            except:
                results["roc_auc"] = None
        
        return results

    def predict(self, texts: list) -> Tuple[np.ndarray, np.ndarray]:
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
        
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs

    def save_model(self, path: str):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def train_text_model_keras(
    text_file: str,
    labels_path: str,
    model_type: str = "cnn",
    max_words: int = 10000,
    max_length: int = 512,
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 16,
    epochs: int = 10,
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
        X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=42
    )

    trainer = KerasTextModelTrainer(model_type=model_type, max_words=max_words, max_length=max_length)
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    trainer.tokenizer = tokenizer
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding="post", truncating="post")
    
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding="post", truncating="post")
    
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding="post", truncating="post")

    trainer.build_model()

    logger.info("Training text model (Keras)...")
    trainer.train(X_train_pad, y_train, X_val_pad, y_val, epochs=epochs, batch_size=batch_size)

    logger.info("Evaluating model...")
    results = trainer.evaluate(X_test_pad, y_test)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"text_{model_type}_keras_model.h5")
    trainer.save_model(model_path)

    with open(os.path.join(output_dir, f"text_{model_type}_keras_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text Model Training (Keras)")
    parser.add_argument("--text", required=True, help="Text file")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "lstm", "gru"])
    parser.add_argument("--max-words", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_text_model_keras(
        text_file=args.text,
        labels_path=args.labels,
        model_type=args.model_type,
        max_words=args.max_words,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )