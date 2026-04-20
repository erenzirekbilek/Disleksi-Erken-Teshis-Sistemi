import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
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


def build_cnn_model(input_shape: Tuple[int, int, int] = (13, 87, 1), n_classes: int = 3):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SpeechCNN_Keras")
    return model


def build_lstm_model(input_features: int, n_classes: int = 3):
    inputs = layers.Input(shape=(None, input_features))
    
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2))(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SpeechBiLSTM_Keras")
    return model


def build_gru_model(input_features: int, n_classes: int = 3):
    inputs = layers.Input(shape=(None, input_features))
    
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.GRU(32, dropout=0.2))(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SpeechBiGRU_Keras")
    return model


def build_transformer_model(input_features: int, n_classes: int = 3, seq_length: int = 100):
    inputs = layers.Input(shape=(seq_length, input_features))
    
    x = layers.PositionalEmbedding(input_dim=seq_length, output_dim=input_features)(inputs)
    
    for _ in range(4):
        attn = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dense(input_features)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SpeechTransformer_Keras")
    return model


def prepare_sequences(features: np.ndarray, max_length: int = 100) -> np.ndarray:
    if features.shape[1] > max_length:
        return features[:, :max_length, :]
    else:
        padded = np.zeros((features.shape[0], max_length, features.shape[2]))
        padded[:, :features.shape[1], :] = features
        return padded


class KerasSpeechModelTrainer:
    def __init__(
        self,
        model_type: str = "cnn",
        input_features: int = 13,
        n_classes: int = 3,
        learning_rate: float = 1e-4,
    ):
        self.model_type = model_type
        self.input_features = input_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self, input_shape: Tuple[int, int, int] = None):
        if self.model_type == "cnn":
            if input_shape is None:
                input_shape = (13, 87, 1)
            self.model = build_cnn_model(input_shape, self.n_classes)
        elif self.model_type == "lstm":
            self.model = build_lstm_model(self.input_features, self.n_classes)
        elif self.model_type == "gru":
            self.model = build_gru_model(self.input_features, self.n_classes)
        elif self.model_type == "transformer":
            self.model = build_transformer_model(self.input_features, self.n_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model.summary()
        logger.info(f"Built Keras {self.model_type} model")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
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

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs

    def save_model(self, path: str):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def extract_mfcc_features(audio_paths: List[str], max_length: int = 100) -> np.ndarray:
    import librosa
    
    features_list = []
    for path in audio_paths:
        try:
            y, sr = librosa.load(path, duration=10)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            if mfcc.shape[1] < max_length:
                padded = np.zeros((13, max_length))
                padded[:, :mfcc.shape[1]] = mfcc
                mfcc = padded
            else:
                mfcc = mfcc[:, :max_length]
            
            features_list.append(mfcc.T)
        except Exception as e:
            logger.warning(f"Error processing {path}: {e}")
            features_list.append(np.zeros((max_length, 13)))
    
    return np.array(features_list)


def train_speech_model_keras(
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
        X_train, y_train, test_size=val_size / (1 - test_size), stratify=y_train, random_state=42
    )

    input_features = X.shape[1]
    
    trainer = KerasSpeechModelTrainer(
        model_type=model_type,
        input_features=input_features,
        n_classes=3
    )
    
    if model_type == "cnn":
        cnn_input_shape = (13, 87, 1)
        X_train_cnn = X_train.reshape(-1, 13, 87, 1) if X_train.shape[1] == 1017 else X_train.reshape(-1, 13, 100, 1)
        X_val_cnn = X_val.reshape(-1, 13, 87, 1) if X_val.shape[1] == 1017 else X_val.reshape(-1, 13, 100, 1)
        X_test_cnn = X_test.reshape(-1, 13, 87, 1) if X_test.shape[1] == 1017 else X_test.reshape(-1, 13, 100, 1)
        trainer.build_model(input_shape=cnn_input_shape)
        trainer.train(X_train_cnn, y_train, X_val_cnn, y_val, epochs=epochs, batch_size=batch_size)
        results = trainer.evaluate(X_test_cnn, y_test)
    else:
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(-1, X_train.shape[1], 1)
            X_val = X_val.reshape(-1, X_val.shape[1], 1)
            X_test = X_test.reshape(-1, X_test.shape[1], 1)
        
        trainer.build_model()
        trainer.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
        results = trainer.evaluate(X_test, y_test)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"speech_{model_type}_keras_model.h5")
    trainer.save_model(model_path)

    with open(os.path.join(output_dir, f"speech_{model_type}_keras_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech Model Training (Keras)")
    parser.add_argument("--features", required=True, help="Features JSON file")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "lstm", "gru", "transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_speech_model_keras(
        features_path=args.features,
        labels_path=args.labels,
        model_type=args.model_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )