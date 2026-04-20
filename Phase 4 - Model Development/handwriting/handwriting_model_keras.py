import os
import glob
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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


def build_cnn_model(input_shape: Tuple[int, int, int] = (224, 224, 1), n_classes: int = 3):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
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
    
    model = Model(inputs=inputs, outputs=outputs, name="HandwritingCNN_Keras")
    return model


def build_resnet_model(input_shape: Tuple[int, int, int] = (224, 224, 1), n_classes: int = 3):
    base_model = keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (1, 1))(inputs)
    x = base_model(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="HandwritingResNet_Keras")
    return model


def build_efficientnet_model(input_shape: Tuple[int, int, int] = (224, 224, 1), n_classes: int = 3):
    base_model = keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="HandwritingEfficientNet_Keras")
    return model


def build_vit_model(image_size: int = 224, patch_size: int = 16, n_classes: int = 3):
    num_patches = (image_size // patch_size) ** 2
    
    inputs = layers.Input(shape=(image_size, image_size, 1))
    
    x = layers.Conv2D(64, (patch_size, patch_size), strides=(patch_size, patch_size))(inputs)
    x = layers.Reshape((num_patches, 64))(x)
    
    for _ in range(8):
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dense(64)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="HandwritingViT_Keras")
    return model


def load_and_preprocess_images(image_paths: List[str], image_size: int = 224) -> np.ndarray:
    import cv2
    
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (image_size, image_size))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
        else:
            images.append(np.zeros((image_size, image_size, 1)))
    
    return np.array(images)


class KerasHandwritingModelTrainer:
    def __init__(
        self,
        model_type: str = "cnn",
        image_size: int = 224,
        n_classes: int = 3,
        learning_rate: float = 1e-4,
    ):
        self.model_type = model_type
        self.image_size = image_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self):
        input_shape = (self.image_size, self.image_size, 1)
        
        if self.model_type == "cnn":
            self.model = build_cnn_model(input_shape, self.n_classes)
        elif self.model_type == "resnet":
            self.model = build_resnet_model(input_shape, self.n_classes)
        elif self.model_type == "efficientnet":
            self.model = build_efficientnet_model(input_shape, self.n_classes)
        elif self.model_type == "vit":
            self.model = build_vit_model(self.image_size, n_classes=self.n_classes)
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
        
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )
        
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
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

    def predict(self, image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        images = load_and_preprocess_images(image_paths, self.image_size)
        
        probs = self.model.predict(images)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs

    def save_model(self, path: str):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


def train_handwriting_model_keras(
    image_dir: str,
    labels_path: str,
    model_type: str = "cnn",
    image_size: int = 224,
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
        random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        stratify=y_train,
        random_state=42
    )

    logger.info("Loading and preprocessing images...")
    X_train_images = load_and_preprocess_images(X_train, image_size)
    X_val_images = load_and_preprocess_images(X_val, image_size)
    X_test_images = load_and_preprocess_images(X_test, image_size)

    trainer = KerasHandwritingModelTrainer(
        model_type=model_type,
        image_size=image_size,
        n_classes=3
    )
    trainer.build_model()

    logger.info("Training handwriting model (Keras)...")
    trainer.train(X_train_images, y_train, X_val_images, y_val, epochs=epochs, batch_size=batch_size)

    logger.info("Evaluating model...")
    results = trainer.evaluate(X_test_images, y_test)

    logger.info(f"Results: {results}")

    model_path = os.path.join(output_dir, f"handwriting_{model_type}_keras_model.h5")
    trainer.save_model(model_path)

    with open(os.path.join(output_dir, f"handwriting_{model_type}_keras_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return trainer, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Handwriting Model Training (Keras)")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", required=True, help="Labels CSV file")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "resnet", "efficientnet", "vit"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_handwriting_model_keras(
        image_dir=args.images,
        labels_path=args.labels,
        model_type=args.model_type,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output,
    )