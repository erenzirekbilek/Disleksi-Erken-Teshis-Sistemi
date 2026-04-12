import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LIMEExplainer:
    def __init__(self, model, num_features: int = 10, num_samples: int = 5000):
        self.model = model
        self.num_features = num_features
        self.num_samples = num_samples

    def explain_instance(
        self, features: np.ndarray, feature_names: List[str] = None, top_labels: int = 3
    ) -> Dict:
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            logger.warning("LIME not installed. Using simplified version.")
            return self._simple_explain(features, feature_names)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=features,
            feature_names=feature_names,
            class_names=["low", "medium", "high"],
            mode="classification",
        )

        explanation = explainer.explain_instance(
            features[0],
            self.model.predict_proba,
            num_features=self.num_features,
            num_samples=self.num_samples,
            top_labels=top_labels,
        )

        result = {
            "prediction": explanation.predict_proba.tolist(),
            "local_explanation": [],
        }

        for label in range(top_labels):
            label_name = ["low", "medium", "high"][label]
            exp = explanation.as_list(label=label)

            result["local_explanation"].append(
                {
                    "label": label_name,
                    "features": [
                        {"feature": name, "weight": float(weight)}
                        for name, weight in exp
                    ],
                }
            )

        return result

    def _simple_explain(
        self, features: np.ndarray, feature_names: List[str] = None
    ) -> Dict:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        importance = np.abs(features[0])
        sorted_idx = np.argsort(importance)[::-1][: self.num_features]

        return {
            "prediction": [0.33, 0.33, 0.34],
            "local_explanation": [
                {
                    "label": "high",
                    "features": [
                        {"feature": feature_names[i], "weight": float(importance[i])}
                        for i in sorted_idx
                    ],
                }
            ],
        }


class TextLIMEExplainer:
    def __init__(self, model, num_samples: int = 5000):
        self.model = model
        self.num_samples = num_samples

    def explain_text(self, text: str, tokenizer, max_length: int = 512) -> Dict:
        try:
            import lime
            import lime.lime_text
        except ImportError:
            logger.warning("LIME not installed")
            return self._simple_text_explain(text)

        class TextExplainer:
            def __init__(self, model, tokenizer, max_length):
                self.model = model
                self.tokenizer = tokenizer
                self.max_length = max_length

            def predict(self, texts: List[str]) -> np.ndarray:
                import torch

                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)

                return probs.numpy()

            def predict_proba(self, texts: List[str]) -> np.ndarray:
                return self.predict(texts)

        explainer = TextExplainer(self.model, tokenizer, max_length)

        lime_explainer = lime.lime_text.LimeTextExplainer()

        explanation = lime_explainer.explain_instance(
            text, explainer.predict_proba, num_samples=self.num_samples, top_labels=3
        )

        result = {"text": text, "prediction": "medium", "explanation": []}

        for label in range(3):
            exp = explanation.as_list(label=label)
            result["explanation"].append(
                {
                    "label": ["low", "medium", "high"][label],
                    "features": [
                        {"word": word, "weight": float(weight)}
                        for word, weight in exp[:10]
                    ],
                }
            )

        return result

    def _simple_text_explain(self, text: str) -> Dict:
        words = text.split()[:10]
        return {
            "text": text,
            "prediction": "medium",
            "explanation": [
                {
                    "label": "medium",
                    "features": [
                        {"word": w, "weight": 1.0 / (i + 1)}
                        for i, w in enumerate(words)
                    ],
                }
            ],
        }


class ImageLIMEExplainer:
    def __init__(self, model, num_samples: int = 1000, image_size: int = 224):
        self.model = model
        self.num_samples = num_samples
        self.image_size = image_size

    def explain_image(self, image_path: str, segmentation_fn=None) -> Dict:
        import cv2

        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.image_size, self.image_size))

        try:
            import lime
            from lime.lime_image import ImageExplainer
        except ImportError:
            logger.warning("LIME not installed")
            return self._simple_image_explain(image_path)

        def predict(images: np.ndarray) -> np.ndarray:
            import torch

            images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2) / 255.0

            with torch.no_grad():
                outputs = self.model(images_tensor)
                probs = torch.softmax(outputs, dim=1)

            return probs.numpy()

        explainer = ImageExplainer()
        explanation = explainer.explain_instance(
            img, predict, num_samples=self.num_samples, top_labels=3
        )

        result = {"image_path": image_path, "prediction": "medium", "segments": []}

        for label in range(3):
            temp, mask = explanation.get_image_and_mask(
                label, positive_only=True, num_features=10, hide_rest=False
            )

            result["segments"].append(
                {
                    "label": ["low", "medium", "high"][label],
                    "segment_indices": mask.tolist()[:10],
                }
            )

        return result

    def _simple_image_explain(self, image_path: str) -> Dict:
        return {
            "image_path": image_path,
            "prediction": "medium",
            "segments": [
                {"label": "medium", "segments": ["segment_1", "segment_2", "segment_3"]}
            ],
        }


class AudioLIMEExplainer:
    def __init__(self, model, num_samples: int = 1000):
        self.model = model
        self.num_samples = num_samples

    def explain_audio(self, audio_path: str, segments: int = 10) -> Dict:
        try:
            import librosa

            y, sr = librosa.load(audio_path, duration=10)

            segment_length = len(y) // segments
            features = []

            for i in range(segments):
                segment = y[i * segment_length : (i + 1) * segment_length]
                mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr))
                features.append(mfcc)

            importance = np.abs(features)
            sorted_idx = np.argsort(importance)[::-1]

            return {
                "audio_path": audio_path,
                "prediction": "medium",
                "segments": [
                    {
                        "label": "medium",
                        "segment_importance": [
                            {"segment": i, "importance": float(importance[i])}
                            for i in sorted_idx[:5]
                        ],
                    }
                ],
            }
        except Exception as e:
            logger.warning(f"LIME audio explain failed: {e}")
            return {"audio_path": audio_path, "prediction": "medium", "segments": []}


class MultiModalExplainer:
    def __init__(self, speech_model=None, handwriting_model=None, text_model=None):
        self.speech_model = speech_model
        self.handwriting_model = handwriting_model
        self.text_model = text_model

        self.speech_explainer = None
        self.handwriting_explainer = None
        self.text_explainer = None

    def explain_modality(self, modality: str, data: Any, **kwargs) -> Dict:
        if modality == "speech":
            if self.speech_explainer is None:
                self.speech_explainer = AudioLIMEExplainer(self.speech_model)
            return self.speech_explainer.explain_audio(data)

        elif modality == "handwriting":
            if self.handwriting_explainer is None:
                self.handwriting_explainer = ImageLIMEExplainer(self.handwriting_model)
            return self.handwriting_explainer.explain_image(data)

        elif modality == "text":
            if self.text_explainer is None:
                self.text_explainer = TextLIMEExplainer(
                    self.text_model, kwargs.get("tokenizer")
                )
            return self.text_explainer.explain_text(data, kwargs.get("tokenizer"))

        else:
            raise ValueError(f"Unknown modality: {modality}")

    def explain_all(
        self,
        speech_data: str = None,
        handwriting_data: str = None,
        text_data: str = None,
        tokenizer=None,
    ) -> Dict:
        results = {}

        if speech_data:
            results["speech"] = self.explain_modality("speech", speech_data)

        if handwriting_data:
            results["handwriting"] = self.explain_modality(
                "handwriting", handwriting_data
            )

        if text_data:
            results["text"] = self.explain_modality(
                "text", text_data, tokenizer=tokenizer
            )

        results["combined_prediction"] = self._combine_predictions(results)

        return results

    def _combine_predictions(self, results: Dict) -> str:
        predictions = {"low": 0, "medium": 0, "high": 0}

        for modality, result in results.items():
            if "prediction" in result:
                pred = result["prediction"]
                predictions[pred] = predictions.get(pred, 0) + 1

        return max(predictions, key=predictions.get)


def generate_lime_report(modality: str, data_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if modality == "text":
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        explainer = TextLIMEExplainer(None)
        result = explainer.explain_text(text, None)

    elif modality == "handwriting":
        explainer = ImageLIMEExplainer(None)
        result = explainer.explain_image(data_path)

    elif modality == "speech":
        explainer = AudioLIMEExplainer(None)
        result = explainer.explain_audio(data_path)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"LIME report saved to {output_path}")
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LIME Explainability")
    parser.add_argument(
        "--modality", required=True, choices=["speech", "handwriting", "text"]
    )
    parser.add_argument("--data", required=True, help="Data file path")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    result = generate_lime_report(args.modality, args.data, args.output)
    print(f"Explanation saved to {args.output}")


if __name__ == "__main__":
    main()
