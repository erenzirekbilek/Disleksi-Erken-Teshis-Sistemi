import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    def __init__(self, model, model_type: str = "pytorch"):
        self.model = model
        self.model_type = model_type
        self.explainer = None

    def create_explainer(self, X_train: np.ndarray, feature_names: List[str] = None):
        import shap

        if self.model_type == "pytorch":
            self.explainer = shap.DeepExplainer(self.model, X_train)
        elif self.model_type == "sklearn":
            if hasattr(self.model, "predict_proba"):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, X_train)
        elif self.model_type == "transformers":
            self.explainer = shap.KernelExplainer(self.model.predict, X_train)

        self.feature_names = feature_names
        logger.info(f"Created {self.model_type} SHAP explainer")
        return self.explainer

    def explain_instance(self, X: np.ndarray) -> Dict:
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")

        import shap

        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        result = {
            "shap_values": shap_values.tolist()
            if hasattr(shap_values, "tolist")
            else shap_values,
            "base_value": self.explainer.expected_value,
            "feature_importance": self._get_feature_importance(shap_values, X),
        }

        return result

    def _get_feature_importance(
        self, shap_values: np.ndarray, X: np.ndarray
    ) -> List[Dict]:
        importance = np.abs(shap_values).mean(axis=0)

        if self.feature_names:
            features = self.feature_names
        else:
            features = [f"feature_{i}" for i in range(len(importance))]

        sorted_idx = np.argsort(importance)[::-1]

        return [
            {
                "feature": features[i],
                "importance": float(importance[i]),
                "contribution": float(shap_values[0, i])
                if shap_values.ndim > 1
                else float(shap_values[i]),
            }
            for i in sorted_idx[:20]
        ]

    def explain_dataset(self, X: np.ndarray, output_dir: str) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for i in range(min(len(X), 100)):
            try:
                result = self.explain_instance(X[i : i + 1])
                result["sample_index"] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to explain sample {i}: {e}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "shap_explanations.csv"), index=False)

        logger.info(f"Saved explanations for {len(results)} samples")
        return df

    def plot_feature_importance(
        self, X: np.ndarray, output_path: str, max_features: int = 20
    ):
        import shap

        if hasattr(self.explainer, "shap_values"):
            shap_values = self.explainer.shap_values(X[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            logger.warning("Explainer doesn't support shap_values")
            return

        plt.figure(figsize=(10, 8))

        if self.model_type == "pytorch" or self.model_type == "transformers":
            shap.summary_plot(
                shap_values,
                X[:100],
                feature_names=self.feature_names,
                max_display=max_features,
                show=False,
            )
        else:
            shap.summary_plot(
                shap_values,
                X[:100],
                feature_names=self.feature_names,
                max_display=max_features,
                show=False,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance plot to {output_path}")

    def plot_dependence(self, X: np.ndarray, feature_idx: int, output_path: str):
        import shap

        shap_values = self.explainer.shap_values(X[:200])
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        feature_name = (
            self.feature_names[feature_idx]
            if self.feature_names
            else f"feature_{feature_idx}"
        )

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X[:200],
            feature_names=self.feature_names,
            show=False,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved dependence plot to {output_path}")


class ModalityExplainer:
    def __init__(
        self, model_path: str, model_type: str, feature_names: List[str] = None
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.feature_names = feature_names
        self.model = None
        self.explainer = None

    def load_model(self):
        import torch

        if self.model_type == "pytorch":
            checkpoint = torch.load(self.model_path, map_location="cpu")
            logger.info(f"Loaded PyTorch model from {self.model_path}")
        else:
            logger.info(f"Model type {self.model_type} loaded")

    def explain(self, features: np.ndarray) -> Dict:
        raise NotImplementedError("Subclass must implement explain method")


class SpeechSHAPExplainer(ModalityExplainer):
    def __init__(self, model_path: str, feature_names: List[str] = None):
        super().__init__(
            model_path,
            "pytorch",
            feature_names
            or [
                "mfcc_mean",
                "pitch_mean",
                "energy_mean",
                "spectral_centroid",
                "tempo",
                "pause_count",
                "voiced_ratio",
                "zcr_mean",
            ],
        )

    def explain(self, features: np.ndarray) -> Dict:
        explainer = SHAPExplainer(self.model, "pytorch")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        background = np.zeros((10, features.shape[1]))
        explainer.create_explainer(background, self.feature_names)

        result = explainer.explain_instance(features)

        return {
            "modality": "speech",
            "shap_values": result["shap_values"],
            "top_features": result["feature_importance"][:10],
            "base_value": result["base_value"],
        }


class HandwritingSHAPExplainer(ModalityExplainer):
    def __init__(self, model_path: str, feature_names: List[str] = None):
        super().__init__(
            model_path,
            "pytorch",
            feature_names
            or [
                "character_count",
                "size_cv",
                "baseline_deviation",
                "spacing_mean",
                "reversal_count",
                "ink_ratio",
                "pressure_consistency",
            ],
        )

    def explain(self, features: np.ndarray) -> Dict:
        explainer = SHAPExplainer(self.model, "pytorch")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        background = np.zeros((10, features.shape[1]))
        explainer.create_explainer(background, self.feature_names)

        result = explainer.explain_instance(features)

        return {
            "modality": "handwriting",
            "shap_values": result["shap_values"],
            "top_features": result["feature_importance"][:10],
            "base_value": result["base_value"],
        }


class TextSHAPExplainer(ModalityExplainer):
    def __init__(self, model_path: str, feature_names: List[str] = None):
        super().__init__(
            model_path,
            "transformers",
            feature_names
            or [
                "spelling_error_rate",
                "grammar_score",
                "flesch_reading_ease",
                "type_token_ratio",
                "avg_sentence_length",
                "complex_word_ratio",
            ],
        )

    def explain(self, features: np.ndarray) -> Dict:
        explainer = SHAPExplainer(self.model, "transformers")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        background = np.zeros((10, features.shape[1]))
        explainer.create_explainer(background, self.feature_names)

        result = explainer.explain_instance(features)

        return {
            "modality": "text",
            "shap_values": result["shap_values"],
            "top_features": result["feature_importance"][:10],
            "base_value": result["base_value"],
        }


class ExplainerFactory:
    @staticmethod
    def create_explainer(
        modality: str, model_path: str, feature_names: List[str] = None
    ):
        if modality == "speech":
            return SpeechSHAPExplainer(model_path, feature_names)
        elif modality == "handwriting":
            return HandwritingSHAPExplainer(model_path, feature_names)
        elif modality == "text":
            return TextSHAPExplainer(model_path, feature_names)
        else:
            raise ValueError(f"Unknown modality: {modality}")


def generate_global_importance_report(
    speech_model: str, handwriting_model: str, text_model: str, output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    report = {"speech": {}, "handwriting": {}, "text": {}}

    for modality, model_path in [
        ("speech", speech_model),
        ("handwriting", handwriting_model),
        ("text", text_model),
    ]:
        if os.path.exists(model_path):
            explainer = ExplainerFactory.create_explainer(modality, model_path)
            explainer.load_model()
            report[modality]["status"] = "ready"
        else:
            report[modality]["status"] = "model_not_found"

    with open(os.path.join(output_dir, "global_importance_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Global importance report saved to {output_dir}")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SHAP Explainability")
    parser.add_argument(
        "--modality", required=True, choices=["speech", "handwriting", "text"]
    )
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--features", required=True, help="Features JSON file")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    with open(args.features, "r") as f:
        features_data = json.load(f)

    features = np.array(features_data.get("features", [features_data]))

    explainer = ExplainerFactory.create_explainer(args.modality, args.model)
    explainer.load_model()

    result = explainer.explain(features)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, f"{args.modality}_explanation.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"Explanation saved to {args.output}")


if __name__ == "__main__":
    main()
