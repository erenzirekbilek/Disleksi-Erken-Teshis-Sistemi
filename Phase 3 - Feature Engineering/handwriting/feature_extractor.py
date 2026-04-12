import os
import glob
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HandwritingFeatureExtractor:
    def __init__(
        self,
        min_component_area: int = 50,
        max_components: int = 500,
        baseline_sample_rate: int = 100,
    ):
        self.min_component_area = min_component_area
        self.max_components = max_components
        self.baseline_sample_rate = baseline_sample_rate

    def load_image(self, image_path: str) -> np.ndarray:
        try:
            import cv2

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            logger.info(f"Loaded image: {image_path}, shape: {img.shape}")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def segment_characters(self, img: np.ndarray) -> Dict:
        import cv2

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        components = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_component_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            components.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": float(area),
                }
            )

        components = sorted(components, key=lambda c: c["x"])

        logger.info(f"Segmented {len(components)} characters")
        return {
            "character_count": len(components),
            "characters": components[: self.max_components],
        }

    def analyze_sizes(self, components: List[Dict]) -> Dict:
        if not components:
            return {
                "size_mean": 0,
                "size_std": 0,
                "size_cv": 0,
                "size_min": 0,
                "size_max": 0,
                "size_range": 0,
            }

        areas = [c["area"] for c in components]
        widths = [c["width"] for c in components]
        heights = [c["height"] for c in components]

        areas = np.array(areas)
        widths = np.array(widths)
        heights = np.array(heights)

        size_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0

        features = {
            "size_mean": float(np.mean(areas)),
            "size_std": float(np.std(areas)),
            "size_cv": float(size_cv),
            "size_min": float(np.min(areas)),
            "size_max": float(np.max(areas)),
            "size_range": float(np.max(areas) - np.min(areas)),
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "height_mean": float(np.mean(heights)),
            "height_std": float(np.std(heights)),
            "aspect_ratio_mean": float(np.mean(widths / heights))
            if len(widths) > 0
            else 0,
            "aspect_ratio_std": float(np.std(widths / heights))
            if len(widths) > 0
            else 0,
        }

        logger.info("Analyzed character sizes")
        return features

    def analyze_baseline(self, img: np.ndarray, components: List[Dict]) -> Dict:
        if not components:
            return {
                "baseline_deviation": 0,
                "baseline_consistency": 0,
                "baseline_y_mean": 0,
            }

        y_positions = [c["y"] + c["height"] for c in components]
        y_positions = np.array(y_positions)

        mean_y = np.mean(y_positions)
        std_y = np.std(y_positions)

        img_height = img.shape[0]
        deviation_ratio = std_y / img_height if img_height > 0 else 0

        features = {
            "baseline_y_mean": float(mean_y),
            "baseline_deviation": float(std_y),
            "baseline_deviation_ratio": float(deviation_ratio),
            "baseline_consistency": float(1 - min(deviation_ratio, 1)),
            "baseline_y_min": float(np.min(y_positions)),
            "baseline_y_max": float(np.max(y_positions)),
            "baseline_y_range": float(np.max(y_positions) - np.min(y_positions)),
        }

        logger.info("Analyzed baseline adherence")
        return features

    def analyze_spacing(self, components: List[Dict], img_width: int) -> Dict:
        if len(components) < 2:
            return {
                "spacing_mean": 0,
                "spacing_std": 0,
                "spacing_cv": 0,
                "spacing_min": 0,
                "spacing_max": 0,
            }

        x_positions = [c["x"] + c["width"] for c in components[:-1]]
        next_x = [c["x"] for c in components[1:]]

        gaps = np.array(next_x) - np.array(x_positions)
        gaps = gaps[gaps > 0]

        if len(gaps) == 0:
            return {
                "spacing_mean": 0,
                "spacing_std": 0,
                "spacing_cv": 0,
                "spacing_min": 0,
                "spacing_max": 0,
            }

        features = {
            "spacing_mean": float(np.mean(gaps)),
            "spacing_std": float(np.std(gaps)),
            "spacing_cv": float(np.std(gaps) / np.mean(gaps))
            if np.mean(gaps) > 0
            else 0,
            "spacing_min": float(np.min(gaps)),
            "spacing_max": float(np.max(gaps)),
            "spacing_range": float(np.max(gaps) - np.min(gaps)),
            "spacing_irregularity_count": int(
                np.sum(np.abs(gaps - np.mean(gaps)) > 2 * np.std(gaps))
            ),
        }

        logger.info("Analyzed spacing")
        return features

    def detect_reversals(self, img: np.ndarray, components: List[Dict]) -> Dict:
        import cv2

        reversal_patterns = ["b", "d", "p", "q", "m", "w", "n", "u"]
        reversal_counts = {p: 0 for p in reversal_patterns}

        for component in components:
            x, y, w, h = (
                component["x"],
                component["y"],
                component["width"],
                component["height"],
            )

            if w < 5 or h < 5:
                continue

            roi = img[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            roi_binary = roi > 127

            left_half = (
                roi_binary[:, : w // 2] if w > 2 else np.zeros((h, 1), dtype=bool)
            )
            right_half = (
                roi_binary[:, w // 2 :] if w > 2 else np.zeros((h, 1), dtype=bool)
            )

            left_density = np.mean(left_half) if left_half.size > 0 else 0
            right_density = np.mean(right_half) if right_half.size > 0 else 0

            aspect_ratio = w / h if h > 0 else 0

            if aspect_ratio > 0.6 and aspect_ratio < 1.4:
                if abs(left_density - right_density) > 0.3:
                    if left_density > right_density:
                        reversal_counts["b"] += 1
                    else:
                        reversal_counts["d"] += 1

            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                if w > h * 0.8:
                    if left_density > right_density:
                        reversal_counts["p"] += 1
                    else:
                        reversal_counts["q"] += 1

        total_reversals = sum(reversal_counts.values())

        features = {
            "reversal_count": total_reversals,
            "reversal_b_count": reversal_counts["b"],
            "reversal_d_count": reversal_counts["d"],
            "reversal_p_count": reversal_counts["p"],
            "reversal_q_count": reversal_counts["q"],
            "reversal_other_count": total_reversals
            - reversal_counts["b"]
            - reversal_counts["d"]
            - reversal_counts["p"]
            - reversal_counts["q"],
            "reversal_ratio": float(total_reversals / len(components))
            if components
            else 0,
        }

        logger.info(f"Detected {total_reversals} potential reversals")
        return features

    def analyze_strokes(self, img: np.ndarray, components: List[Dict]) -> Dict:
        import cv2

        total_pixels = img.size
        ink_pixels = np.sum(img < 127)
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0

        stroke_density = ink_pixels / len(components) if components else 0

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        perimeters = [
            cv2.arcLength(c, True) for c in contours if cv2.arcLength(c, True) > 10
        ]

        compactness = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                compactness.append(4 * np.pi * area / (perimeter**2))

        features = {
            "ink_ratio": float(ink_ratio),
            "stroke_density": float(stroke_density),
            "total_contours": len(contours),
            "mean_perimeter": float(np.mean(perimeters)) if perimeters else 0,
            "perimeter_std": float(np.std(perimeters)) if perimeters else 0,
            "compactness_mean": float(np.mean(compactness)) if compactness else 0,
            "compactness_std": float(np.std(compactness)) if compactness else 0,
        }

        logger.info("Analyzed stroke patterns")
        return features

    def analyze_pressure(self, img: np.ndarray, components: List[Dict]) -> Dict:
        import cv2

        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        pixel_intensities = img[binary > 0] if np.any(binary > 0) else np.array([0])

        features = {
            "pressure_mean": float(255 - np.mean(pixel_intensities))
            if len(pixel_intensities) > 0
            else 0,
            "pressure_std": float(np.std(pixel_intensities))
            if len(pixel_intensities) > 0
            else 0,
            "pressure_min": float(255 - np.max(pixel_intensities))
            if len(pixel_intensities) > 0
            else 0,
            "pressure_max": float(255 - np.min(pixel_intensities))
            if len(pixel_intensities) > 0
            else 0,
            "pressure_range": float(
                255 - np.min(pixel_intensities) - (255 - np.max(pixel_intensities))
            )
            if len(pixel_intensities) > 0
            else 0,
            "pressure_consistency": float(1 - np.std(pixel_intensities) / 255)
            if len(pixel_intensities) > 0
            else 0,
        }

        logger.info("Analyzed pressure consistency")
        return features

    def extract_all(self, image_path: str) -> Dict:
        img = self.load_image(image_path)

        features = {}
        features["image_path"] = image_path
        features["image_width"] = img.shape[1]
        features["image_height"] = img.shape[0]

        segmentation = self.segment_characters(img)
        components = segmentation["characters"]

        features["character_count"] = segmentation["character_count"]

        features.update(self.analyze_sizes(components))
        features.update(self.analyze_baseline(img, components))
        features.update(self.analyze_spacing(components, img.shape[1]))
        features.update(self.detect_reversals(img, components))
        features.update(self.analyze_strokes(img, components))
        features.update(self.analyze_pressure(img, components))

        logger.info(f"Total features extracted: {len(features)}")
        return features

    def process_file(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        features = self.extract_all(image_path)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved features to: {output_path}")

        return features

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".tif"],
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for ext in extensions:
            image_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
            image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        image_files = list(set(image_files))

        for image_path in image_files:
            try:
                filename = Path(image_path).stem
                output_path = os.path.join(output_dir, f"{filename}_features.json")

                features = self.process_file(image_path, output_path)

                result = {
                    "filename": filename,
                    "status": "success",
                    "output_path": output_path,
                    "character_count": features.get("character_count", 0),
                    "size_cv": features.get("size_cv", 0),
                    "reversal_count": features.get("reversal_count", 0),
                }
                results.append(result)
                logger.info(f"Processed: {filename}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(
                    {
                        "filename": Path(image_path).stem,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "feature_extraction_report.csv")
        df.to_csv(report_path, index=False)
        logger.info(f"Report saved to: {report_path}")

        return df


def flatten_features(features: Dict, prefix: str = "") -> Dict:
    flat = {}
    for key, value in features.items():
        if isinstance(value, list) and len(value) > 10:
            continue
        if isinstance(value, dict):
            flat.update(flatten_features(value, f"{prefix}{key}_"))
        elif isinstance(value, (int, float, str)):
            flat[f"{prefix}{key}"] = value
    return flat


def create_feature_matrix(
    features_dir: str, output_path: Optional[str] = None
) -> pd.DataFrame:
    feature_files = glob.glob(os.path.join(features_dir, "*.json"))

    all_features = []
    for f in feature_files:
        with open(f, "r") as fp:
            features = json.load(fp)

        flat_features = flatten_features(features)
        all_features.append(flat_features)

    df = pd.DataFrame(all_features)

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Feature matrix saved to: {output_path}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Handwriting Feature Extraction")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "--min-area", type=int, default=50, help="Minimum component area"
    )
    parser.add_argument("--matrix", action="store_true", help="Create feature matrix")

    args = parser.parse_args()

    extractor = HandwritingFeatureExtractor(min_component_area=args.min_area)

    if os.path.isdir(args.input):
        df = extractor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(df)} files")
        print(df["status"].value_counts())

        if args.matrix and args.output:
            create_feature_matrix(args.output)
    else:
        result = extractor.process_file(args.input, args.output)
        print(f"Extracted {len(result)} feature groups")
        print(f"Character count: {result.get('character_count', 0)}")


if __name__ == "__main__":
    main()
