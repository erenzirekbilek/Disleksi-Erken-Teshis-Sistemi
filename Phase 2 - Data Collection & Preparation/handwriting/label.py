import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    label: str
    confidence: float = 1.0


@dataclass
class HandwritingLabels:
    student_hash: str
    image_path: str
    task_type: str

    letter_reversals: str = "none"
    letter_reversals_boxes: List[Dict] = None
    letter_reversals_notes: str = ""

    spacing_irregularity: str = "none"
    spacing_irregularity_boxes: List[Dict] = None
    spacing_notes: str = ""

    character_misplacement: str = "none"
    character_misplacement_boxes: List[Dict] = None
    misplacement_notes: str = ""

    baseline_adherence: str = "none"
    baseline_notes: str = ""

    size_consistency: str = "none"
    size_notes: str = ""

    overall_risk: str = "low"
    overall_score: float = 0.0

    annotator_id: str = ""
    annotation_date: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.letter_reversals_boxes is None:
            result["letter_reversals_boxes"] = []
        if self.spacing_irregularity_boxes is None:
            result["spacing_irregularity_boxes"] = []
        if self.character_misplacement_boxes is None:
            result["character_misplacement_boxes"] = []
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "HandwritingLabels":
        return cls(**data)


class HandwritingLabeler:
    def __init__(self):
        pass

    def classify_severity(self, score: float) -> str:
        if score < 0.1:
            return "none"
        elif score < 0.3:
            return "mild"
        elif score < 0.6:
            return "moderate"
        else:
            return "severe"

    def calculate_overall_risk(
        self,
        letter_reversals: str,
        spacing: str,
        misplacement: str,
        baseline: str,
        size_consistency: str,
    ) -> Tuple[str, float]:
        levels = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}

        weights = {
            "letter_reversals": 0.30,
            "spacing": 0.20,
            "misplacement": 0.20,
            "baseline": 0.15,
            "size_consistency": 0.15,
        }

        weighted_score = (
            weights["letter_reversals"] * levels[letter_reversals]
            + weights["spacing"] * levels[spacing]
            + weights["misplacement"] * levels[misplacement]
            + weights["baseline"] * levels[baseline]
            + weights["size_consistency"] * levels[size_consistency]
        ) / sum(weights.values())

        normalized_score = weighted_score / 3

        if normalized_score < 0.15:
            return "low", normalized_score
        elif normalized_score < 0.45:
            return "medium", normalized_score
        else:
            return "high", normalized_score

    def analyze_image_features(
        self, image_path: str, features: dict
    ) -> HandwritingLabels:
        student_hash = features.get("student_hash", "")
        task_type = features.get("task_type", "copying")

        reversal_count = features.get("reversal_count", 0)
        reversal_score = min(reversal_count / 10, 1.0)
        letter_reversals = self.classify_severity(reversal_score)

        spacing_irregularity = self.classify_severity(features.get("spacing_score", 0))

        misplacement_score = features.get("misplacement_score", 0)
        character_misplacement = self.classify_severity(misplacement_score)

        baseline_deviation = features.get("baseline_deviation", 0)
        baseline_adherence = self.classify_severity(baseline_deviation)

        size_variation = features.get("size_variation", 0)
        size_consistency = self.classify_severity(size_variation)

        overall_risk, overall_score = self.calculate_overall_risk(
            letter_reversals,
            spacing_irregularity,
            character_misplacement,
            baseline_adherence,
            size_consistency,
        )

        labels = HandwritingLabels(
            student_hash=student_hash,
            image_path=image_path,
            task_type=task_type,
            letter_reversals=letter_reversals,
            letter_reversals_notes=f"Count: {reversal_count}",
            spacing_irregularity=spacing_irregularity,
            spacing_notes=f"Score: {features.get('spacing_score', 0):.2f}",
            character_misplacement=character_misplacement,
            misplacement_notes=f"Score: {misplacement_score:.2f}",
            baseline_adherence=baseline_adherence,
            baseline_notes=f"Deviation: {baseline_deviation:.2f}",
            size_consistency=size_consistency,
            size_notes=f"Variation: {size_variation:.2f}",
            overall_risk=overall_risk,
            overall_score=overall_score,
        )

        return labels

    def export_labels(
        self, labels: List[HandwritingLabels], output_path: str, format: str = "json"
    ):
        if format == "json":
            with open(output_path, "w") as f:
                json.dump([l.to_dict() for l in labels], f, indent=2)
        elif format == "csv":
            import pandas as pd

            df = pd.DataFrame([l.to_dict() for l in labels])
            df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(labels)} labels to {output_path}")


def manual_annotation(
    image_path: str, student_hash: str, task_type: str
) -> HandwritingLabels:
    from datetime import datetime

    print(f"\nAnnotating: {image_path}")
    print(f"Student: {student_hash}, Task Type: {task_type}")

    print("\nLetter Reversals (b/d, p/q, etc.):")
    print("0 = None, 1 = Mild (1-2), 2 = Moderate (3-5), 3 = Severe (6+)")
    reversal_input = input("Enter level (0-3): ")

    print("\nSpacing Irregularity:")
    print("0 = None, 1 = Mild, 2 = Moderate, 3 = Severe")
    spacing_input = input("Enter level (0-3): ")

    print("\nCharacter Misplacement:")
    print("0 = None, 1 = Mild, 2 = Moderate, 3 = Severe")
    misplacement_input = input("Enter level (0-3): ")

    print("\nBaseline Adherence:")
    print("0 = None (good), 1 = Mild, 2 = Moderate, 3 = Severe")
    baseline_input = input("Enter level (0-3): ")

    print("\nSize Consistency:")
    print("0 = None (consistent), 1 = Mild, 2 = Moderate, 3 = Severe")
    size_input = input("Enter level (0-3): ")

    levels = ["none", "mild", "moderate", "severe"]

    def get_level(val):
        if val.isdigit() and 0 <= int(val) <= 3:
            return levels[int(val)]
        return "none"

    letter_reversals = get_level(reversal_input)
    spacing = get_level(spacing_input)
    misplacement = get_level(misplacement_input)
    baseline = get_level(baseline_input)
    size_consistency = get_level(size_input)

    labeler = HandwritingLabeler()
    overall_risk, overall_score = labeler.calculate_overall_risk(
        letter_reversals, spacing, misplacement, baseline, size_consistency
    )

    labels = HandwritingLabels(
        student_hash=student_hash,
        image_path=image_path,
        task_type=task_type,
        letter_reversals=letter_reversals,
        spacing_irregularity=spacing,
        character_misplacement=misplacement,
        baseline_adherence=baseline,
        size_consistency=size_consistency,
        overall_risk=overall_risk,
        overall_score=overall_score,
        annotator_id=input("Annotator ID: ") or "manual",
        annotation_date=datetime.now().isoformat(),
    )

    return labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Handwriting Data Labeling Tool")
    parser.add_argument("--image", help="Image file path")
    parser.add_argument("--student-hash", help="Student hash ID")
    parser.add_argument("--task-type", help="Task type (copying/free/dictation)")
    parser.add_argument("--batch", help="Directory with image files")
    parser.add_argument("--export", help="Output file path")

    args = parser.parse_args()

    if args.image and args.student_hash and args.task_type:
        label = manual_annotation(args.image, args.student_hash, args.task_type)
        print(
            f"\nLabel created: {label.overall_risk} risk (score: {label.overall_score:.2f})"
        )

        if args.export:
            labeler = HandwritingLabeler()
            labeler.export_labels([label], args.export)
    else:
        print("Please provide --image, --student-hash, and --task-type")
