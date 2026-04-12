import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhonologicalSeverity(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class FluencySeverity(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class PronunciationSeverity(Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class SpeechLabels:
    student_hash: str
    audio_path: str
    passage_id: str

    phonological: str
    phonological_notes: str = ""

    fluency_wpm: float = 0.0
    fluency_repetitions: int = 0
    fluency: str = "normal"
    fluency_notes: str = ""

    pronunciation: str = "normal"
    pronunciation_notes: str = ""

    overall_risk: str = "low"
    overall_score: float = 0.0

    annotator_id: str = ""
    annotation_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeechLabels":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class SpeechLabeler:
    def __init__(self):
        self.phonological_thresholds = {
            "normal": (0, 0.15),
            "mild": (0.15, 0.35),
            "moderate": (0.35, 0.60),
            "severe": (0.60, 1.0),
        }

        self.fluency_wpm_thresholds = {
            "normal": (100, 200),
            "mild": (80, 100),
            "moderate": (60, 80),
            "severe": (0, 60),
        }

        self.fluency_rep_thresholds = {
            "normal": (0, 2),
            "mild": (2, 5),
            "moderate": (5, 10),
            "severe": (10, float("inf")),
        }

        self.pronunciation_thresholds = {
            "normal": (0, 0.10),
            "mild": (0.10, 0.25),
            "moderate": (0.25, 0.50),
            "severe": (0.50, 1.0),
        }

    def classify_phonological(self, score: float) -> str:
        for level, (min_val, max_val) in self.phonological_thresholds.items():
            if min_val <= score < max_val:
                return level
        return "severe"

    def classify_fluency_wpm(self, wpm: float) -> str:
        for level, (min_val, max_val) in self.fluency_wpm_thresholds.items():
            if min_val <= wpm < max_val:
                return level
        return "severe" if wpm < 60 else "normal"

    def classify_fluency_repetitions(self, reps: int) -> str:
        for level, (min_val, max_val) in self.fluency_rep_thresholds.items():
            if min_val <= reps < max_val:
                return level
        return "severe"

    def classify_pronunciation(self, score: float) -> str:
        for level, (min_val, max_val) in self.pronunciation_thresholds.items():
            if min_val <= score < max_val:
                return level
        return "severe"

    def calculate_fluency(self, wpm: float, repetitions: int) -> str:
        wpm_level = self.classify_fluency_wpm(wpm)
        rep_level = self.classify_fluency_repetitions(repetitions)

        levels = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
        max_level = max(levels[wpm_level], levels[rep_level])

        for level, score in levels.items():
            if score == max_level:
                return level
        return "normal"

    def calculate_overall_risk(
        self, phonological: str, fluency: str, pronunciation: str
    ) -> Tuple[str, float]:
        levels = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}

        weights = {"phonological": 0.4, "fluency": 0.35, "pronunciation": 0.25}

        weighted_score = (
            weights["phonological"] * levels[phonological]
            + weights["fluency"] * levels[fluency]
            + weights["pronunciation"] * levels[pronunciation]
        ) / 3

        normalized_score = weighted_score / 3

        if normalized_score < 0.15:
            return "low", normalized_score
        elif normalized_score < 0.40:
            return "medium", normalized_score
        else:
            return "high", normalized_score

    def analyze_audio_features(self, audio_path: str, features: dict) -> SpeechLabels:
        student_hash = features.get("student_hash", "")
        passage_id = features.get("passage_id", "")

        pause_frequency = features.get("pause_frequency", 0)
        sound_omissions = features.get("sound_omissions", 0)
        phonological_score = pause_frequency * 0.5 + sound_omissions * 0.5
        phonological_score = min(phonological_score, 1.0)
        phonological = self.classify_phonological(phonological_score)

        wpm = features.get("wpm", 150)
        repetitions = features.get("repetitions", 0)
        fluency = self.calculate_fluency(wpm, repetitions)

        pronunciation_errors = features.get("pronunciation_errors", 0)
        pronunciation_score = min(pronunciation_errors / 10, 1.0)
        pronunciation = self.classify_pronunciation(pronunciation_score)

        overall_risk, overall_score = self.calculate_overall_risk(
            phonological, fluency, pronunciation
        )

        labels = SpeechLabels(
            student_hash=student_hash,
            audio_path=audio_path,
            passage_id=passage_id,
            phonological=phonological,
            phonological_notes=f"Score: {phonological_score:.2f}",
            fluency_wpm=wpm,
            fluency_repetitions=repetitions,
            fluency=fluency,
            fluency_notes=f"WPM: {wpm}, Repetitions: {repetitions}",
            pronunciation=pronunciation,
            pronunciation_notes=f"Error rate: {pronunciation_score:.2f}",
            overall_risk=overall_risk,
            overall_score=overall_score,
        )

        return labels

    def export_labels(
        self, labels: List[SpeechLabels], output_path: str, format: str = "json"
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
    audio_path: str, student_hash: str, passage_id: str
) -> SpeechLabels:
    from datetime import datetime

    print(f"\nAnnotating: {audio_path}")
    print(f"Student: {student_hash}, Passage: {passage_id}")

    print("\nPhonological Assessment:")
    print("0 = Normal (no significant issues)")
    print("1 = Mild (occasional pauses, minor omissions)")
    print("2 = Moderate (frequent pauses, some sound errors)")
    print("3 = Severe (significant phonological difficulties)")
    phonological_input = input("Enter level (0-3): ")

    print("\nFluency Assessment:")
    print("Enter reading speed (words per minute)")
    wpm_input = input("WPM: ")
    print("Enter number of repetitions/hesitations")
    reps_input = input("Repetitions: ")

    print("\nPronunciation Assessment:")
    print("0 = Normal")
    print("1 = Mild (occasional errors)")
    print("2 = Moderate (frequent errors)")
    print("3 = Severe (consistent errors)")
    pronunciation_input = input("Enter level (0-3): ")

    levels = ["normal", "mild", "moderate", "severe"]
    phonological_level = levels[
        int(phonological_input) if phonological_input.isdigit() else 0
    ]
    pronunciation_level = levels[
        int(pronunciation_input) if pronunciation_input.isdigit() else 0
    ]

    labeler = SpeechLabeler()
    fluency_level = labeler.calculate_fluency(
        float(wpm_input) if wpm_input.isdigit() else 150,
        int(reps_input) if reps_input.isdigit() else 0,
    )

    overall_risk, overall_score = labeler.calculate_overall_risk(
        phonological_level, fluency_level, pronunciation_level
    )

    labels = SpeechLabels(
        student_hash=student_hash,
        audio_path=audio_path,
        passage_id=passage_id,
        phonological=phonological_level,
        fluency_wpm=float(wpm_input) if wpm_input.isdigit() else 150,
        fluency_repetitions=int(reps_input) if reps_input.isdigit() else 0,
        fluency=fluency_level,
        pronunciation=pronunciation_level,
        overall_risk=overall_risk,
        overall_score=overall_score,
        annotator_id=input("Annotator ID: ") or "manual",
        annotation_date=datetime.now().isoformat(),
    )

    return labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech Data Labeling Tool")
    parser.add_argument("--audio", help="Audio file path")
    parser.add_argument("--student-hash", help="Student hash ID")
    parser.add_argument("--passage-id", help="Reading passage ID")
    parser.add_argument("--batch", help="Directory with audio files")
    parser.add_argument("--export", help="Output file path")

    args = parser.parse_args()

    if args.audio and args.student_hash and args.passage_id:
        label = manual_annotation(args.audio, args.student_hash, args.passage_id)
        print(
            f"\nLabel created: {label.overall_risk} risk (score: {label.overall_score:.2f})"
        )

        if args.export:
            labeler = SpeechLabeler()
            labeler.export_labels([label], args.export)
    else:
        print("Please provide --audio, --student-hash, and --passage-id")
