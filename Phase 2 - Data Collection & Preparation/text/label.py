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
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class TextLabels:
    student_hash: str
    text_path: str
    prompt_id: str

    spelling_error_rate: float = 0.0
    spelling: str = "normal"
    spelling_notes: str = ""

    grammar_score: float = 0.0
    grammar: str = "normal"
    grammar_notes: str = ""

    sentence_complexity: str = "normal"
    complexity_score: float = 0.0
    complexity_notes: str = ""

    flesch_reading_ease: float = 0.0
    reading_ease: str = "normal"
    reading_ease_notes: str = ""

    vocabulary_diversity: float = 0.0
    vocabulary: str = "normal"
    vocabulary_notes: str = ""

    overall_risk: str = "low"
    overall_score: float = 0.0

    word_count: int = 0

    annotator_id: str = ""
    annotation_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TextLabels":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class TextLabeler:
    def __init__(self):
        pass

    def classify_spelling(self, error_rate: float) -> str:
        if error_rate < 0.05:
            return "normal"
        elif error_rate < 0.15:
            return "mild"
        elif error_rate < 0.30:
            return "moderate"
        else:
            return "severe"

    def classify_grammar(self, score: float) -> str:
        if score > 0.85:
            return "normal"
        elif score > 0.70:
            return "mild"
        elif score > 0.50:
            return "moderate"
        else:
            return "severe"

    def classify_complexity(self, avg_sentence_length: float, age: int = 8) -> str:
        expected_length = 10 + age * 1.5

        if avg_sentence_length <= expected_length * 1.2:
            return "normal"
        elif avg_sentence_length <= expected_length * 1.5:
            return "mild"
        elif avg_sentence_length <= expected_length * 2:
            return "moderate"
        else:
            return "severe"

    def classify_reading_ease(self, flesch_score: float) -> str:
        if flesch_score >= 60:
            return "normal"
        elif flesch_score >= 40:
            return "mild"
        elif flesch_score >= 20:
            return "moderate"
        else:
            return "severe"

    def classify_vocabulary(self, ttr: float) -> str:
        if ttr >= 0.5:
            return "normal"
        elif ttr >= 0.4:
            return "mild"
        elif ttr >= 0.3:
            return "moderate"
        else:
            return "severe"

    def calculate_overall_risk(
        self,
        spelling: str,
        grammar: str,
        complexity: str,
        reading_ease: str,
        vocabulary: str,
    ) -> Tuple[str, float]:
        levels = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}

        weights = {
            "spelling": 0.30,
            "grammar": 0.20,
            "complexity": 0.15,
            "reading_ease": 0.20,
            "vocabulary": 0.15,
        }

        weighted_score = (
            weights["spelling"] * levels[spelling]
            + weights["grammar"] * levels[grammar]
            + weights["complexity"] * levels[complexity]
            + weights["reading_ease"] * levels[reading_ease]
            + weights["vocabulary"] * levels[vocabulary]
        ) / sum(weights.values())

        normalized_score = weighted_score / 3

        if normalized_score < 0.15:
            return "low", normalized_score
        elif normalized_score < 0.45:
            return "medium", normalized_score
        else:
            return "high", normalized_score

    def analyze_text_features(self, text_path: str, features: dict) -> TextLabels:
        student_hash = features.get("student_hash", "")
        prompt_id = features.get("prompt_id", "")

        error_rate = features.get("spelling_error_rate", 0)
        spelling = self.classify_spelling(error_rate)

        grammar_score = features.get("grammar_score", 1.0)
        grammar = self.classify_grammar(grammar_score)

        avg_sentence_length = features.get("avg_sentence_length", 15)
        age = features.get("age", 8)
        complexity = self.classify_complexity(avg_sentence_length, age)
        complexity_score = avg_sentence_length / (10 + age * 1.5)

        flesch_score = features.get("flesch_reading_ease", 70)
        reading_ease = self.classify_reading_ease(flesch_score)

        ttr = features.get("type_token_ratio", 0.5)
        vocabulary = self.classify_vocabulary(ttr)

        overall_risk, overall_score = self.calculate_overall_risk(
            spelling, grammar, complexity, reading_ease, vocabulary
        )

        labels = TextLabels(
            student_hash=student_hash,
            text_path=text_path,
            prompt_id=prompt_id,
            spelling_error_rate=error_rate,
            spelling=spelling,
            spelling_notes=f"Error rate: {error_rate:.2%}",
            grammar_score=grammar_score,
            grammar=grammar,
            grammar_notes=f"Score: {grammar_score:.2f}",
            sentence_complexity=complexity,
            complexity_score=complexity_score,
            complexity_notes=f"Avg length: {avg_sentence_length:.1f}",
            flesch_reading_ease=flesch_score,
            reading_ease=reading_ease,
            reading_ease_notes=f"Flesch: {flesch_score:.1f}",
            vocabulary_diversity=ttr,
            vocabulary=vocabulary,
            vocabulary_notes=f"TTR: {ttr:.2f}",
            overall_risk=overall_risk,
            overall_score=overall_score,
            word_count=features.get("word_count", 0),
        )

        return labels

    def export_labels(
        self, labels: List[TextLabels], output_path: str, format: str = "json"
    ):
        if format == "json":
            with open(output_path, "w") as f:
                json.dump([l.to_dict() for l in labels], f, indent=2)
        elif format == "csv":
            import pandas as pd

            df = pd.DataFrame([l.to_dict() for l in labels])
            df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(labels)} labels to {output_path}")


def calculate_spelling_errors(text: str) -> float:
    common_misspellings = [
        "teh",
        "hte",
        "recieve",
        "recieved",
        "wierd",
        "seperate",
        "definately",
        "definitely",
        "occured",
        "occurred",
        "untill",
        "begining",
        "beginning",
        "beleive",
        "believe",
        "calender",
        "calendar",
        "commitment",
        "cemetery",
        "concensus",
        "consensus",
    ]

    words = text.lower().split()
    errors = sum(1 for w in words if w in common_misspellings)

    return errors / len(words) if words else 0


def calculate_flesch_reading_ease(text: str) -> float:
    words = text.split()
    if not words:
        return 100.0

    sentences = text.split(".")
    if not sentences:
        return 100.0

    syllable_count = sum(estimate_syllables(word) for word in words)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    flesch = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )
    return max(0, min(100, flesch))


def estimate_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    if word.endswith("e"):
        count -= 1

    return max(1, count)


def calculate_type_token_ratio(text: str) -> float:
    words = text.lower().split()
    if not words:
        return 0.0

    unique_words = set(words)
    return len(unique_words) / len(words)


def calculate_avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0

    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text Data Labeling Tool")
    parser.add_argument("--text", help="Text file path")
    parser.add_argument("--student-hash", help="Student hash ID")
    parser.add_argument("--prompt-id", help="Prompt ID")
    parser.add_argument("--batch", help="Directory with text files")
    parser.add_argument("--export", help="Output file path")

    args = parser.parse_args()

    labeler = TextLabeler()

    if args.text and args.student_hash and args.prompt_id:
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()

        features = {
            "student_hash": args.student_hash,
            "prompt_id": args.prompt_id,
            "word_count": len(text.split()),
            "spelling_error_rate": calculate_spelling_errors(text),
            "grammar_score": 0.85,
            "avg_sentence_length": calculate_avg_sentence_length(text),
            "flesch_reading_ease": calculate_flesch_reading_ease(text),
            "type_token_ratio": calculate_type_token_ratio(text),
        }

        labels = labeler.analyze_text_features(args.text, features)
        print(
            f"\nLabel created: {labels.overall_risk} risk (score: {labels.overall_score:.2f})"
        )
        print(f"Spelling: {labels.spelling}, Grammar: {labels.grammar}")

        if args.export:
            labeler.export_labels([labels], args.export)
    else:
        print("Please provide --text, --student-hash, and --prompt-id")


if __name__ == "__main__":
    main()
