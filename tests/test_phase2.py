import pytest
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "Phase 2 - Data Collection & Preparation"
        / "speech"
    ),
)
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "Phase 2 - Data Collection & Preparation"
        / "handwriting"
    ),
)
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "Phase 2 - Data Collection & Preparation"
        / "text"
    ),
)
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "Phase 2 - Data Collection & Preparation"
        / "infrastructure"
    ),
)


class TestAudioPreprocessor:
    def test_initialization(self):
        from speech.preprocess import AudioPreprocessor

        preprocessor = AudioPreprocessor(sample_rate=48000)
        assert preprocessor.sample_rate == 48000
        assert preprocessor.target_lufs == -20.0
        assert preprocessor.top_db == 20

    def test_normalize_audio(self):
        from speech.preprocess import AudioPreprocessor

        preprocessor = AudioPreprocessor()
        audio = np.random.randn(1000)
        normalized = preprocessor.normalize_audio(audio)
        assert -1.0 <= normalized.max() <= 1.0
        assert -1.0 <= normalized.min() <= 1.0

    def test_remove_silence(self):
        from speech.preprocess import AudioPreprocessor

        preprocessor = AudioPreprocessor(top_db=20)
        audio = np.concatenate([np.zeros(100), np.random.randn(500), np.zeros(100)])
        trimmed = preprocessor.remove_silence(audio)
        assert len(trimmed) < len(audio)

    def test_apply_filters(self):
        from speech.preprocess import AudioPreprocessor

        preprocessor = AudioPreprocessor()
        audio = np.random.randn(1000)
        filtered = preprocessor.apply_filters(audio, 48000)
        assert len(filtered) == len(audio)


class TestHandwritingPreprocessor:
    def test_initialization(self):
        from handwriting.preprocess import HandwritingPreprocessor

        preprocessor = HandwritingPreprocessor(target_dpi=300)
        assert preprocessor.target_dpi == 300
        assert preprocessor.max_dimension == 2048

    def test_resize_image(self):
        from handwriting.preprocess import HandwritingPreprocessor
        import numpy as np

        preprocessor = HandwritingPreprocessor(max_dimension=512)
        img = np.random.randint(0, 255, (2000, 3000), dtype=np.uint8)
        resized = preprocessor.resize_image(img)
        assert max(resized.shape) == 512

    def test_binarize_image(self):
        from handwriting.preprocess import HandwritingPreprocessor
        import numpy as np

        preprocessor = HandwritingPreprocessor()
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = preprocessor.binarize_image(img)
        assert binary.max() == 255
        assert binary.min() == 0


class TestTextPreprocessor:
    def test_initialization(self):
        from text.preprocess import TextPreprocessor

        preprocessor = TextPreprocessor(lowercase=True)
        assert preprocessor.lowercase is True
        assert preprocessor.max_length == 10000

    def test_normalize_unicode(self):
        from text.preprocess import TextPreprocessor

        preprocessor = TextPreprocessor()
        text = "café"
        normalized = preprocessor.normalize_unicode(text)
        assert normalized is not None

    def test_normalize_whitespace(self):
        from text.preprocess import TextPreprocessor

        preprocessor = TextPreprocessor()
        text = "Hello   world\n\n\n"
        normalized = preprocessor.normalize_whitespace(text)
        assert "\n\n\n" not in normalized


class TestSpeechLabeler:
    def test_initialization(self):
        from speech.label import SpeechLabeler

        labeler = SpeechLabeler()
        assert labeler.phonological_thresholds is not None
        assert labeler.fluency_wpm_thresholds is not None

    def test_classify_phonological(self):
        from speech.label import SpeechLabeler

        labeler = SpeechLabeler()
        assert labeler.classify_phonological(0.1) == "normal"
        assert labeler.classify_phonological(0.25) == "mild"
        assert labeler.classify_phonological(0.5) == "moderate"
        assert labeler.classify_phonological(0.8) == "severe"

    def test_calculate_overall_risk(self):
        from speech.label import SpeechLabeler

        labeler = SpeechLabeler()
        risk, score = labeler.calculate_overall_risk("normal", "normal", "normal")
        assert risk == "low"
        assert score < 0.15


class TestHandwritingLabeler:
    def test_initialization(self):
        from handwriting.label import HandwritingLabeler

        labeler = HandwritingLabeler()
        assert labeler is not None

    def test_classify_severity(self):
        from handwriting.label import HandwritingLabeler

        labeler = HandwritingLabeler()
        assert labeler.classify_severity(0.05) == "none"
        assert labeler.classify_severity(0.2) == "mild"
        assert labeler.classify_severity(0.5) == "moderate"
        assert labeler.classify_severity(0.8) == "severe"


class TestTextLabeler:
    def test_initialization(self):
        from text.label import TextLabeler

        labeler = TextLabeler()
        assert labeler is not None

    def test_classify_spelling(self):
        from text.label import TextLabeler

        labeler = TextLabeler()
        assert labeler.classify_spelling(0.02) == "normal"
        assert labeler.classify_spelling(0.1) == "mild"
        assert labeler.classify_spelling(0.25) == "moderate"

    def test_classify_reading_ease(self):
        from text.label import TextLabeler

        labeler = TextLabeler()
        assert labeler.classify_reading_ease(70) == "normal"
        assert labeler.classify_reading_ease(50) == "mild"
        assert labeler.classify_reading_ease(30) == "moderate"


class TestAnonymizer:
    def test_initialization(self):
        from infrastructure.anonymizer import Anonymizer

        anon = Anonymizer()
        assert anon.config.algorithm == "sha256"

    def test_hash_identifier(self):
        from infrastructure.anonymizer import Anonymizer

        anon = Anonymizer()
        student_hash = anon.hash_identifier("student123", salt="test123")
        assert len(student_hash) == 16

    def test_remove_pii_from_text(self):
        from infrastructure.anonymizer import Anonymizer

        anon = Anonymizer()
        text = "John Smith has email john@test.com and SSN 123-45-6789"
        anonymized = anon.remove_pii_from_text(text)
        assert "[NAME]" in anonymized
        assert "[EMAIL]" in anonymized
        assert "[SSN]" in anonymized

    def test_anonymize_metadata(self):
        from infrastructure.anonymizer import Anonymizer

        anon = Anonymizer()
        metadata = {"name": "John", "age": 10, "grade": "3", "email": "test@test.com"}
        anonymized = anon.anonymize_metadata(metadata)
        assert "name" not in anonymized or anonymized.get("name") is None
        assert anonymized.get("age") == 10
        assert anonymized.get("grade") == "3"


class TestDatabaseSchema:
    def test_schema_exists(self):
        schema_path = (
            Path(__file__).parent.parent
            / "Phase 2 - Data Collection & Preparation"
            / "infrastructure"
            / "database_schema.sql"
        )
        assert schema_path.exists()
        with open(schema_path, "r") as f:
            content = f.read()
            assert "CREATE TABLE students" in content
            assert "CREATE TABLE speech_samples" in content
            assert "CREATE TABLE handwriting_samples" in content
            assert "CREATE TABLE text_samples" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
