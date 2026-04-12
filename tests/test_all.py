import os
import sys
import pytest
import numpy as np

# Fix path - dynamically find modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestAudioPreprocessor:
    def test_initialization(self):
        from Phase_2___Data_Collection___Preparation.speech.preprocess import (
            AudioPreprocessor,
        )

        preprocessor = AudioPreprocessor(sample_rate=48000)
        assert preprocessor.sample_rate == 48000

    def test_normalize_audio(self):
        from Phase_2___Data_Collection___Preparation.speech.preprocess import (
            AudioPreprocessor,
        )

        preprocessor = AudioPreprocessor()
        audio = np.random.randn(1000)
        normalized = preprocessor.normalize_audio(audio)
        assert -1.0 <= normalized.max() <= 1.0


class TestHandwritingPreprocessor:
    def test_initialization(self):
        from Phase_2___Data_Collection___Preparation.handwriting.preprocess import (
            HandwritingPreprocessor,
        )

        preprocessor = HandwritingPreprocessor()
        assert preprocessor is not None


class TestTextPreprocessor:
    def test_initialization(self):
        from Phase_2___Data_Collection___Preparation.text.preprocess import (
            TextPreprocessor,
        )

        preprocessor = TextPreprocessor(lowercase=True)
        assert preprocessor.lowercase is True


class TestSpeechLabeler:
    def test_initialization(self):
        from Phase_2___Data_Collection___Preparation.speech.label import SpeechLabeler

        labeler = SpeechLabeler()
        assert labeler is not None

    def test_classify_phonological(self):
        from Phase_2___Data_Collection___Preparation.speech.label import SpeechLabeler

        labeler = SpeechLabeler()
        assert labeler.classify_phonological(0.1) == "normal"


class TestHandwritingLabeler:
    def test_classify_severity(self):
        from Phase_2___Data_Collection___Preparation.handwriting.label import (
            HandwritingLabeler,
        )

        labeler = HandwritingLabeler()
        assert labeler.classify_severity(0.05) == "none"


class TestTextLabeler:
    def test_classify_spelling(self):
        from Phase_2___Data_Collection___Preparation.text.label import TextLabeler

        labeler = TextLabeler()
        assert labeler.classify_spelling(0.02) == "normal"


class TestAnonymizer:
    def test_initialization(self):
        from Phase_2___Data_Collection___Preparation.infrastructure.anonymizer import (
            Anonymizer,
        )

        anon = Anonymizer()
        assert anon.config.algorithm == "sha256"

    def test_hash_identifier(self):
        from Phase_2___Data_Collection___Preparation.infrastructure.anonymizer import (
            Anonymizer,
        )

        anon = Anonymizer()
        student_hash = anon.hash_identifier("student123", salt="test123")
        assert len(student_hash) == 16


class TestDatabaseSchema:
    def test_schema_exists(self):
        schema_path = os.path.join(
            project_root,
            "Phase 2 - Data Collection & Preparation",
            "infrastructure",
            "database_schema.sql",
        )
        assert os.path.exists(schema_path)
        with open(schema_path, "r") as f:
            content = f.read()
            assert "CREATE TABLE students" in content


class TestSpeechFeatureExtractor:
    def test_initialization(self):
        from Phase_3___Feature_Engineering.speech.feature_extractor import (
            SpeechFeatureExtractor,
        )

        extractor = SpeechFeatureExtractor(n_mfcc=13)
        assert extractor.n_mfcc == 13


class TestHandwritingFeatureExtractor:
    def test_initialization(self):
        from Phase_3___Feature_Engineering.handwriting.feature_extractor import (
            HandwritingFeatureExtractor,
        )

        extractor = HandwritingFeatureExtractor()
        assert extractor is not None


class TestTextFeatureExtractor:
    def test_initialization(self):
        from Phase_3___Feature_Engineering.text.feature_extractor import (
            TextFeatureExtractor,
        )

        extractor = TextFeatureExtractor(use_bert=False)
        assert extractor.use_bert is False

    def test_extract_spelling_features(self):
        from Phase_3___Feature_Engineering.text.feature_extractor import (
            TextFeatureExtractor,
        )

        extractor = TextFeatureExtractor(use_bert=False)
        text = "The cat sat on the mat. Teh dog ran."
        features = extractor.extract_spelling_features(text)
        assert "word_count" in features
        assert "spelling_error_count" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
