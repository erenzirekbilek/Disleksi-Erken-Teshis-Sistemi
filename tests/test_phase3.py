import pytest
import os
import sys
import numpy as np
from pathlib import Path
import json
import tempfile

sys.path.insert(
    0, str(Path(__file__).parent.parent / "Phase 3 - Feature Engineering" / "speech")
)
sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "Phase 3 - Feature Engineering" / "handwriting"),
)
sys.path.insert(
    0, str(Path(__file__).parent.parent / "Phase 3 - Feature Engineering" / "text")
)


class TestSpeechFeatureExtractor:
    def test_initialization(self):
        from speech.feature_extractor import SpeechFeatureExtractor

        extractor = SpeechFeatureExtractor(sample_rate=48000, n_mfcc=13)
        assert extractor.sample_rate == 48000
        assert extractor.n_mfcc == 13

    def test_extract_mfcc_shape(self):
        from speech.feature_extractor import SpeechFeatureExtractor
        import librosa

        extractor = SpeechFeatureExtractor()
        y = np.random.randn(48000)
        features = extractor.extract_mfcc(y, 48000)
        assert "mfcc_mean" in features
        assert len(features["mfcc_mean"]) == 39

    def test_extract_energy(self):
        from speech.feature_extractor import SpeechFeatureExtractor

        extractor = SpeechFeatureExtractor()
        y = np.random.randn(48000)
        features = extractor.extract_energy(y, 48000)
        assert "energy_mean" in features
        assert "rms_mean" in features
        assert "zcr_mean" in features

    def test_extract_pitch(self):
        from speech.feature_extractor import SpeechFeatureExtractor

        extractor = SpeechFeatureExtractor()
        y = np.random.randn(48000)
        features = extractor.extract_pitch(y, 48000)
        assert "pitch_mean" in features
        assert "voiced_ratio" in features

    def test_extract_spectral(self):
        from speech.feature_extractor import SpeechFeatureExtractor

        extractor = SpeechFeatureExtractor()
        y = np.random.randn(48000)
        features = extractor.extract_spectral(y, 48000)
        assert "spectral_centroid_mean" in features
        assert "spectral_rolloff_mean" in features


class TestHandwritingFeatureExtractor:
    def test_initialization(self):
        from handwriting.feature_extractor import HandwritingFeatureExtractor

        extractor = HandwritingFeatureExtractor(min_component_area=50)
        assert extractor.min_component_area == 50

    def test_analyze_sizes(self):
        from handwriting.feature_extractor import HandwritingFeatureExtractor

        extractor = HandwritingFeatureExtractor()
        components = [
            {"x": 0, "y": 0, "width": 20, "height": 30, "area": 600},
            {"x": 30, "y": 0, "width": 25, "height": 35, "area": 875},
            {"x": 65, "y": 0, "width": 18, "height": 28, "area": 504},
        ]
        features = extractor.analyze_sizes(components)
        assert "size_mean" in features
        assert "size_std" in features
        assert "size_cv" in features
        assert features["character_count"] == 3

    def test_analyze_spacing(self):
        from handwriting.feature_extractor import HandwritingFeatureExtractor

        extractor = HandwritingFeatureExtractor()
        components = [
            {"x": 0, "y": 0, "width": 20, "height": 30, "area": 600},
            {"x": 40, "y": 0, "width": 25, "height": 35, "area": 875},
            {"x": 80, "y": 0, "width": 18, "height": 28, "area": 504},
        ]
        features = extractor.analyze_spacing(components, 200)
        assert "spacing_mean" in features
        assert "spacing_std" in features

    def test_detect_reversals(self):
        from handwriting.feature_extractor import HandwritingFeatureExtractor
        import numpy as np

        extractor = HandwritingFeatureExtractor()
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        components = [
            {"x": 10, "y": 10, "width": 20, "height": 30, "area": 600},
            {"x": 40, "y": 10, "width": 20, "height": 30, "area": 600},
        ]
        features = extractor.detect_reversals(img, components)
        assert "reversal_count" in features
        assert "reversal_b_count" in features


class TestTextFeatureExtractor:
    def test_initialization(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        assert extractor.use_bert is False

    def test_extract_spelling_features(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        text = "The cat sat on the mat. Teh dog ran."
        features = extractor.extract_spelling_features(text)
        assert "word_count" in features
        assert "spelling_error_count" in features
        assert "spelling_error_rate" in features

    def test_extract_grammar_features(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        text = "The cat sat on the mat. The dog ran."
        features = extractor.extract_grammar_features(text)
        assert "sentence_count" in features
        assert "grammar_score" in features

    def test_extract_readability_features(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        text = "This is a simple sentence. Here is another one."
        features = extractor.extract_readability_features(text)
        assert "flesch_reading_ease" in features
        assert "avg_words_per_sentence" in features

    def test_extract_vocabulary_features(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        text = "The cat and the dog ran fast."
        features = extractor.extract_vocabulary_features(text)
        assert "type_token_ratio" in features
        assert "vocabulary_richness_score" in features

    def test_extract_complexity_features(self):
        from text.feature_extractor import TextFeatureExtractor

        extractor = TextFeatureExtractor(use_bert=False)
        text = "Short. This is a longer sentence with more words."
        features = extractor.extract_complexity_features(text)
        assert "avg_sentence_length" in features
        assert "sentence_length_std" in features


class TestFeatureFlattening:
    def test_speech_flatten(self):
        from speech.feature_extractor import flatten_features

        features = {
            "audio_path": "test.wav",
            "mfcc_mean": [1.0, 2.0, 3.0],
            "pitch_mean": 150.0,
            "energy_mean": 0.5,
        }
        flat = flatten_features(features)
        assert "audio_path" in flat
        assert "pitch_mean" in flat
        assert "energy_mean" in flat

    def test_handwriting_flatten(self):
        from handwriting.feature_extractor import flatten_features

        features = {"image_path": "test.png", "character_count": 10, "size_mean": 500.0}
        flat = flatten_features(features)
        assert "image_path" in flat
        assert "character_count" in flat


class TestFeatureImportance:
    def test_feature_importance_analyzer(self):
        from feature_importance import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        assert analyzer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
