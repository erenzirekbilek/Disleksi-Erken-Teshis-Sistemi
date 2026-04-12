import os
import glob
import logging
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert
        self.bert_model = None
        self.bert_tokenizer = None

    def load_text(self, text_path: str) -> str:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(text_path, "r", encoding=encoding) as f:
                    text = f.read()
                return text
            except (UnicodeDecodeError, FileNotFoundError):
                continue

        raise ValueError(f"Could not decode {text_path} with any encoding")

    def extract_spelling_features(self, text: str) -> Dict:
        words = text.split()

        if not words:
            return {
                "word_count": 0,
                "spelling_error_rate": 0,
                "spelling_error_count": 0,
            }

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
            " independant",
            "independant",
            "intresting",
            "interesting",
            "judgement",
            "judgement",
            "knowlege",
            "knowledge",
            "neccessary",
            "necessary",
            "noticable",
            "noticeable",
            "occassion",
            "occasion",
            "persistant",
            "persistent",
            "posession",
            "possession",
            "prefered",
            "preferred",
            "privelege",
            "privilege",
            "priviledge",
            "recomend",
            "recommend",
            "reffered",
            "referred",
            "relevent",
            "relevant",
            "tommorow",
            "tomorrow",
            "tommorrow",
            "truely",
            "truly",
            "usefull",
            "useful",
            "writting",
            "writing",
        ]

        words_lower = [w.lower().strip(".,!?;:") for w in words]
        errors = sum(1 for w in words_lower if w in common_misspellings)

        simple_errors = 0
        for word in words_lower:
            if len(word) > 3:
                if word.endswith("te") or word.endswith("ed") or word.endswith("ing"):
                    if word[:-2] not in words_lower:
                        simple_errors += 1

        features = {
            "word_count": len(words),
            "unique_word_count": len(set(words_lower)),
            "spelling_error_count": errors,
            "spelling_error_rate": errors / len(words) if words else 0,
            "simple_error_count": simple_errors,
            "simple_error_rate": simple_errors / len(words) if words else 0,
        }

        logger.info(f"Spelling: {errors} errors in {len(words)} words")
        return features

    def extract_grammar_features(self, text: str) -> Dict:
        sentences = self._split_sentences(text)

        if not sentences:
            return self._empty_grammar_features()

        subject_verb_patterns = [
            r"\b(I|we|they|he|she|it)\s+\w+ing\b",
            r"\b(I|we|they|he|she|it)\s+\w+ed\b",
        ]

        issues = 0
        for sentence in sentences:
            for pattern in subject_verb_patterns:
                if re.search(pattern, sentence):
                    issues += 1

        word_count = len(text.split())
        sentence_count = len(sentences)

        article_patterns = [
            r"\ba\s+[aeiou]",
            r"\ban\s+[bcdfghjklmnpqrstvwxyz]",
        ]

        article_errors = 0
        for sentence in sentences:
            for pattern in article_patterns:
                if re.search(pattern, sentence.lower()):
                    article_errors += 1

        features = {
            "sentence_count": sentence_count,
            "subject_verb_issues": issues,
            "subject_verb_issue_rate": issues / sentence_count
            if sentence_count > 0
            else 0,
            "article_errors": article_errors,
            "article_error_rate": article_errors / sentence_count
            if sentence_count > 0
            else 0,
            "grammar_score": 1
            - min((issues + article_errors) / max(sentence_count, 1), 1),
        }

        logger.info(
            f"Grammar: {issues + article_errors} issues in {sentence_count} sentences"
        )
        return features

    def _empty_grammar_features(self) -> Dict:
        return {
            "sentence_count": 0,
            "subject_verb_issues": 0,
            "subject_verb_issue_rate": 0,
            "article_errors": 0,
            "article_error_rate": 0,
            "grammar_score": 1.0,
        }

    def extract_readability_features(self, text: str) -> Dict:
        words = text.split()
        sentences = self._split_sentences(text)

        if not words or not sentences:
            return self._empty_readability_features()

        syllables = sum(self._estimate_syllables(word) for word in words)

        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words) if words else 0

        flesch = (
            206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        )
        flesch = max(0, min(100, flesch))

        ari = 4.71 * avg_syllables_per_word + 0.5 * avg_words_per_sentence - 21.43
        ari = max(0, min(14, ari))

        coleman_liau = (
            0.0588 * (sum(len(w) for w in words) / len(words) * 100)
            - 0.296 * (len(sentences) / len(words) * 100)
            - 15.8
        )
        coleman_liau = max(0, min(14, coleman_liau))

        features = {
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_syllables_per_word": avg_syllables_per_word,
            "total_syllables": syllables,
            "flesch_reading_ease": flesch,
            "flesch_kincaid_grade": 0.39 * avg_words_per_sentence
            + 11.8 * avg_syllables_per_word
            - 15.59,
            "automated_readability_index": ari,
            "coleman_liau_index": coleman_liau,
            "reading_grade_level": max(flesch, ari, coleman_liau) / 2,
        }

        logger.info(f"Readability: Flesch={flesch:.1f}, ARI={ari:.1f}")
        return features

    def _empty_readability_features(self) -> Dict:
        return {
            "avg_words_per_sentence": 0,
            "avg_syllables_per_word": 0,
            "total_syllables": 0,
            "flesch_reading_ease": 100,
            "flesch_kincaid_grade": 0,
            "automated_readability_index": 0,
            "coleman_liau_index": 0,
            "reading_grade_level": 0,
        }

    def _estimate_syllables(self, word: str) -> int:
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

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def extract_vocabulary_features(self, text: str) -> Dict:
        words = text.lower().split()

        if not words:
            return self._empty_vocabulary_features()

        unique_words = set(words)

        ttr = len(unique_words) / len(words) if words else 0

        hapax_legomena = sum(1 for w in unique_words if words.count(w) == 1)
        hapax_ratio = hapax_legomena / len(unique_words) if unique_words else 0

        word_lengths = [len(w) for w in words]

        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / len(words) if words else 0

        features = {
            "type_token_ratio": ttr,
            "hapax_legomena_count": hapax_legomena,
            "hapax_ratio": hapax_ratio,
            "vocabulary_richness_score": (ttr + hapax_ratio + long_word_ratio) / 3,
            "avg_word_length": np.mean(word_lengths) if word_lengths else 0,
            "word_length_std": np.std(word_lengths) if word_lengths else 0,
            "long_word_count": long_words,
            "long_word_ratio": long_word_ratio,
        }

        logger.info(
            f"Vocabulary: TTR={ttr:.3f}, Richness={features['vocabulary_richness_score']:.3f}"
        )
        return features

    def _empty_vocabulary_features(self) -> Dict:
        return {
            "type_token_ratio": 0,
            "hapax_legomena_count": 0,
            "hapax_ratio": 0,
            "vocabulary_richness_score": 0,
            "avg_word_length": 0,
            "word_length_std": 0,
            "long_word_count": 0,
            "long_word_ratio": 0,
        }

    def extract_complexity_features(self, text: str) -> Dict:
        sentences = self._split_sentences(text)

        if not sentences:
            return self._empty_complexity_features()

        sentence_lengths = [len(s.split()) for s in sentences]

        complex_words = sum(
            1 for s in sentences for w in s.split() if self._estimate_syllables(w) >= 3
        )
        total_words = sum(len(s.split()) for s in sentences)

        features = {
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_min": min(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_max": max(sentence_lengths) if sentence_lengths else 0,
            "sentence_length_range": max(sentence_lengths) - min(sentence_lengths)
            if sentence_lengths
            else 0,
            "complex_word_count": complex_words,
            "complex_word_ratio": complex_words / total_words if total_words > 0 else 0,
            "sentence_complexity_score": (
                np.mean(sentence_lengths) + complex_words / len(sentences)
            )
            / 2
            if sentences
            else 0,
        }

        logger.info(
            f"Complexity: Avg sentence length={features['avg_sentence_length']:.1f}"
        )
        return features

    def _empty_complexity_features(self) -> Dict:
        return {
            "avg_sentence_length": 0,
            "sentence_length_std": 0,
            "sentence_length_min": 0,
            "sentence_length_max": 0,
            "sentence_length_range": 0,
            "complex_word_count": 0,
            "complex_word_ratio": 0,
            "sentence_complexity_score": 0,
        }

    def extract_pos_features(self, text: str) -> Dict:
        try:
            import spacy

            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                nlp = spacy.load("en_core_web_md")
        except:
            return self._empty_pos_features()

        doc = nlp(text)

        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        total_tokens = len(doc)

        features = {
            "noun_count": pos_counts.get("NOUN", 0),
            "noun_ratio": pos_counts.get("NOUN", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "verb_count": pos_counts.get("VERB", 0),
            "verb_ratio": pos_counts.get("VERB", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "adj_count": pos_counts.get("ADJ", 0),
            "adj_ratio": pos_counts.get("ADJ", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "adv_count": pos_counts.get("ADV", 0),
            "adv_ratio": pos_counts.get("ADV", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "pronoun_count": pos_counts.get("PRON", 0),
            "pronoun_ratio": pos_counts.get("PRON", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "preposition_count": pos_counts.get("ADP", 0),
            "preposition_ratio": pos_counts.get("ADP", 0) / total_tokens
            if total_tokens > 0
            else 0,
            "conjunction_count": pos_counts.get("CONJ", 0),
            "conjunction_ratio": pos_counts.get("CONJ", 0) / total_tokens
            if total_tokens > 0
            else 0,
        }

        logger.info(f"POS: {total_tokens} tokens analyzed")
        return features

    def _empty_pos_features(self) -> Dict:
        return {
            "noun_count": 0,
            "noun_ratio": 0,
            "verb_count": 0,
            "verb_ratio": 0,
            "adj_count": 0,
            "adj_ratio": 0,
            "adv_count": 0,
            "adv_ratio": 0,
            "pronoun_count": 0,
            "pronoun_ratio": 0,
            "preposition_count": 0,
            "preposition_ratio": 0,
            "conjunction_count": 0,
            "conjunction_ratio": 0,
        }

    def extract_bert_embeddings(self, text: str) -> Dict:
        if not self.use_bert:
            return {"embedding_available": False}

        try:
            if self.bert_model is None or self.bert_tokenizer is None:
                from transformers import AutoModel, AutoTokenizer

                self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
                self.bert_model.eval()

            inputs = self.bert_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            mean_pooling = outputs.last_hidden_state.mean(dim=1).numpy()[0]

            features = {
                "embedding_available": True,
                "bert_cls_mean": float(np.mean(cls_embedding)),
                "bert_cls_std": float(np.std(cls_embedding)),
                "bert_pooled_mean": float(np.mean(mean_pooling)),
                "bert_pooled_std": float(np.std(mean_pooling)),
            }

            logger.info("Extracted BERT embeddings")
            return features

        except Exception as e:
            logger.warning(f"BERT embedding extraction failed: {e}")
            return {"embedding_available": False, "error": str(e)}

    def extract_all(self, text_path: str) -> Dict:
        if text_path.endswith(".txt"):
            text = self.load_text(text_path)
        else:
            text = text_path

        features = {}
        features["text_path"] = text_path
        features["text_length"] = len(text)

        features.update(self.extract_spelling_features(text))
        features.update(self.extract_grammar_features(text))
        features.update(self.extract_readability_features(text))
        features.update(self.extract_vocabulary_features(text))
        features.update(self.extract_complexity_features(text))
        features.update(self.extract_pos_features(text))
        features.update(self.extract_bert_embeddings(text))

        logger.info(f"Total features extracted: {len(features)}")
        return features

    def process_file(self, text_path: str, output_path: Optional[str] = None) -> Dict:
        features = self.extract_all(text_path)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved features to: {output_path}")

        return features

    def process_directory(
        self, input_dir: str, output_dir: str, extensions: List[str] = [".txt"]
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for ext in extensions:
            text_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
            text_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        text_files = list(set(text_files))

        for text_path in text_files:
            try:
                filename = Path(text_path).stem
                output_path = os.path.join(output_dir, f"{filename}_features.json")

                features = self.process_file(text_path, output_path)

                result = {
                    "filename": filename,
                    "status": "success",
                    "output_path": output_path,
                    "word_count": features.get("word_count", 0),
                    "spelling_error_rate": features.get("spelling_error_rate", 0),
                    "flesch_reading_ease": features.get("flesch_reading_ease", 0),
                }
                results.append(result)
                logger.info(f"Processed: {filename}")

            except Exception as e:
                logger.error(f"Failed to process {text_path}: {e}")
                results.append(
                    {
                        "filename": Path(text_path).stem,
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

    parser = argparse.ArgumentParser(description="Text Feature Extraction")
    parser.add_argument("input", help="Input text file, directory, or raw text")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "--no-bert", action="store_true", help="Disable BERT embeddings"
    )
    parser.add_argument("--matrix", action="store_true", help="Create feature matrix")

    args = parser.parse_args()

    extractor = TextFeatureExtractor(use_bert=not args.no_bert)

    if os.path.isdir(args.input):
        df = extractor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(df)} files")
        print(df["status"].value_counts())

        if args.matrix and args.output:
            create_feature_matrix(args.output)
    else:
        result = extractor.process_file(args.input, args.output)
        print(f"Extracted {len(result)} feature groups")
        print(f"Word count: {result.get('word_count', 0)}")


if __name__ == "__main__":
    main()
