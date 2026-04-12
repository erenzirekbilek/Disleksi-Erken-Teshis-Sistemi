import re
import os
import logging
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict
import json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        max_length: int = 10000,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.max_length = max_length

    def load_text(self, text_path: str) -> str:
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(text_path, "r", encoding=encoding) as f:
                    text = f.read()
                logger.info(f"Loaded text from {text_path} with {encoding}")
                return text
            except (UnicodeDecodeError, FileNotFoundError):
                continue

        raise ValueError(f"Could not decode {text_path} with any encoding")

    def normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = unicodedata.normalize("NFC", text)
        logger.info("Normalized Unicode")
        return text

    def normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()
        logger.info("Normalized whitespace")
        return text

    def remove_special_characters(self, text: str) -> str:
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s\n]", "", text)
            logger.info("Removed punctuation")

        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)
            logger.info("Removed numbers")

        return text

    def fix_common_errors(self, text: str) -> str:
        text = re.sub(r"(\w)\1{2,}", r"\1\1", text)

        text = re.sub(r"\s([.,!?])", r"\1", text)

        text = re.sub(r"([.,!?])\s*([.,!?])", r"\1\2", text)

        logger.info("Fixed common text errors")
        return text

    def truncate(self, text: str) -> str:
        if len(text) > self.max_length:
            text = text[: self.max_length]
            logger.warning(f"Text truncated to {self.max_length} characters")
        return text

    def apply_lowercase(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
            logger.info("Applied lowercase")
        return text

    def process(
        self,
        text_path: str,
        output_path: Optional[str] = None,
        keep_original: bool = True,
    ) -> Dict:
        if text_path.endswith(".txt"):
            original = self.load_text(text_path)
        else:
            original = text_path

        original_length = len(original)

        processed = self.normalize_unicode(original)
        processed = self.normalize_whitespace(processed)
        processed = self.fix_common_errors(processed)
        processed = self.remove_special_characters(processed)
        processed = self.apply_lowercase(processed)
        processed = self.truncate(processed)

        processed_length = len(processed)

        word_count = len(processed.split())
        line_count = len(processed.split("\n"))

        result = {
            "original_length": original_length,
            "processed_length": processed_length,
            "word_count": word_count,
            "line_count": line_count,
            "text": processed,
            "original_text": original if keep_original else None,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved processed text to: {output_path}")

        return result

    def process_directory(
        self, input_dir: str, output_dir: str, extensions: List[str] = [".txt", ".json"]
    ) -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for ext in extensions:
            text_files = glob_pattern(input_dir, ext)

        for text_path in text_files:
            try:
                filename = Path(text_path).stem
                output_path = os.path.join(output_dir, f"{filename}_processed.json")

                result = self.process(text_path, output_path, keep_original=True)
                result["filename"] = filename
                result["status"] = "success"
                result["input_path"] = text_path

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

        import pandas as pd

        df = pd.DataFrame(results)
        report_path = os.path.join(output_dir, "processing_report.csv")
        df.to_csv(report_path, index=False)
        logger.info(f"Processing report saved to: {report_path}")

        return results


def glob_pattern(directory: str, pattern: str) -> List[str]:
    import glob

    files = glob.glob(os.path.join(directory, f"*{pattern}"))
    files.extend(glob.glob(os.path.join(directory, f"*{pattern.upper()}")))
    return list(set(files))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument("input", help="Input text file, directory, or raw text")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--lowercase", action="store_true", help="Convert to lowercase")
    parser.add_argument(
        "--remove-punctuation", action="store_true", help="Remove punctuation"
    )
    parser.add_argument("--remove-numbers", action="store_true", help="Remove numbers")
    parser.add_argument("--max-length", type=int, default=10000, help="Max text length")

    args = parser.parse_args()

    preprocessor = TextPreprocessor(
        lowercase=args.lowercase,
        remove_punctuation=args.remove_punctuation,
        remove_numbers=args.remove_numbers,
        max_length=args.max_length,
    )

    if os.path.isdir(args.input):
        results = preprocessor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(results)} files")
        successful = sum(1 for r in results if r.get("status") == "success")
        print(f"Successful: {successful}")
    else:
        result = preprocessor.process(args.input, args.output)
        print(f"Word count: {result['word_count']}")
        print(f"Line count: {result['line_count']}")
        print(f"Processed length: {result['processed_length']}")


if __name__ == "__main__":
    main()
