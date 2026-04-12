import hashlib
import secrets
import logging
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
import csv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AnonymizationConfig:
    algorithm: str = "sha256"
    salt_length: int = 16
    preserve_age: bool = True
    preserve_grade: bool = True
    pii_patterns: List[str] = field(
        default_factory=lambda: [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b\d{10}\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ]
    )


class Anonymizer:
    def __init__(self, config: Optional[AnonymizationConfig] = None):
        self.config = config or AnonymizationConfig()
        self.salt = None

    def generate_salt(self) -> str:
        return secrets.token_hex(self.config.salt_length)

    def set_salt(self, salt: str):
        self.salt = salt

    def hash_identifier(self, identifier: str, salt: Optional[str] = None) -> str:
        salt = salt or self.salt or self.generate_salt()

        if self.config.algorithm == "sha256":
            hasher = hashlib.sha256
        elif self.config.algorithm == "sha512":
            hasher = hashlib.sha512
        elif self.config.algorithm == "blake2b":
            hasher = hashlib.blake2b
        else:
            hasher = hashlib.sha256

        combined = f"{identifier}{salt}"
        return hasher(combined.encode()).hexdigest()[:16]

    def anonymize_student_id(self, student_id: str) -> Tuple[str, str]:
        salt = self.generate_salt()
        student_hash = self.hash_identifier(student_id, salt)
        return student_hash, salt

    def remove_pii_from_text(self, text: str) -> str:
        anonymized = text

        anonymized = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[NAME]", anonymized)

        anonymized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", anonymized)

        anonymized = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            anonymized,
        )

        anonymized = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", anonymized)

        logger.info("Removed PII from text")
        return anonymized

    def anonymize_metadata(
        self, metadata: Dict, fields_to_remove: List[str] = None
    ) -> Dict:
        if fields_to_remove is None:
            fields_to_remove = [
                "name",
                "first_name",
                "last_name",
                "full_name",
                "email",
                "phone",
                "address",
                "dob",
                "date_of_birth",
                "parent_name",
                "parent_email",
                "student_id",
                "original_id",
            ]

        anonymized = {}

        for key, value in metadata.items():
            if key.lower() in fields_to_remove:
                continue

            if key.lower() in ["age", "grade"]:
                if key.lower() == "age":
                    anonymized["age"] = value if self.config.preserve_age else None
                elif key.lower() == "grade":
                    anonymized["grade"] = value if self.config.preserve_grade else None
            else:
                anonymized[key] = value

        return anonymized

    def process_student_record(
        self, student_id: str, metadata: Dict, text_content: Optional[str] = None
    ) -> Dict:
        student_hash, salt = self.anonymize_student_id(student_id)

        anonymized = {
            "student_hash": student_hash,
            "salt": salt,
            "age": metadata.get("age"),
            "grade": metadata.get("grade"),
            "gender": metadata.get("gender"),
        }

        anonymized_metadata = self.anonymize_metadata(metadata)
        anonymized["additional_metadata"] = anonymized_metadata

        if text_content:
            anonymized["text_sample"] = self.remove_pii_from_text(text_content)

        return anonymized


class DataAnonymizer:
    def __init__(self, salt: Optional[str] = None):
        self.anonymizer = Anonymizer()
        if salt:
            self.anonymizer.set_salt(salt)

    def anonymize_csv(
        self,
        input_path: str,
        output_path: str,
        id_column: str = "student_id",
        pii_columns: List[str] = None,
    ):
        if pii_columns is None:
            pii_columns = ["name", "email", "phone", "address"]

        with open(input_path, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames

            new_fieldnames = []
            for field in fieldnames:
                if field == id_column:
                    new_fieldnames.append("student_hash")
                elif field.lower() not in pii_columns:
                    new_fieldnames.append(field)

            results = []
            for row in reader:
                if id_column in row:
                    student_hash, _ = self.anonymizer.anonymize_student_id(
                        row[id_column]
                    )
                    row["student_hash"] = student_hash

                for col in pii_columns:
                    if col in row:
                        del row[col]

                results.append(row)

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Anonymized CSV saved to {output_path}")

    def anonymize_json(
        self,
        input_path: str,
        output_path: str,
        id_field: str = "student_id",
        data_field: str = "text_samples",
    ):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if id_field in item:
                    student_hash, _ = self.anonymizer.anonymize_student_id(
                        item[id_field]
                    )
                    item["student_hash"] = student_hash
                    del item[id_field]

                if data_field in item and isinstance(item[data_field], str):
                    item[data_field] = self.anonymizer.remove_pii_from_text(
                        item[data_field]
                    )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Anonymized JSON saved to {output_path}")

    def verify_anonymization(
        self, original_path: str, anonymized_path: str, check_columns: List[str]
    ) -> Dict:
        with open(original_path, "r", encoding="utf-8") as f:
            original = (
                json.load(f) if original_path.endswith(".json") else csv.DictReader(f)
            )

        with open(anonymized_path, "r", encoding="utf-8") as f:
            anonymized = (
                json.load(f) if anonymized_path.endswith(".json") else csv.DictReader(f)
            )

        issues = []

        for col in check_columns:
            if col in original:
                if col not in anonymized:
                    issues.append(f"Column {col} not found in anonymized data")

        hasher = hashlib.sha256()
        for item in anonymized:
            if "student_hash" in item:
                if len(item["student_hash"]) != 16:
                    issues.append(f"Invalid hash length: {item['student_hash']}")

        return {"passed": len(issues) == 0, "issues": issues}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data Anonymization Pipeline")
    parser.add_argument("--input", required=True, help="Input file (CSV or JSON)")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--salt", help="Salt for hashing (optional)")
    parser.add_argument("--type", choices=["csv", "json"], help="Input file type")
    parser.add_argument("--id-column", default="student_id", help="ID column name")
    parser.add_argument("--verify", action="store_true", help="Verify anonymization")

    args = parser.parse_args()

    if args.type is None:
        args.type = "json" if args.input.endswith(".json") else "csv"

    anonymizer = DataAnonymizer(salt=args.salt)

    if args.type == "csv":
        anonymizer.anonymize_csv(args.input, args.output, args.id_column)
    elif args.type == "json":
        anonymizer.anonymize_json(args.input, args.output, args.id_column)

    print(f"Anonymization complete: {args.output}")


if __name__ == "__main__":
    main()
