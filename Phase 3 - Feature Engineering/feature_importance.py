import os
import glob
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(self):
        self.features_df = None
        self.labels_df = None

    def load_features(self, features_dir: str) -> pd.DataFrame:
        feature_files = glob.glob(os.path.join(features_dir, "*.json"))

        all_features = []
        for f in feature_files:
            with open(f, "r") as fp:
                features = json.load(fp)

            flat_features = self._flatten_dict(features)
            all_features.append(flat_features)

        df = pd.DataFrame(all_features)

        if "text_path" in df.columns:
            df["sample_id"] = df["text_path"].apply(lambda x: Path(x).stem)
        elif "image_path" in df.columns:
            df["sample_id"] = df["image_path"].apply(lambda x: Path(x).stem)
        elif "audio_path" in df.columns:
            df["sample_id"] = df["audio_path"].apply(lambda x: Path(x).stem)

        self.features_df = df
        logger.info(f"Loaded {len(df)} feature samples with {len(df.columns)} columns")
        return df

    def load_labels(self, labels_file: str) -> pd.DataFrame:
        if labels_file.endswith(".json"):
            with open(labels_file, "r") as f:
                labels = json.load(f)
            df = pd.DataFrame(labels)
        elif labels_file.endswith(".csv"):
            df = pd.read_csv(labels_file)
        else:
            raise ValueError("Labels file must be JSON or CSV")

        if "text_path" in df.columns:
            df["sample_id"] = df["text_path"].apply(lambda x: Path(x).stem)
        elif "image_path" in df.columns:
            df["sample_id"] = df["image_path"].apply(lambda x: Path(x).stem)
        elif "audio_path" in df.columns:
            df["sample_id"] = df["audio_path"].apply(lambda x: Path(x).stem)

        self.labels_df = df
        logger.info(f"Loaded {len(df)} labels")
        return df

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) > 10:
                continue
            elif isinstance(v, (int, float, str)):
                items.append((new_key, v))
        return dict(items)

    def merge_features_labels(self) -> pd.DataFrame:
        if self.features_df is None or self.labels_df is None:
            raise ValueError("Features and labels must be loaded first")

        merged = pd.merge(self.features_df, self.labels_df, on="sample_id", how="inner")
        logger.info(f"Merged dataset: {len(merged)} samples")
        return merged

    def calculate_correlation_importance(
        self, target_column: str = "overall_score", method: str = "pearson"
    ) -> pd.DataFrame:
        if self.features_df is None:
            raise ValueError("Features must be loaded first")

        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != "sample_id"]

        correlations = []
        for col in numeric_cols:
            try:
                if method == "pearson":
                    corr, p_value = stats.pearsonr(
                        self.features_df[col].fillna(0),
                        self.labels_df.set_index("sample_id")
                        .loc[self.features_df["sample_id"], target_column]
                        .fillna(0),
                    )
                else:
                    corr, p_value = stats.spearmanr(
                        self.features_df[col].fillna(0),
                        self.labels_df.set_index("sample_id")
                        .loc[self.features_df["sample_id"], target_column]
                        .fillna(0),
                    )

                correlations.append(
                    {
                        "feature": col,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {e}")

        df = pd.DataFrame(correlations)
        df = df.sort_values("abs_correlation", ascending=False)

        logger.info(f"Calculated correlations for {len(df)} features")
        return df

    def calculate_auc_importance(
        self, target_column: str = "overall_risk", threshold: float = 0.5
    ) -> pd.DataFrame:
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            logger.warning("sklearn not available, using correlation instead")
            return self.calculate_correlation_importance()

        if self.features_df is None or self.labels_df is None:
            raise ValueError("Features and labels must be loaded first")

        merged = self.merge_features_labels()

        y = (merged[target_column] >= threshold).astype(int)

        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            c for c in numeric_cols if c not in ["sample_id", target_column]
        ]

        auc_scores = []
        for col in numeric_cols:
            try:
                if len(y.unique()) < 2:
                    continue

                X = merged[col].fillna(0).values.reshape(-1, 1)
                auc = roc_auc_score(y, X)
                auc = max(auc, 1 - auc)

                auc_scores.append(
                    {"feature": col, "auc": auc, "auc_deviation": abs(auc - 0.5) * 2}
                )
            except Exception as e:
                logger.warning(f"Could not calculate AUC for {col}: {e}")

        df = pd.DataFrame(auc_scores)
        df = df.sort_values("auc_deviation", ascending=False)

        logger.info(f"Calculated AUC for {len(df)} features")
        return df

    def calculate_mutual_information(
        self, target_column: str = "overall_risk"
    ) -> pd.DataFrame:
        try:
            from sklearn.feature_selection import mutual_info_classif
        except ImportError:
            logger.warning("sklearn not available, using correlation instead")
            return self.calculate_correlation_importance()

        if self.features_df is None or self.labels_df is None:
            raise ValueError("Features and labels must be loaded first")

        merged = self.merge_features_labels()

        if merged[target_column].dtype == "object":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(merged[target_column])
        else:
            y = merged[target_column]

        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            c for c in numeric_cols if c not in ["sample_id", target_column]
        ]

        X = merged[numeric_cols].fillna(0).values

        mi_scores = mutual_info_classif(X, y, random_state=42)

        df = pd.DataFrame(
            {"feature": numeric_cols, "mutual_information": mi_scores}
        ).sort_values("mutual_information", ascending=False)

        logger.info(f"Calculated mutual information for {len(df)} features")
        return df

    def calculate_tree_importance(
        self, target_column: str = "overall_risk", n_estimators: int = 100
    ) -> pd.DataFrame:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.warning("sklearn not available, using correlation instead")
            return self.calculate_correlation_importance()

        if self.features_df is None or self.labels_df is None:
            raise ValueError("Features and labels must be loaded first")

        merged = self.merge_features_labels()

        if merged[target_column].dtype == "object":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(merged[target_column])
        else:
            y = merged[target_column]

        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            c for c in numeric_cols if c not in ["sample_id", target_column]
        ]

        X = merged[numeric_cols].fillna(0).values

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X, y)

        df = pd.DataFrame(
            {"feature": numeric_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info(f"Calculated tree importance for {len(df)} features")
        return df

    def rank_features(
        self, target_column: str = "overall_risk", methods: List[str] = None
    ) -> pd.DataFrame:
        if methods is None:
            methods = ["correlation", "auc", "tree"]

        results = {}

        if "correlation" in methods:
            results["correlation"] = self.calculate_correlation_importance(
                target_column
            )

        if "auc" in methods:
            results["auc"] = self.calculate_auc_importance(target_column)

        if "mutual_info" in methods:
            results["mutual_info"] = self.calculate_mutual_information(target_column)

        if "tree" in methods:
            results["tree"] = self.calculate_tree_importance(target_column)

        combined = None
        for method, df in results.items():
            df_method = df.copy()
            df_method["rank"] = range(1, len(df_method) + 1)
            df_method = df_method.rename(
                columns={
                    df_method.columns[0]: "feature",
                    df_method.columns[1]: f"{method}_score",
                }
            )

            if combined is None:
                combined = df_method[["feature", f"{method}_score", "rank"]]
                combined = combined.rename(columns={"rank": f"{method}_rank"})
            else:
                combined = combined.merge(
                    df_method[["feature", f"{method}_score", "rank"]],
                    on="feature",
                    how="outer",
                )
                combined = combined.rename(columns={"rank": f"{method}_rank"})

        rank_cols = [c for c in combined.columns if c.endswith("_rank")]
        combined["avg_rank"] = combined[rank_cols].mean(axis=1)
        combined = combined.sort_values("avg_rank")

        logger.info("Feature ranking complete")
        return combined

    def get_top_features(
        self, n: int = 20, target_column: str = "overall_risk"
    ) -> pd.DataFrame:
        ranking = self.rank_features(target_column)
        return ranking.head(n)

    def generate_report(self, output_path: str, target_column: str = "overall_risk"):
        ranking = self.rank_features(target_column)

        report = {
            "total_features": len(ranking),
            "top_20_features": ranking.head(20).to_dict("records"),
            "top_10_per_modality": self._group_by_modality(ranking),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_path}")
        return report

    def _group_by_modality(self, ranking: pd.DataFrame) -> Dict:
        modality_features = {"speech": [], "handwriting": [], "text": []}

        for _, row in ranking.iterrows():
            feature = row["feature"].lower()
            if any(
                x in feature
                for x in ["mfcc", "pitch", "energy", "spectral", "tempo", "duration"]
            ):
                modality_features["speech"].append(row.to_dict())
            elif any(
                x in feature
                for x in [
                    "size",
                    "baseline",
                    "spacing",
                    "reversal",
                    "stroke",
                    "pressure",
                ]
            ):
                modality_features["handwriting"].append(row.to_dict())
            elif any(
                x in feature
                for x in [
                    "word",
                    "sentence",
                    "spelling",
                    "grammar",
                    "flesch",
                    "vocabulary",
                ]
            ):
                modality_features["text"].append(row.to_dict())

        return {
            modality: features[:10] for modality, features in modality_features.items()
        }


def analyze_feature_importance(
    features_dir: str,
    labels_file: str,
    output_dir: str,
    target_column: str = "overall_risk",
):
    os.makedirs(output_dir, exist_ok=True)

    analyzer = FeatureImportanceAnalyzer()

    print("Loading features...")
    analyzer.load_features(features_dir)

    print("Loading labels...")
    analyzer.load_labels(labels_file)

    print("\nCalculating importance scores...")

    print("1. Correlation analysis...")
    corr_df = analyzer.calculate_correlation_importance(target_column)
    corr_path = os.path.join(output_dir, "correlation_importance.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"   Saved to {corr_path}")

    print("2. AUC analysis...")
    auc_df = analyzer.calculate_auc_importance(target_column)
    auc_path = os.path.join(output_dir, "auc_importance.csv")
    auc_df.to_csv(auc_path, index=False)
    print(f"   Saved to {auc_path}")

    print("3. Random Forest importance...")
    tree_df = analyzer.calculate_tree_importance(target_column)
    tree_path = os.path.join(output_dir, "tree_importance.csv")
    tree_df.to_csv(tree_path, index=False)
    print(f"   Saved to {tree_path}")

    print("\nGenerating combined ranking...")
    ranking = analyzer.rank_features(target_column)
    ranking_path = os.path.join(output_dir, "feature_ranking.csv")
    ranking.to_csv(ranking_path, index=False)
    print(f"   Saved to {ranking_path}")

    print("\nTop 20 features:")
    print(ranking.head(20)[["feature", "avg_rank"]].to_string())

    print("\nGenerating full report...")
    report_path = os.path.join(output_dir, "feature_importance_report.json")
    analyzer.generate_report(report_path, target_column)
    print(f"   Report saved to {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument("--features", required=True, help="Features directory")
    parser.add_argument("--labels", required=True, help="Labels file (JSON/CSV)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target", default="overall_risk", help="Target column name")

    args = parser.parse_args()

    analyze_feature_importance(
        features_dir=args.features,
        labels_file=args.labels,
        output_dir=args.output,
        target_column=args.target,
    )


if __name__ == "__main__":
    main()
