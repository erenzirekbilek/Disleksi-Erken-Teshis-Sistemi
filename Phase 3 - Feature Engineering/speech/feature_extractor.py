import os
import glob
import logging
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpeechFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 48000,
        frame_length: int = 0.025,
        frame_shift: int = 0.010,
        n_mfcc: int = 13,
        n_mels: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.win_length = int(frame_length * sample_rate)
        self.hop_length = int(frame_shift * sample_rate)

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio: {audio_path}, duration: {len(y) / sr:.2f}s")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def extract_mfcc(self, y: np.ndarray, sr: int) -> Dict:
        import librosa

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        features = {
            "mfcc_mean": np.mean(mfcc_combined, axis=1).tolist(),
            "mfcc_std": np.std(mfcc_combined, axis=1).tolist(),
            "mfcc_min": np.min(mfcc_combined, axis=1).tolist(),
            "mfcc_max": np.max(mfcc_combined, axis=1).tolist(),
            "mfcc": mfcc.tolist(),
            "mfcc_delta": mfcc_delta.tolist(),
            "mfcc_delta2": mfcc_delta2.tolist(),
        }

        logger.info(f"Extracted MFCC features: {self.n_mfcc * 3} values")
        return features

    def extract_pitch(self, y: np.ndarray, sr: int) -> Dict:
        import librosa

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=self.hop_length,
        )

        f0_filled = np.nan_to_num(f0, nan=0.0)

        features = {
            "pitch_mean": float(np.mean(f0_filled[f0_filled > 0]))
            if np.any(f0_filled > 0)
            else 0.0,
            "pitch_std": float(np.std(f0_filled[f0_filled > 0]))
            if np.any(f0_filled > 0)
            else 0.0,
            "pitch_min": float(np.min(f0_filled[f0_filled > 0]))
            if np.any(f0_filled > 0)
            else 0.0,
            "pitch_max": float(np.max(f0_filled[f0_filled > 0]))
            if np.any(f0_filled > 0)
            else 0.0,
            "pitch_range": float(np.max(f0_filled) - np.min(f0_filled))
            if np.any(f0_filled > 0)
            else 0.0,
            "voiced_ratio": float(np.mean(voiced_flag))
            if len(voiced_flag) > 0
            else 0.0,
            "pitch_continuity": float(self._pitch_continuity(f0)),
            "f0": f0.tolist(),
            "voiced_flag": voiced_flag.tolist(),
        }

        logger.info("Extracted pitch features")
        return features

    def _pitch_continuity(self, f0: np.ndarray) -> float:
        f0_filled = np.nan_to_num(f0, nan=0.0)
        if len(f0_filled) < 2:
            return 0.0

        diff = np.diff(f0_filled)
        continuity = np.sum(np.abs(diff) < 50) / len(diff)
        return float(continuity)

    def extract_energy(self, y: np.ndarray, sr: int) -> Dict:
        import librosa

        rms = librosa.feature.rms(
            y=y, frame_length=self.win_length, hop_length=self.hop_length
        )[0]

        energy = rms**2

        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=self.win_length, hop_length=self.hop_length
        )[0]

        features = {
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "energy_min": float(np.min(energy)),
            "energy_max": float(np.max(energy)),
            "energy_range": float(np.max(energy) - np.min(energy)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "rms": rms.tolist(),
            "zcr": zcr.tolist(),
        }

        logger.info("Extracted energy features")
        return features

    def extract_spectral(self, y: np.ndarray, sr: int) -> Dict:
        import librosa

        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]

        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        spectral_flux = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )

        features = {
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "spectral_centroid_std": float(np.std(spectral_centroid)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_rolloff_std": float(np.std(spectral_rolloff)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
            "spectral_contrast_mean": np.mean(spectral_contrast, axis=1).tolist(),
            "spectral_contrast_std": np.std(spectral_contrast, axis=1).tolist(),
            "spectral_flux_mean": float(np.mean(spectral_flux)),
            "spectral_flux_std": float(np.std(spectral_flux)),
        }

        logger.info("Extracted spectral features")
        return features

    def extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict:
        import librosa

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)

        duration = len(y) / sr

        silence = librosa.effects.split(y, top_db=20)
        pause_count = len(silence)
        total_pause_duration = sum((end - start) / sr for start, end in silence)
        pause_ratio = total_pause_duration / duration if duration > 0 else 0

        features = {
            "duration_seconds": float(duration),
            "tempo": float(tempo),
            "beat_count": int(len(beats)),
            "onset_count": int(len(onset_env)),
            "onset_density": float(len(onset_env) / duration) if duration > 0 else 0,
            "pause_count": pause_count,
            "pause_duration_total": float(total_pause_duration),
            "pause_ratio": float(pause_ratio),
            "speaking_rate_estimated": float(
                len(y) / sr / (duration - total_pause_duration)
            )
            if (duration - total_pause_duration) > 0
            else 0,
        }

        logger.info("Extracted temporal features")
        return features

    def extract_all(self, audio_path: str) -> Dict:
        y, sr = self.load_audio(audio_path)

        features = {}

        features.update(self.extract_mfcc(y, sr))
        features.update(self.extract_pitch(y, sr))
        features.update(self.extract_energy(y, sr))
        features.update(self.extract_spectral(y, sr))
        features.update(self.extract_temporal_features(y, sr))

        features["audio_path"] = audio_path
        features["sample_rate"] = sr
        features["samples"] = len(y)

        logger.info(f"Total features extracted: {len(features)}")
        return features

    def process_file(self, audio_path: str, output_path: Optional[str] = None) -> Dict:
        features = self.extract_all(audio_path)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved features to: {output_path}")

        return features

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".wav", ".mp3", ".flac"],
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for ext in extensions:
            audio_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
            audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        audio_files = list(set(audio_files))

        for audio_path in audio_files:
            try:
                filename = Path(audio_path).stem
                output_path = os.path.join(output_dir, f"{filename}_features.json")

                features = self.process_file(audio_path, output_path)

                result = {
                    "filename": filename,
                    "status": "success",
                    "output_path": output_path,
                    "duration": features.get("duration_seconds", 0),
                    "tempo": features.get("tempo", 0),
                    "mfcc_mean_0": features.get("mfcc_mean", [0] * 39)[0]
                    if features.get("mfcc_mean")
                    else 0,
                }
                results.append(result)
                logger.info(f"Processed: {filename}")

            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append(
                    {
                        "filename": Path(audio_path).stem,
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

    parser = argparse.ArgumentParser(description="Speech Feature Extraction")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCCs")
    parser.add_argument("--matrix", action="store_true", help="Create feature matrix")

    args = parser.parse_args()

    extractor = SpeechFeatureExtractor(sample_rate=args.sample_rate, n_mfcc=args.n_mfcc)

    if os.path.isdir(args.input):
        df = extractor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(df)} files")
        print(df["status"].value_counts())

        if args.matrix and args.output:
            create_feature_matrix(args.output)
    else:
        result = extractor.process_file(args.input, args.output)
        print(f"Extracted {len(result)} feature groups")
        print(f"Duration: {result.get('duration_seconds', 0):.2f}s")


if __name__ == "__main__":
    main()
