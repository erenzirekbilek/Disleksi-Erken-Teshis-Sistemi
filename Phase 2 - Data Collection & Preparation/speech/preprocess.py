import os
import glob
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 48000,
        target_lufs: float = -20.0,
        top_db: int = 20,
        noise_reduction_strength: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.top_db = top_db
        self.noise_reduction_strength = noise_reduction_strength

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio: {audio_path}, duration: {len(y) / sr:.2f}s")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def remove_silence(
        self, y: np.ndarray, frame_length: int = 2048, hop_length: int = 512
    ) -> np.ndarray:
        try:
            import librosa

            y_trimmed, _ = librosa.effects.trim(y, top_db=self.top_db)
            logger.info(f"Trimmed silence: {len(y)} -> {len(y_trimmed)} samples")
            return y_trimmed
        except Exception as e:
            logger.error(f"Error removing silence: {e}")
            return y

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        try:
            import librosa

            y_normalized = librosa.util.normalize(y)

            target_peak = 0.95
            peak = np.abs(y_normalized).max()
            if peak > 0:
                y_normalized = y_normalized * (target_peak / peak)

            logger.info(f"Normalized audio: peak = {peak:.4f}")
            return y_normalized
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return y

    def reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            import noisereduce as nr

            y_denoised = nr.reduce_noise(
                y=y, sr=sr, stationary=True, prop_decrease=self.noise_reduction_strength
            )
            logger.info("Applied noise reduction")
            return y_denoised
        except ImportError:
            logger.warning("noisereduce not installed, skipping noise reduction")
            return y
        except Exception as e:
            logger.error(f"Error reducing noise: {e}")
            return y

    def apply_filters(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            from scipy.signal import butter, lfilter

            lowcut = 80
            highcut = 8000

            nyquist = sr / 2
            low = lowcut / nyquist
            high = highcut / nyquist

            b, a = butter(4, [low, high], btype="band")
            y_filtered = lfilter(b, a, y)

            logger.info("Applied bandpass filter (80Hz - 8kHz)")
            return y_filtered
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return y

    def process(self, audio_path: str, output_path: Optional[str] = None) -> dict:
        y, sr = self.load_audio(audio_path)

        original_duration = len(y) / sr

        y = self.reduce_noise(y, sr)

        y = self.remove_silence(
            y, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr)
        )

        y = self.apply_filters(y, sr)

        y = self.normalize_audio(y)

        final_duration = len(y) / sr

        if output_path:
            import soundfile as sf

            sf.write(output_path, y, sr)
            logger.info(f"Saved processed audio to: {output_path}")

        return {
            "original_duration": original_duration,
            "final_duration": final_duration,
            "sample_rate": sr,
            "samples": len(y),
            "process_info": {
                "noise_reduction": True,
                "silence_trimming": True,
                "bandpass_filter": True,
                "normalization": True,
            },
        }

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
                output_path = os.path.join(output_dir, f"{filename}_processed.wav")

                result = self.process(audio_path, output_path)
                result["filename"] = filename
                result["status"] = "success"
                result["output_path"] = output_path

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
        report_path = os.path.join(output_dir, "processing_report.csv")
        df.to_csv(report_path, index=False)
        logger.info(f"Processing report saved to: {report_path}")

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audio Preprocessing Pipeline")
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "--sample-rate", type=int, default=48000, help="Target sample rate"
    )
    parser.add_argument("--target-lufs", type=float, default=-20.0, help="Target LUFS")
    parser.add_argument("--top-db", type=int, default=20, help="Silence threshold")

    args = parser.parse_args()

    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate, target_lufs=args.target_lufs, top_db=args.top_db
    )

    if os.path.isdir(args.input):
        df = preprocessor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(df)} files")
        print(df["status"].value_counts())
    else:
        result = preprocessor.process(args.input, args.output)
        print(f"Original duration: {result['original_duration']:.2f}s")
        print(f"Final duration: {result['final_duration']:.2f}s")
        print(f"Sample rate: {result['sample_rate']} Hz")


if __name__ == "__main__":
    main()
