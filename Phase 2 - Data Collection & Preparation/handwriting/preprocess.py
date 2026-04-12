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


class HandwritingPreprocessor:
    def __init__(
        self,
        target_dpi: int = 300,
        max_dimension: int = 2048,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
    ):
        self.target_dpi = target_dpi
        self.max_dimension = max_dimension
        self.apply_deskew = apply_deskew
        self.apply_denoise = apply_denoise

    def load_image(self, image_path: str) -> np.ndarray:
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            logger.info(f"Loaded image: {image_path}, shape: {img.shape}")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        import cv2

        height, width = img.shape[:2]

        if max(height, width) > self.max_dimension:
            scale = self.max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized to: {new_width}x{new_height}")

        return img

    def convert_to_grayscale(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            logger.info("Converted to grayscale")
            return gray
        return img

    def binarize_image(self, img: np.ndarray) -> np.ndarray:
        import cv2

        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        logger.info("Applied adaptive binarization")
        return binary

    def otsu_binarize(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("Applied Otsu binarization")
        return binary

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        import cv2
        import numpy as np

        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            logger.warning("No text found for deskewing")
            return img

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            logger.info(f"Image already straight (angle: {angle:.2f})")
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        logger.info(f"Deskewed by {angle:.2f} degrees")
        return rotated

    def remove_noise(self, img: np.ndarray) -> np.ndarray:
        import cv2

        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        logger.info("Applied noise removal (morphological open)")
        return opened

    def remove_border(self, img: np.ndarray, border_size: int = 10) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        mask = np.ones((h - 2 * border_size, w - 2 * border_size), dtype=np.uint8) * 255
        result = img.copy()
        result[border_size : h - border_size, border_size : w - border_size] = mask
        logger.info(f"Removed border: {border_size}px")
        return result

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        import cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        logger.info("Applied contrast enhancement")
        return enhanced

    def process(
        self, image_path: str, output_path: Optional[str] = None, enhance: bool = True
    ) -> dict:
        img = self.load_image(image_path)

        original_shape = img.shape

        img = self.resize_image(img)

        img = self.convert_to_grayscale(img)

        if enhance:
            img = self.enhance_contrast(img)

        if self.apply_denoise:
            img = self.remove_noise(img)

        if self.apply_deskew:
            img = self.deskew_image(img)

        img = self.otsu_binarize(img)

        if output_path:
            import cv2

            cv2.imwrite(output_path, img)
            logger.info(f"Saved processed image to: {output_path}")

        return {
            "original_shape": original_shape,
            "processed_shape": img.shape,
            "binarization": "otsu",
            "deskew": self.apply_deskew,
            "denoise": self.apply_denoise,
            "enhance": enhance,
        }

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = [".png", ".jpg", ".jpeg", ".tiff", ".tif"],
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for ext in extensions:
            image_files = glob.glob(os.path.join(input_dir, f"*{ext}"))
            image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

        image_files = list(set(image_files))

        for image_path in image_files:
            try:
                filename = Path(image_path).stem
                ext = Path(image_path).suffix
                output_path = os.path.join(output_dir, f"{filename}_processed{ext}")

                result = self.process(image_path, output_path)
                result["filename"] = filename
                result["status"] = "success"
                result["output_path"] = output_path

                results.append(result)
                logger.info(f"Processed: {filename}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(
                    {
                        "filename": Path(image_path).stem,
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

    parser = argparse.ArgumentParser(
        description="Handwriting Image Preprocessing Pipeline"
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("--no-deskew", action="store_true", help="Skip deskewing")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    parser.add_argument(
        "--max-dimension", type=int, default=2048, help="Max image dimension"
    )

    args = parser.parse_args()

    preprocessor = HandwritingPreprocessor(
        max_dimension=args.max_dimension,
        apply_deskew=not args.no_deskew,
        apply_denoise=not args.no_denoise,
    )

    if os.path.isdir(args.input):
        df = preprocessor.process_directory(args.input, args.output or args.input)
        print(f"\nProcessed {len(df)} images")
        print(df["status"].value_counts())
    else:
        result = preprocessor.process(args.input, args.output)
        print(f"Original shape: {result['original_shape']}")
        print(f"Processed shape: {result['processed_shape']}")


if __name__ == "__main__":
    main()
