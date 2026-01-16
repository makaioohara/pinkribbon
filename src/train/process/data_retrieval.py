# Copyright (c) 2026 Pritilata AI Contributors
#
# File Name: data_retrieval.py
# Description: Builds structured training and test metadata CSV files for the CBIS-DDSM dataset by merging mammogram images with their corresponding annotation masks.
# Output CSV Schema:
#   1. Mammogram Path : Local filesystem path to the mammogram image
#   2. Mask Path      : Local filesystem path to the corresponding annotation mask
#   3. Label          : Classification label encoded as:
#                       1 – Malignant Calcification
#                       2 – Benign Calcification
#                       3 – Malignant Mass
#                       4 – Benign Mass
# Notes: This script relies on the official train/test split provided by the dataset authors. Ensure all required CSV metadata files and image directories exist before execution.
# Flow: This script is typically manually executed after image preprocessing (e.g., DICOM-to-PNG conversion).
#       It generates two output files:
#           - training_dataset.csv
#           - test_dataset.csv

from pathlib import Path
from retrieval_utils import mam_data, mask_data, final_dataset


BASE_DIR = Path(__file__).resolve().parents[3]

IMG_ROOT = BASE_DIR / "data/train/ddsm/images/png"
CSV_ROOT = BASE_DIR / "data/train/ddsm/csv"
OUTPUT_DIR = BASE_DIR / "data/train/ddsm/csv/retrieved"


def build_datasets(img_root: Path, csv_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_mam_path = mam_data(str(img_root), str(csv_root))
    csv_mask_path = mask_data(str(img_root), str(csv_root))

    final_dataset(csv_mask_path, csv_mam_path, str(output_dir))


if __name__ == "__main__":
    build_datasets(IMG_ROOT, CSV_ROOT, OUTPUT_DIR)
