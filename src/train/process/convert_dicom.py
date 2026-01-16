# Copyright (c) 2026 Pritilata AI Contributors
#
# File Name: convert_dicom.py
# Description: Converts DICOM mammogram images to PNG format with optional resizing and normalization. Supports 8-bit or 16-bit PNG output and applies DICOM rescale slope/intercept automatically.
# Notes: Input and output paths are resolved relative to the project root.
# Flow: This script is intended to be run manually after dataset download and before metadata generation.

import numpy as np
import pydicom
import cv2
import png
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]

DICOM_ROOT = BASE_DIR / "data/train/ddsm/images/dicom"
PNG_ROOT = BASE_DIR / "data/train/ddsm/images/png"


def save_dicom_image_as_png(
    dicom_path: Path,
    png_path: Path,
    target_size=(896, 1152),
    output_bitdepth=16,
):
    dicom_path = Path(dicom_path).resolve()
    png_path = Path(png_path).resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ds = pydicom.dcmread(dicom_path)
        image = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        if target_size is not None:
            image = cv2.resize(
                image,
                dsize=target_size,
                interpolation=cv2.INTER_CUBIC,
            )

        max_val = (2**output_bitdepth) - 1
        image -= image.min()
        image /= image.max()
        image *= max_val
        image = image.astype(np.uint16 if output_bitdepth > 8 else np.uint8)

        with open(png_path, "wb") as f:
            writer = png.Writer(
                width=image.shape[1],
                height=image.shape[0],
                bitdepth=output_bitdepth,
                greyscale=True,
            )
            writer.write(f, image.tolist())

    except Exception as e:
        raise RuntimeError(f"Failed to process {dicom_path}") from e
