# Copyright (c) 2025 PinkRibbon Contributors.
# A little portion of this code were developed and refined with the assistance of ChatGPT.
# 
# File Name: preprcoess.py
# File Description: This script processes raw mammogram images to prepare them for training.
# The dataset contains:
#   - Negative images: images without any lesions.
#   - Positive images: images with lesions (annotated with masks).
#
# Processing steps:
# 1. Negative images:
#    - Divided into large tiles of 598x598 pixels.
#    - Each tile is resized to 299x299 pixels.
#    - Tiles are saved to the processed dataset folder.
#
# 2. Positive images:
#    - Lesion areas are extracted using the corresponding mask.
#    - A small padding is added around the lesion for context.
#    - From this region of interest (ROI), random crops are taken.
#    - Each crop is augmented using random flips (horizontal/vertical) and rotations (0, 90, 180, 270 degrees).
#    - Each augmented crop is resized to 299x299 pixels.
#    - Multiple augmented images are generated per lesion to increase dataset diversity.
#
# Additional details:
# - The output images are stored in separate folders for negative and positive images.
# - The script ensures that all directories exist before saving images.
# - Images that are too small for tiling are resized to the tile size.
# - This preprocessing step standardizes the data, increases dataset size, 
#   and adds variability to help train a robust AI model.
#
# Notes:
#   - NULL


import cv2
import numpy as np
import os
import random

# Paths to raw images
RAW_NEG = "../data/raw/images/negative"
RAW_POS = "../data/raw/images/positive"

# Paths to save processed images
OUT_NEG = "../data/processed/images/negative"
OUT_POS = "../data/processed/images/positive"

# Size settings
TILE_SIZE = 598        # Size of tiles for negative images
OUT_SIZE = 299         # Final output image size
ROI_PADDING = 20       # Extra pixels around positive ROI
AUGMENTATIONS = 3      # Number of random crops + augmentations per positive image


# Create directories if they don't exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Read image in grayscale
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


# Resize image to 299x299
def resize_299(img):
    return cv2.resize(img, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


# Process negative images by tiling them into smaller parts
def process_negative_image(img, base_name):
    h, w = img.shape
    count = 0

    # Slide a TILE_SIZE window over the image
    for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            tile = resize_299(tile)  # Resize tile to 299x299

            # Save the tile
            out_name = f"{base_name}_tile_{count}.png"
            cv2.imwrite(os.path.join(OUT_NEG, out_name), tile)
            count += 1


# Extract region of interest (ROI) from positive image using mask
def extract_roi(image, mask):
    ys, xs = np.where(mask > 0)  # Get coordinates of non-zero pixels
    if len(xs) == 0:
        return None  # No lesion found

    # Get bounding box of ROI
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Add some padding
    x1 = max(0, x1 - ROI_PADDING)
    y1 = max(0, y1 - ROI_PADDING)
    x2 = min(image.shape[1], x2 + ROI_PADDING)
    y2 = min(image.shape[0], y2 + ROI_PADDING)

    return image[y1:y2, x1:x2]


# Random crop of a given size from an image
def random_crop(img):
    h, w = img.shape
    if h < TILE_SIZE or w < TILE_SIZE:
        return cv2.resize(img, (TILE_SIZE, TILE_SIZE))  # Upscale if too small

    y = random.randint(0, h - TILE_SIZE)
    x = random.randint(0, w - TILE_SIZE)
    return img[y:y+TILE_SIZE, x:x+TILE_SIZE]


# Apply random augmentations: flip and rotation
def augment(img):
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # Vertical flip

    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)  # Rotate around center
        img = cv2.warpAffine(img, M, (w, h))

    return img


# Process positive images (extract ROI + crop + augment + resize)
def process_positive_image(img, mask, base_name):
    roi = extract_roi(img, mask)
    if roi is None:
        return  # Skip if no ROI

    for i in range(AUGMENTATIONS):
        crop = random_crop(roi)
        crop = augment(crop)
        crop = resize_299(crop)

        # Save processed crop
        out_name = f"{base_name}_aug_{i}.png"
        cv2.imwrite(os.path.join(OUT_POS, out_name), crop)


# Run preprocessing for both negative and positive images
def run_preprocessing(raw_neg=RAW_NEG, raw_pos=RAW_POS, out_neg=OUT_NEG, out_pos=OUT_POS):
    ensure_dir(out_neg)
    ensure_dir(out_pos)

    # Process negative images
    for file in os.listdir(raw_neg):
        if not file.lower().endswith((".png", ".jpg")):
            continue
        img = read_gray(os.path.join(raw_neg, file))
        base = os.path.splitext(file)[0]
        process_negative_image(img, base)

    # Process positive images
    for file in os.listdir(raw_pos):
        if "_mask" in file:
            continue  # Skip mask files themselves
        if not file.lower().endswith((".png", ".jpg")):
            continue

        img_path = os.path.join(raw_pos, file)
        mask_path = img_path.replace(".png", "_mask.png")

        if not os.path.exists(mask_path):
            continue  # Skip if mask does not exist

        img = read_gray(img_path)
        mask = read_gray(mask_path)

        base = os.path.splitext(file)[0]
        process_positive_image(img, mask, base)


if __name__ == "__main__":
    run_preprocessing()
