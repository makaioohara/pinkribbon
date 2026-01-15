# Copyright (c) 2025 PinkRibbon Contributors
#
# File Name: main.py
# Description: Entry point for processing test images
# Notes: Processes all images in the ../data/test directory using the preprocessing pipeline

import os
from preprocess import preprocess_image  # function from preprocess.py

TEST_DATA_DIR = "../data/test"

def main():

    # Checking if the test directory exists
    if not os.path.exists(TEST_DATA_DIR):
        print("Error: data/test directory not found.")
        return

    # Listing all files inside data/test
    image_files = os.listdir(TEST_DATA_DIR)

    for file_name in image_files:
        image_path = os.path.join(TEST_DATA_DIR, file_name)

        # Skip folders or non-image files
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        print("Processing image...")

        # Send the image path to preprocess.py
        preprocess_image(image_path)

    print("All images processed successfully.")

# Run the main function only if this file is executed directly
if __name__ == "__main__":
    main()
