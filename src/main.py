# Copyright (c) 2025 PinkRibbon Contributors.
# A little portion of this code were developed and refined with the assistance of ChatGPT.
# 
# File Name: main.py
# File Description: NULL
# Notes:
#   - NULL

from preprocess import run_preprocessing

def main():
    print("Starting preprocessing of mammogram images...")
    
    raw_neg_path = "../data/raw/images/negative"
    raw_pos_path = "../data/raw/images/positive"
    out_neg_path = "../data/processed/images/negative"
    out_pos_path = "../data/processed/images/positive"

    # Run preprocessing
    run_preprocessing(raw_neg_path, raw_pos_path, out_neg_path, out_pos_path)

    print("Preprocessing completed. Ready for training model!")

if __name__ == "__main__":
    main()
