# Copyright (c) 2026 Pritilata AI Contributors
#
# File Name: retrieval_utils.py
# Description: Provides utility functions to retrieve and merge CBIS-DDSM mammogram and mask data. Includes functions to locate mammograms, locate masks, and generate final training/test CSV datasets. This file supports the dataset preparation stage of the pipeline and provides:
#   1. mam_data       – Collects and stores local paths for all mammogram images.
#   2. mask_data      – Collects and stores local paths for all ROI mask annotations.
#   3. final_dataset  – Merges mammogram and mask metadata using official dataset splits and outputs finalized training and test CSV files.
# Notes: NULL
# Flow: This file is automatically executed when data_retrieval.py is manually run. The final output consists of training_dataset.csv and test_dataset.csv. It is imported and used by data_retrieval.py to build structured CSV metadata.

from pathlib import Path
import shutil
import pandas as pd
import pydicom


TYPE_MAP = {
    "Calc-Test": "calc_case_description_test_set.csv",
    "Calc-Training": "calc_case_description_train_set.csv",
    "Mass-Test": "mass_case_description_test_set.csv",
    "Mass-Training": "mass_case_description_train_set.csv",
}


def _normalize_pathology(series: pd.Series) -> pd.Series:
    return series.replace({"BENIGN_WITHOUT_CALLBACK": "BENIGN"})


def mam_data(img_root: str, csv_root: str) -> str:
    img_root = Path(img_root).resolve()
    csv_root = Path(csv_root).resolve()
    output_dir = Path.cwd() / "temp_mammograms_csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for folder in img_root.iterdir():
        if folder.name.endswith(("_CC", "_MLO")):
            files = list(folder.rglob("*"))
            if files:
                records.append(
                    {
                        "img": folder.name,
                        "img_path": str(files[-1]),
                    }
                )

    df_imgs = pd.DataFrame(records)

    for prefix, csv_name in TYPE_MAP.items():
        df_subset = df_imgs[df_imgs["img"].str.startswith(prefix)]

        df_meta = pd.read_csv(
            csv_root / csv_name,
            usecols=["pathology", "image file path"],
        )

        df_meta["img"] = df_meta["image file path"].str.split("/").str[0]
        df_meta["pathology"] = _normalize_pathology(df_meta["pathology"])
        df_meta = df_meta.drop_duplicates(subset="img")

        merged = df_subset.merge(df_meta, on="img", how="inner")
        merged.drop(columns=["image file path"], inplace=True)

        merged.to_csv(output_dir / f"{prefix.lower()}.csv", index=False)

    return str(output_dir)


def mask_data(img_root: str, csv_root: str) -> str:
    img_root = Path(img_root).resolve()
    csv_root = Path(csv_root).resolve()
    output_dir = Path.cwd() / "temp_masks_csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = {}

    for folder in img_root.iterdir():
        if not folder.name[-1].isdigit():
            continue

        for path in folder.rglob("*.dcm"):
            try:
                img = pydicom.dcmread(path).pixel_array
                if img.dtype == "uint16":
                    records[folder.name] = str(path)
                    break
            except Exception:
                continue

    df_masks = pd.DataFrame(
        records.items(), columns=["img", "roi_path"]
    )

    for prefix, csv_name in TYPE_MAP.items():
        df_subset = df_masks[df_masks["img"].str.startswith(prefix)]

        df_meta = pd.read_csv(
            csv_root / csv_name,
            usecols=["pathology", "ROI mask file path"],
        )

        df_meta["img"] = df_meta["ROI mask file path"].str.split("/").str[0]
        df_meta["pathology"] = _normalize_pathology(df_meta["pathology"])

        merged = df_subset.merge(df_meta, on="img", how="inner")
        merged.drop(columns=["ROI mask file path"], inplace=True)

        merged.to_csv(output_dir / f"{prefix.lower()}.csv", index=False)

    return str(output_dir)


def final_dataset(
    csv_mask_path: str,
    csv_mam_path: str,
    output_path: str,
    delete_temp_csv: bool = True,
) -> None:
    csv_mask_path = Path(csv_mask_path)
    csv_mam_path = Path(csv_mam_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    label_map = {
        "calc": {"MALIGNANT": 1, "BENIGN": 2},
        "mass": {"MALIGNANT": 3, "BENIGN": 4},
    }

    for split in ("training", "test"):
        mam_frames = []
        mask_frames = []

        for lesion in ("calc", "mass"):
            df_mam = pd.read_csv(csv_mam_path / f"{lesion}-{split}.csv")
            df_mask = pd.read_csv(csv_mask_path / f"{lesion}-{split}.csv")

            df_mam["label"] = df_mam["pathology"].map(label_map[lesion])
            df_mask["label"] = df_mask["pathology"].map(label_map[lesion])

            mam_frames.append(df_mam)
            mask_frames.append(df_mask)

        df_mam = pd.concat(mam_frames, ignore_index=True).drop(columns="pathology")
        df_mask = pd.concat(mask_frames, ignore_index=True).drop(columns="pathology")

        df_mam.rename(columns={"img": "merge_key"}, inplace=True)
        df_mask["merge_key"] = df_mask["img"].str[:-2]

        final_df = (
            df_mam.merge(df_mask, on="merge_key", how="inner")
            .drop(columns=["img", "merge_key"])
        )

        final_df.to_csv(output_path / f"{split}_dataset.csv", index=False)

    if delete_temp_csv:
        shutil.rmtree(csv_mask_path, ignore_errors=True)
        shutil.rmtree(csv_mam_path, ignore_errors=True)
