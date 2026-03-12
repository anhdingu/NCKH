import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_excel"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"


EXPECTED_COLUMNS = ["ID"] + [f"TBK{i}" for i in range(1, 9)]


def load_excel_files(paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate multiple Excel files with a consistent schema."""
    frames: List[pd.DataFrame] = []

    for p in paths:
        df = pd.read_excel(p)
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"File {p} is missing columns: {missing_cols}")

        df = df[EXPECTED_COLUMNS].copy()
        frames.append(df)

    if not frames:
        raise ValueError("No Excel files found to load.")

    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove duplicate IDs, handle missing values, and clip scores."""
    df = df.copy()

    # Drop full-duplicate rows
    df = df.drop_duplicates()

    # Drop rows where all TBK values are missing
    tbk_cols = [c for c in df.columns if c.startswith("TBK")]
    df = df.dropna(subset=tbk_cols, how="all")

    # Option 1: simple imputation – fill missing TBK with row mean of available TBKs
    row_means = df[tbk_cols].mean(axis=1)
    for c in tbk_cols:
        df[c] = df[c].fillna(row_means)

    # Clip scores into a reasonable GPA range (0–10)
    df[tbk_cols] = df[tbk_cols].clip(lower=0.0, upper=10.0)

    return df


def main(output_name: str = "clean_semester_scores.csv") -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    excel_files = sorted(RAW_DATA_DIR.glob("*.xlsx")) + sorted(
        RAW_DATA_DIR.glob("*.xls")
    )
    if not excel_files:
        raise SystemExit(
            f"No Excel files found in {RAW_DATA_DIR}. "
            f"Please place files with columns: {', '.join(EXPECTED_COLUMNS)}"
        )

    merged = load_excel_files(excel_files)
    cleaned = clean_data(merged)

    output_path = PROCESSED_DIR / output_name
    cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path.resolve()}")
    print(f"Shape: {cleaned.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw Excel GPA files into a cleaned CSV dataset."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="clean_semester_scores.csv",
        help="Name of the cleaned CSV file to write into data/processed_dataset.",
    )
    args = parser.parse_args()
    main(output_name=args.output_name)

