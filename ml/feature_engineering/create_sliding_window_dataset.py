import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.neutrosophic_encoder import encode_sequence


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"


def create_sliding_window(
    df: pd.DataFrame, window_size: int, min_semester: int = 1, max_semester: int = 8
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a sliding-window supervised dataset from semester-wise GPA.

    For each student:
    TBK1, TBK2, ..., TBK8

    With window_size = 3, we generate:
    [TBK1, TBK2, TBK3] -> TBK4
    [TBK2, TBK3, TBK4] -> TBK5
    ...
    """
    df = df.copy()
    tbk_cols = [f"TBK{i}" for i in range(min_semester, max_semester + 1)]

    missing_tbk = [c for c in tbk_cols if c not in df.columns]
    if missing_tbk:
        raise ValueError(f"Missing expected TBK columns: {missing_tbk}")

    features = []
    targets = []

    for _, row in df.iterrows():
        values = row[tbk_cols].values.astype(float)
        n_semesters = len(values)
        for start in range(0, n_semesters - window_size):
            end = start + window_size
            if end >= n_semesters:
                break
            window = values[start:end]
            target = values[end]
            if np.isnan(window).any() or np.isnan(target):
                continue
            features.append(window)
            targets.append(target)

    if not features:
        raise ValueError("Sliding window produced an empty dataset. Check input data.")

    X = pd.DataFrame(
        features,
        columns=[f"TBK_t{i+1}" for i in range(window_size)],
    )
    y = pd.Series(targets, name="TBK_next")
    return X, y


def main(
    input_name: str = "clean_semester_scores.csv",
    output_features: str = "X_sliding.csv",
    output_targets: str = "y_sliding.csv",
    output_neutro: str = "X_neutro.npy",
    window_size: int = 3,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    input_path = PROCESSED_DIR / input_name

    if not input_path.exists():
        raise SystemExit(
            f"Input file {input_path} not found. "
            f"Run preprocess_excel.py first to generate a cleaned dataset."
        )

    df = pd.read_csv(input_path)
    X, y = create_sliding_window(df, window_size=window_size)

    X_path = PROCESSED_DIR / output_features
    y_path = PROCESSED_DIR / output_targets
    neutro_path = PROCESSED_DIR / output_neutro

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    X_neutro = np.stack(
        [encode_sequence(window.tolist()) for window in X.values.astype(float)],
        axis=0,
    ).astype(np.float32)
    np.save(neutro_path, X_neutro)

    print(f"Saved sliding-window features to {X_path.resolve()} with shape {X.shape}")
    print(f"Saved sliding-window targets to {y_path.resolve()} with shape {y.shape}")
    print(
        f"Saved neutrosophic tensor to {neutro_path.resolve()} "
        f"with shape {X_neutro.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sliding-window supervised dataset from cleaned semester GPAs."
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="clean_semester_scores.csv",
        help="Name of the cleaned CSV file in data/processed_dataset.",
    )
    parser.add_argument(
        "--output-features",
        type=str,
        default="X_sliding.csv",
        help="Output CSV name for features.",
    )
    parser.add_argument(
        "--output-targets",
        type=str,
        default="y_sliding.csv",
        help="Output CSV name for targets.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sliding window size (number of previous semesters used as features).",
    )
    parser.add_argument(
        "--output-neutro",
        type=str,
        default="X_neutro.npy",
        help="Output NPY name for neutrosophic encoded feature tensor.",
    )

    args = parser.parse_args()
    main(
        input_name=args.input_name,
        output_features=args.output_features,
        output_targets=args.output_targets,
        output_neutro=args.output_neutro,
        window_size=args.window_size,
    )

