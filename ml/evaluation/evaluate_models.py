import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"
SAVED_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "saved_models"


def load_dataset() -> Dict[str, np.ndarray]:
    X_path = PROCESSED_DIR / "X_sliding.csv"
    y_path = PROCESSED_DIR / "y_sliding.csv"
    if not X_path.exists() or not y_path.exists():
        raise SystemExit(
            "Sliding-window dataset not found. "
            "Run create_sliding_window_dataset.py and train_models.py first."
        )

    X = pd.read_csv(X_path).values.astype(float)
    y = pd.read_csv(y_path).iloc[:, 0].values.astype(float)
    return {"X": X, "y": y}


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def main() -> None:
    data = load_dataset()
    X, y = data["X"], data["y"]

    best_model_path = SAVED_MODELS_DIR / "best_model.joblib"
    if not best_model_path.exists():
        raise SystemExit(
            f"Best model not found at {best_model_path}. "
            "Run train_models.py first."
        )

    model = joblib.load(best_model_path)
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred)

    metrics_path = SAVED_MODELS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics: {metrics}")
    print(f"Saved metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

