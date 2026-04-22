import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from ml.training.train_models import build_models


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"
SAVED_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "saved_models"


def load_dataset() -> Dict[str, np.ndarray]:
    X_path = PROCESSED_DIR / "X_neutro.npy"
    y_path = PROCESSED_DIR / "y_sliding.csv"
    if not X_path.exists() or not y_path.exists():
        raise SystemExit(
            "Neutrosophic dataset not found. "
            "Run create_sliding_window_dataset.py and train_models.py first."
        )

    X = np.load(X_path).astype(np.float32)
    y = pd.read_csv(y_path).iloc[:, 0].values.astype(np.float32)
    return {"X": X, "y": y}


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def main() -> None:
    data = load_dataset()
    X, y = data["X"], data["y"]

    best_model_path = SAVED_MODELS_DIR / "best_model.pt"
    if not best_model_path.exists():
        raise SystemExit(
            f"Best model not found at {best_model_path}. "
            "Run train_models.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(best_model_path, map_location=device)
    model_name = str(checkpoint["model_name"])
    input_dim = int(checkpoint["input_dim"])
    state_dict = checkpoint["state_dict"]

    model = build_models(input_dim=input_dim)[model_name]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        y_pred = model(xb).squeeze(1).detach().cpu().numpy()
    y_pred = np.clip(y_pred, 0.0, 10.0)
    metrics = compute_metrics(y, y_pred)

    metrics_path = SAVED_MODELS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation metrics: {metrics}")
    print(f"Saved metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    main()

