import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"
SAVED_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "saved_models"


def load_dataset(
    features_name: str = "X_sliding.csv", targets_name: str = "y_sliding.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    X_path = PROCESSED_DIR / features_name
    y_path = PROCESSED_DIR / targets_name

    if not X_path.exists() or not y_path.exists():
        raise SystemExit(
            f"Missing sliding-window dataset. Expected:\n"
            f"  {X_path}\n"
            f"  {y_path}\n"
            f"Run create_sliding_window_dataset.py first."
        )

    X = pd.read_csv(X_path).values.astype(float)
    y = pd.read_csv(y_path).iloc[:, 0].values.astype(float)
    return X, y


def build_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=random_state,
        ),
    }
    return models


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def main(
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = build_models(random_state=random_state)
    all_metrics: Dict[str, Dict[str, float]] = {}

    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        all_metrics[name] = metrics

        print(f"{name} metrics: {metrics}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name

        # Save each model separately for reproducibility
        model_path = SAVED_MODELS_DIR / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path.resolve()}")

    if best_model_name is None:
        raise RuntimeError("No best model selected. Check training pipeline.")

    # Save the best model under a standard name for the API
    best_model_path = SAVED_MODELS_DIR / "best_model.joblib"
    joblib.dump(models[best_model_name], best_model_path)
    print(
        f"Best model: {best_model_name} with R2={best_r2:.4f}. "
        f"Saved to {best_model_path.resolve()}"
    )

    # Persist metrics for later inspection (used by evaluation script and backend)
    metrics_path = SAVED_MODELS_DIR / "raw_metrics.json"
    import json

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved raw model metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RandomForest, XGBoost, and MLP models on sliding-window GPA data."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state)

