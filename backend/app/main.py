from pathlib import Path
import sys
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from ml.neutrosophic_encoder import (
    LINGUISTIC_SETS,
    encode_sequence,
    infer_risk_from_score,
)
from ml.training.train_models import build_models
from . import settings


ML_DIR = BASE_DIR / "ml"
SAVED_MODELS_DIR = ML_DIR / "saved_models"
DATA_RAW_DIR = BASE_DIR / "data" / "raw_excel"
PROCESSED_DIR = BASE_DIR / "data" / "processed_dataset"
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CYCLE = [4, 5, 6, 7, 8]


class NextSemesterRequest(BaseModel):
    # Backward compatibility input for TBK8 prediction.
    TBK5: float | None = Field(default=None, description="Average score of semester 5")
    TBK6: float | None = Field(default=None, description="Average score of semester 6")
    TBK7: float | None = Field(default=None, description="Average score of semester 7")
    # Flexible input: select target semester and provide its previous 3 TBKs.
    target_semester: int | None = Field(
        default=None, ge=4, le=8, description="Target semester to predict."
    )
    scores: Dict[str, float] | None = Field(
        default=None,
        description="Map of previous TBKs. Example for target 6: {TBK3, TBK4, TBK5}.",
    )


class PredictionResponse(BaseModel):
    predicted_TBK8: float
    confidence: float
    risk_label: str
    truths: Dict[str, float]


class MetricsResponse(BaseModel):
    target: int
    MAE: float
    RMSE: float
    R2: float


def load_best_model() -> Any:
    model_path = SAVED_MODELS_DIR / "best_model.pt"
    if not model_path.exists():
        raise RuntimeError(
            f"Best model not found at {model_path}. "
            "Run the ML pipeline (train_models.py and evaluate_models.py) first."
        )
    checkpoint = torch.load(model_path, map_location=inference_device)
    model_name = str(checkpoint["model_name"])
    input_dim = int(checkpoint["input_dim"])
    state_dict = checkpoint["state_dict"]

    loaded_model = build_models(input_dim=input_dim)[model_name]
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(inference_device)
    loaded_model.eval()
    return loaded_model


def load_neutro_tensor() -> np.ndarray:
    x_path = PROCESSED_DIR / "X_neutro.npy"
    if x_path.exists():
        return np.load(x_path).astype(np.float32)

    # Fallback: derive neutrosophic tensor from X_sliding.csv on-the-fly.
    x_sliding_path = PROCESSED_DIR / "X_sliding.csv"
    if not x_sliding_path.exists():
        raise RuntimeError(
            f"Neither {x_path} nor {x_sliding_path} exists. "
            "Run create_sliding_window_dataset.py first."
        )
    x_sliding = pd.read_csv(x_sliding_path).values.astype(np.float32)
    return np.stack(
        [encode_sequence(row.tolist()) for row in x_sliding],
        axis=0,
    ).astype(np.float32)


def load_sliding_targets() -> np.ndarray:
    y_path = PROCESSED_DIR / "y_sliding.csv"
    if not y_path.exists():
        raise RuntimeError(
            f"Targets file not found at {y_path}. "
            "Run create_sliding_window_dataset.py first."
        )
    return pd.read_csv(y_path).iloc[:, 0].values.astype(np.float32)


def infer_batch(x_neutro: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xb = torch.tensor(x_neutro, dtype=torch.float32, device=inference_device)
        pred = model(xb).squeeze(1).detach().cpu().numpy()
    return np.clip(pred, 0.0, 10.0)


def sample_target_labels(n_rows: int) -> np.ndarray:
    return np.asarray([TARGET_CYCLE[i % len(TARGET_CYCLE)] for i in range(n_rows)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    tbk_cols = [c for c in df.columns if c.startswith("TBK")]
    numeric = df[tbk_cols].apply(pd.to_numeric, errors="coerce")
    missing = {col: int(df[col].isna().sum()) for col in df.columns}

    summary = {}
    for col in tbk_cols:
        s = numeric[col]
        summary[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    corr = numeric.corr().fillna(0.0)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_by_column": missing,
        "summary_by_semester": summary,
        "correlation": {
            "labels": list(corr.columns),
            "matrix": corr.values.round(4).tolist(),
        },
    }


app = FastAPI(title="Next Semester GPA Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    global model
    model = load_best_model()


@app.post("/predict", response_model=PredictionResponse)
def predict_next_semester(request: NextSemesterRequest):
    try:
        # Flexible mode.
        if request.target_semester is not None and request.scores is not None:
            required = [
                f"TBK{request.target_semester - 3}",
                f"TBK{request.target_semester - 2}",
                f"TBK{request.target_semester - 1}",
            ]
            if any(k not in request.scores for k in required):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Missing required scores for TBK{request.target_semester}: "
                        + ", ".join(required)
                    ),
                )
            scores = [float(request.scores[k]) for k in required]
        else:
            # Legacy mode: TBK5, TBK6, TBK7 -> TBK8.
            if request.TBK5 is None or request.TBK6 is None or request.TBK7 is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Provide either (TBK5,TBK6,TBK7) or "
                        "(target_semester and scores mapping)."
                    ),
                )
            scores = [request.TBK5, request.TBK6, request.TBK7]

        neutro_matrix = encode_sequence(scores).astype(np.float32)
        with torch.no_grad():
            xb = torch.tensor(
                np.expand_dims(neutro_matrix, axis=0),
                dtype=torch.float32,
                device=inference_device,
            )
            pred = model(xb).squeeze(1).detach().cpu().numpy()[0]
        pred = float(np.clip(pred, 0.0, 10.0))
        risk_info = infer_risk_from_score(pred)
        return PredictionResponse(
            predicted_TBK8=pred,
            confidence=float(risk_info["confidence"]),
            risk_label=str(risk_info["risk_label"]),
            truths={k: float(v) for k, v in dict(risk_info["truths"]).items()},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an Excel file (.xlsx or .xls).",
        )

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_RAW_DIR / file.filename

    content = await file.read()
    with destination.open("wb") as f:
        f.write(content)

    try:
        uploaded_df = pd.read_excel(destination)
        preview = build_dataset_profile(uploaded_df)
    except Exception:  # noqa: BLE001
        preview = {"error": "Uploaded file cannot be profiled. Check schema/format."}

    return {
        "message": "File uploaded successfully",
        "filename": file.filename,
        "profile_preview": preview,
    }


@app.get("/model-metrics", response_model=MetricsResponse)
def get_model_metrics(target: int = 8):
    try:
        if target not in TARGET_CYCLE:
            raise HTTPException(status_code=400, detail="target must be in [4, 5, 6, 7, 8].")
        y_true = load_sliding_targets()
        y_pred = infer_batch(load_neutro_tensor())
        labels = sample_target_labels(len(y_true))
        mask = labels == target
        if not np.any(mask):
            raise RuntimeError(f"No samples found for target TBK{target}.")
        return {"target": target, **compute_metrics(y_true[mask], y_pred[mask])}
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predictions-sample")
def get_predictions_sample(target: int = 8, limit: int = 300):
    if target not in TARGET_CYCLE:
        raise HTTPException(status_code=400, detail="target must be in [4, 5, 6, 7, 8].")
    limit = max(10, min(limit, 1000))

    y_true = load_sliding_targets()
    y_pred = infer_batch(load_neutro_tensor())
    labels = sample_target_labels(len(y_true))
    idx = np.where(labels == target)[0][:limit]

    rows: List[Dict[str, Any]] = []
    for i in idx:
        pred = float(y_pred[i])
        actual = float(y_true[i])
        risk = infer_risk_from_score(pred)
        rows.append(
            {
                "actual": actual,
                "predicted": pred,
                "residual": float(pred - actual),
                "risk_label": str(risk["risk_label"]),
            }
        )
    return {"target": target, "data": rows}


@app.get("/risk-distribution")
def get_risk_distribution(target: int = 8):
    if target not in TARGET_CYCLE:
        raise HTTPException(status_code=400, detail="target must be in [4, 5, 6, 7, 8].")

    y_true = load_sliding_targets()
    y_pred = infer_batch(load_neutro_tensor())
    labels = sample_target_labels(len(y_true))
    idx = np.where(labels == target)[0]

    distribution = {k: 0 for k in LINGUISTIC_SETS.keys()}
    for i in idx:
        risk = infer_risk_from_score(float(y_pred[i]))
        distribution[str(risk["risk_label"])] += 1
    return {"target": target, "distribution": distribution}


@app.get("/dataset/profile")
def get_dataset_profile():
    clean_path = PROCESSED_DIR / "clean_semester_scores.csv"
    if not clean_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Clean dataset not found at {clean_path}. Run preprocessing first.",
        )
    df = pd.read_csv(clean_path)
    return build_dataset_profile(df)


@app.get("/risk-bands")
def get_risk_bands():
    return {
        label: {"a": p[0], "b": p[1], "c": p[2], "d": p[3]}
        for label, p in LINGUISTIC_SETS.items()
    }


@app.get("/")
def root():
    return {"message": "Next Semester GPA Prediction API is running."}

