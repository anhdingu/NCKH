from pathlib import Path
from typing import Any, Dict
import json

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import settings


BASE_DIR = Path(__file__).resolve().parents[2]
ML_DIR = BASE_DIR / "ml"
SAVED_MODELS_DIR = ML_DIR / "saved_models"
DATA_RAW_DIR = BASE_DIR / "data" / "raw_excel"


class NextSemesterRequest(BaseModel):
    TBK5: float = Field(..., description="Average score of semester 5")
    TBK6: float = Field(..., description="Average score of semester 6")
    TBK7: float = Field(..., description="Average score of semester 7")


class PredictionResponse(BaseModel):
    predicted_TBK8: float


class MetricsResponse(BaseModel):
    MAE: float
    RMSE: float
    R2: float = Field(..., alias="R²", description="Coefficient of determination")

    class Config:
        populate_by_name = True


def load_best_model():
    model_path = SAVED_MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Best model not found at {model_path}. "
            "Run the ML pipeline (train_models.py and evaluate_models.py) first."
        )
    return joblib.load(model_path)


def load_metrics() -> Dict[str, Any]:
    metrics_path = SAVED_MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"Metrics file not found at {metrics_path}. "
            "Run evaluate_models.py to generate evaluation metrics."
        )
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    """
    Predict TBK8 using TBK5, TBK6, TBK7.
    """
    try:
        # The training pipeline expects a sliding window of size 3,
        # so we mirror that here: [TBK5, TBK6, TBK7] -> TBK8
        features = [[request.TBK5, request.TBK6, request.TBK7]]
        pred = model.predict(features)[0]
        return PredictionResponse(predicted_TBK8=float(pred))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload an Excel file and save it to data/raw_excel.
    """
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

    return {"message": "File uploaded successfully", "filename": file.filename}


@app.get("/model-metrics")
def get_model_metrics():
    """
    Return MAE, RMSE, and R² of the best model.
    """
    try:
        metrics = load_metrics()
        # Expecting metrics as {"MAE": ..., "RMSE": ..., "R2": ...}
        return {
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R²": metrics.get("R2"),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root():
    return {"message": "Next Semester GPA Prediction API is running."}

