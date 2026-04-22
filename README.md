## NCKH Project - Next Semester GPA Prediction

This repository contains a complete research system to predict students' next semester GPA (`TBK`) using a **Neutrosophy-Driven Deep Learning** pipeline.

## Architecture

- `ml/`
  - `preprocessing/` - Excel merging/cleaning
  - `feature_engineering/` - sliding window + neutrosophic encoding
  - `training/` - RNN, LSTM, Transformer training (PyTorch)
  - `evaluation/` - MAE, RMSE, R2 evaluation
  - `saved_models/` - trained checkpoints and metrics
- `backend/` - FastAPI service for prediction and analytics
- `frontend/` - React + Vite dashboard/prediction/dataset analysis UI
- `data/`
  - `raw_excel/` - source Excel files
  - `processed_dataset/` - generated intermediate datasets

## Input Data Schema

Each Excel file must contain:

- `ID`
- `TBK1`, `TBK2`, `TBK3`, `TBK4`, `TBK5`, `TBK6`, `TBK7`, `TBK8`

## Data Pipeline

1. **Preprocessing**: merge multiple Excel files, validate schema, clean missing values.
2. **Sliding Window** (`window_size=3`):
   - `[TBK1, TBK2, TBK3] -> TBK4`
   - `[TBK2, TBK3, TBK4] -> TBK5`
   - `[TBK3, TBK4, TBK5] -> TBK6`
3. **Neutrosophic Encoding**:
   - 6 linguistic sets: Very Poor, Poor, Fair, Good, Very Good, Excellent
   - each scalar score is transformed into `(T, I, F)` memberships
   - model input tensor shape for `window=3`: `(3, 18)`
4. **Training**: compare `rnn`, `lstm`, `transformer`, save best checkpoint.
5. **Evaluation**: export `MAE`, `RMSE`, `R2`.

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2) Run ML pipeline

```bash
python .\ml\preprocessing\preprocess_excel.py
python .\ml\feature_engineering\create_sliding_window_dataset.py --window-size 3
python -m ml.training.train_models
python -m ml.evaluation.evaluate_models
```

### 3) Start backend

```bash
python -m backend.main
```

Backend:

- `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

### 4) Start frontend

```bash
cd frontend
npm run dev
```

Frontend:

- `http://localhost:5173` (or `5174/5175` if port is busy)

## Main API Endpoints

- `POST /predict`
  - accepts legacy input (`TBK5`, `TBK6`, `TBK7`) or flexible input (`target_semester`, `scores`)
  - returns `predicted_TBK8`, `confidence`, `risk_label`, `truths`
- `GET /model-metrics?target=4..8`
- `GET /predictions-sample?target=4..8&limit=...`
- `GET /risk-distribution?target=4..8`
- `GET /risk-bands`
- `GET /dataset/profile`
- `POST /upload-dataset`

## Notes

- Run preprocessing/feature/training/evaluation again whenever you upload new raw data and need updated predictions.
- Keep generated artifacts (`.pt`, processed CSV/NPY) out of Git via `.gitignore`.

