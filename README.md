## NCKH Project – Next Semester GPA Prediction

This project is a complete research system for predicting the **next semester GPA (TBK)** of university students using machine learning.

### Project Structure

- `frontend/` – React + Vite web app  
  - Dashboard with model metrics and charts  
  - Prediction page with GPA input form  
  - Dataset upload page  
- `backend/` – FastAPI server  
  - Endpoints for prediction, dataset upload, and model metrics  
  - Loads the best trained model from the `ml/saved_models` folder  
- `ml/` – Machine learning pipeline  
  - `preprocessing/` – Excel loading, cleaning, and basic preprocessing  
  - `feature_engineering/` – Sliding-window dataset creation  
  - `training/` – Training scripts for RandomForest, XGBoost, and MLP regressors  
  - `evaluation/` – Evaluation scripts and metrics persistence  
  - `saved_models/` – Persisted best model (`best_model.joblib`) and metrics (`metrics.json`)  
- `data/`  
  - `raw_excel/` – Original Excel files with columns: `ID, TBK1, ..., TBK8`  
  - `processed_dataset/` – CSV/Parquet datasets after preprocessing and sliding window  
- `docs/` – Additional documentation and experiment notes

### Data Format

Raw Excel files must follow this schema:

- `ID` – Student identifier  
- `TBK1` → `TBK8` – Average score of semesters 1–8

The ML pipeline uses a **sliding window** transformation so that, for a given window size \(k\):

- **Features**: \[TBK\_t, TBK\_{t+1}, ..., TBK\_{t+k-1}\]  
- **Target**: TBK\_{t+k}

For example, with window size 3:

- Input: `TBK1 TBK2 TBK3` → Target: `TBK4`  
- Input: `TBK2 TBK3 TBK4` → Target: `TBK5`  
- Input: `TBK3 TBK4 TBK5` → Target: `TBK6`

### End-to-End Workflow

1. Place Excel files into `data/raw_excel/`.  
2. Run `ml/preprocessing/preprocess_excel.py` to load and clean the data.  
3. Run `ml/feature_engineering/create_sliding_window_dataset.py` to build the supervised dataset.  
4. Run `ml/training/train_models.py` to train RandomForest, XGBoost, and MLP regressors.  
5. Run `ml/evaluation/evaluate_models.py` to compute metrics (MAE, RMSE, R²) and save `metrics.json`.  
6. Start the FastAPI backend to serve predictions and metrics.  
7. Start the React + Vite frontend to interact with the system via browser.

### Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

