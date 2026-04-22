import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed_dataset"
SAVED_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "saved_models"


def load_dataset(
    neutro_name: str = "X_neutro.npy", targets_name: str = "y_sliding.csv"
) -> Tuple[np.ndarray, np.ndarray]:
    X_path = PROCESSED_DIR / neutro_name
    y_path = PROCESSED_DIR / targets_name

    if not X_path.exists() or not y_path.exists():
        raise SystemExit(
            f"Missing neutrosophic dataset. Expected:\n"
            f"  {X_path}\n"
            f"  {y_path}\n"
            f"Run create_sliding_window_dataset.py first."
        )

    X = np.load(X_path).astype(np.float32)
    y = pd.read_csv(y_path).iloc[:, 0].values.astype(np.float32)
    return X, y


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class DecoderHead(nn.Module):
    """Decoder/defuzzification block: latent state -> GPA in [0, 10]."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) * 10.0


class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.RNN(
            input_size=input_dim,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder = DecoderHead(96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.backbone(x)
        return self.decoder(seq_out[:, -1, :])


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.LSTM(
            input_size=input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder = DecoderHead(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.backbone(x)
        return self.decoder(seq_out[:, -1, :])


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.project = nn.Linear(input_dim, 96)
        layer = nn.TransformerEncoderLayer(
            d_model=96,
            nhead=8,
            dim_feedforward=192,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.decoder = DecoderHead(96)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.project(x)
        encoded = self.backbone(projected)
        pooled = encoded.mean(dim=1)
        return self.decoder(pooled)


def build_models(input_dim: int) -> Dict[str, nn.Module]:
    return {
        "rnn": RNNRegressor(input_dim),
        "lstm": LSTMRegressor(input_dim),
        "transformer": TransformerRegressor(input_dim),
    }


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float = 1e-3,
    patience: int = 12,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_array(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(xb).squeeze(1).detach().cpu().numpy()
    return np.clip(pred, 0.0, 10.0)


def main(
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 120,
    batch_size: int = 32,
) -> None:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(random_state)
    X, y = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=random_state
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    models = build_models(input_dim=X.shape[2])
    all_metrics: Dict[str, Dict[str, float]] = {}

    best_model_name = None
    best_r2 = -np.inf
    best_state = None

    for name, model in models.items():
        print(f"Training {name} ...")
        trained = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
        )
        y_pred = predict_array(trained, X_test, device=device)
        metrics = evaluate(y_test, y_pred)
        all_metrics[name] = metrics

        print(f"{name} metrics: {metrics}")

        if metrics["R2"] > best_r2:
            best_r2 = metrics["R2"]
            best_model_name = name
            best_state = {
                "model_name": name,
                "input_dim": int(X.shape[2]),
                "state_dict": {k: v.detach().cpu() for k, v in trained.state_dict().items()},
            }

        model_path = SAVED_MODELS_DIR / f"{name}_model.pt"
        torch.save(
            {
                "model_name": name,
                "input_dim": int(X.shape[2]),
                "state_dict": {k: v.detach().cpu() for k, v in trained.state_dict().items()},
            },
            model_path,
        )
        print(f"Saved {name} model to {model_path.resolve()}")

    if best_model_name is None or best_state is None:
        raise RuntimeError("No best model selected. Check training pipeline.")

    best_model_path = SAVED_MODELS_DIR / "best_model.pt"
    torch.save(best_state, best_model_path)
    print(
        f"Best model: {best_model_name} with R2={best_r2:.4f}. "
        f"Saved to {best_model_path.resolve()}"
    )

    metrics_path = SAVED_MODELS_DIR / "raw_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved raw model metrics to {metrics_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RNN/LSTM/Transformer models on neutrosophic GPA tensors."
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    args = parser.parse_args()
    main(
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

