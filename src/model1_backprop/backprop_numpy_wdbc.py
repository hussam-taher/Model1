from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


@dataclass
class HyperParams:
    hidden_units: int = 32
    learning_rate: float = 0.05
    epochs: int = 300
    seed: int = 42
    test_size: float = 0.30
    val_fraction_of_tmp: float = 0.50  
    threshold: float = 0.50


def forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2


def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, hp: HyperParams) -> Dict:
    rng = np.random.default_rng(hp.seed)
    n_in = X_train.shape[1]

    W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, hp.hidden_units))
    b1 = np.zeros((1, hp.hidden_units), dtype=float)
    W2 = rng.normal(0.0, 0.01, size=(hp.hidden_units, 1))
    b2 = np.zeros((1, 1), dtype=float)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start = time.time()
    for epoch in range(1, hp.epochs + 1):
        Z1, A1, Z2, y_hat = forward(X_train, W1, b1, W2, b2)
        train_loss = binary_cross_entropy(y_train, y_hat)

        dZ2 = (y_hat - y_train)  # (n,1)
        dW2 = (A1.T @ dZ2) / X_train.shape[0]
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = (X_train.T @ dZ1) / X_train.shape[0]
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        W2 -= hp.learning_rate * dW2
        b2 -= hp.learning_rate * db2
        W1 -= hp.learning_rate * dW1
        b1 -= hp.learning_rate * db1

        y_pred_train = (y_hat >= hp.threshold).astype(int)
        train_acc = float(accuracy_score(y_train, y_pred_train))

        _, _, _, y_val_hat = forward(X_val, W1, b1, W2, b2)
        val_loss = binary_cross_entropy(y_val, y_val_hat)
        y_pred_val = (y_val_hat >= hp.threshold).astype(int)
        val_acc = float(accuracy_score(y_val, y_pred_val))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    elapsed = float(time.time() - start)
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return {"model": model, "history": history, "train_time_sec": elapsed}


def evaluate(X: np.ndarray, y: np.ndarray, model: Dict, threshold: float = 0.50) -> Dict:
    _, _, _, y_hat = forward(X, model["W1"], model["b1"], model["W2"], model["b2"])
    y_pred = (y_hat >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y, y_hat))
    except Exception:
        out["roc_auc"] = None

    return out


def main() -> None:
    hp = HyperParams()

    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int).reshape(-1, 1)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=hp.test_size,
        random_state=hp.seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=hp.val_fraction_of_tmp,
        random_state=hp.seed,
        stratify=y_tmp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    
    trained = train_model(X_train, y_train, X_val, y_val, hp)

    train_metrics = evaluate(X_train, y_train, trained["model"], threshold=hp.threshold)
    val_metrics = evaluate(X_val, y_val, trained["model"], threshold=hp.threshold)
    test_metrics = evaluate(X_test, y_test, trained["model"], threshold=hp.threshold)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.json"
    results_path = out_dir / "results.json"

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(trained["history"], f, indent=2)

    results = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic) (WDBC)",
            "source": "UCI Machine Learning Repository via scikit-learn loader",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "task": "Binary classification (malignant vs benign)",
        },
        "split": {
            "train_size": int(X_train.shape[0]),
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
            "stratified": True,
        },
        "preprocessing": {
            "scaling": "StandardScaler (fit on train; applied to val/test)",
            "cleaning": "Not required for the scikit-learn WDBC loader (data provided without missing values).",
        },
        "hyperparameters": {
            "hidden_units": hp.hidden_units,
            "learning_rate": hp.learning_rate,
            "epochs": hp.epochs,
            "threshold": hp.threshold,
            "seed": hp.seed,
        },
        "training_time_sec": trained["train_time_sec"],
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved:")
    print(f" - {history_path}")
    print(f" - {results_path}")
    print("Test metrics:", results["metrics"]["test"])


if __name__ == "__main__":
    main()
