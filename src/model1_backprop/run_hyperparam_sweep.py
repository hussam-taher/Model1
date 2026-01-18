from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backprop_numpy_wdbc import HyperParams, train_model, evaluate


RUNS = [
    (16, 0.01),
    (16, 0.05),
    (32, 0.01),
    (32, 0.05),
    (64, 0.01),
    (64, 0.05),
]


def run_one(hidden_units: int, learning_rate: float, epochs: int = 300, seed: int = 42) -> dict:
    hp = HyperParams(hidden_units=hidden_units, learning_rate=learning_rate, epochs=epochs, seed=seed)

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

    return {
        "hyperparameters": {
            "hidden_units": hp.hidden_units,
            "learning_rate": hp.learning_rate,
            "epochs": hp.epochs,
            "seed": hp.seed,
            "threshold": hp.threshold,
        },
        "split": {
            "train_size": int(X_train.shape[0]),
            "val_size": int(X_val.shape[0]),
            "test_size": int(X_test.shape[0]),
            "stratified": True,
        },
        "training_time_sec": float(trained["train_time_sec"]),
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    out_dir = base / "results" / "hp_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for hu, lr in RUNS:
        print(f"Running: hidden_units={hu}, learning_rate={lr}")
        res = run_one(hidden_units=hu, learning_rate=lr)

        tag = f"hu{hu}_lr{str(lr).replace('.', '')}"
        out_path = out_dir / f"{tag}.json"
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

    print("Done. All runs saved under:", out_dir)


if __name__ == "__main__":
    main()
