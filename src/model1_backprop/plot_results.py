from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    base = Path(__file__).resolve().parent
    results_path = base / "results" / "results.json"
    history_path = base / "results" / "history.json"
    fig_dir = base / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists() or not history_path.exists():
        raise FileNotFoundError("Run backprop_numpy_wdbc.py first to generate results.json and history.json")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Learning Curves (Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curves_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves (Accuracy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "learning_curves_accuracy.png", dpi=200)
    plt.close()

    cm = np.array(results["metrics"]["test"]["confusion_matrix"], dtype=int)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
