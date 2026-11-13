import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pvf.metrics import evaluate
from pvf.paths import OUTPUT_DIR


# Convert logits to probabilities after dividing by a temperature.
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# Find the temperature that minimizes negative log-likelihood on the validation logits.
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    nll = lambda t: -np.log(softmax(logits, t)[np.arange(len(labels)), labels] + 1e-12).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


# Compute accuracy on the most confident predictions at several coverage levels.
def selective_accuracy(probs: np.ndarray, labels: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.7)) -> dict:
    order = np.argsort(-probs.max(axis=1))
    correct = (probs.argmax(axis=1) == labels)[order]
    return {f"acc_at_cov_{int(c * 100)}": float(correct[: max(1, int(round(c * len(labels))))].mean())
            for c in coverages}


# Evaluate softmax and temperature-scaled probabilities for every saved CNN run.
def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature scaling and selective prediction for CNN runs.")
    parser.add_argument("--scheme", default="session")
    args = parser.parse_args()

    rows = []
    for path in sorted((OUTPUT_DIR / "cnn" / args.scheme).glob("*/*/predictions.npz")):
        data = np.load(path)
        num_classes = data["test_logits"].shape[1]
        temperature = fit_temperature(data["val_logits"], data["val_labels"])
        for method, t in (("softmax", 1.0), ("temperature", temperature)):
            probs = softmax(data["test_logits"], t)
            rows.append({"model": path.parent.parent.name, "run": path.parent.name, "method": method,
                         "temperature": t, **evaluate(probs, data["test_labels"], num_classes),
                         **selective_accuracy(probs, data["test_labels"])})
    results = pd.DataFrame(rows)
    out = OUTPUT_DIR / "results" / f"calibration_{args.scheme}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    columns = ["accuracy", "macro_f1", "ece", "nll", "brier", "aurc", "acc_at_cov_90", "acc_at_cov_80"]
    print(results.groupby(["model", "method"])[columns].mean().round(4).to_string())


if __name__ == "__main__":
    main()
