import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


# Compute the expected calibration error with equal-width confidence bins.
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


# Compute the area under the risk-coverage curve when rejecting the least confident predictions first.
def aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-probs.max(axis=1))
    errors = (probs.argmax(axis=1) != labels)[order]
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


# Compute classification, calibration and selective prediction metrics from class probabilities.
def evaluate(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(num_classes)[labels]
    eps = 1e-12
    result = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "ece": expected_calibration_error(probs, labels),
        "brier": float(((probs - one_hot) ** 2).sum(axis=1).mean()),
        "nll": float(-np.log(probs[np.arange(len(labels)), labels] + eps).mean()),
        "aurc": aurc(probs, labels),
    }
    per_class = recall_score(labels, predictions, labels=range(num_classes), average=None, zero_division=0)
    result.update({f"recall_class_{i}": float(r) for i, r in enumerate(per_class)})
    return {k: float(v) for k, v in result.items()}
