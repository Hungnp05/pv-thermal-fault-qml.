import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from pvf.metrics import evaluate
from pvf.paths import OUTPUT_DIR
from pvf.quantum import VQCClassifier, fidelity_kernel
from pvf.utils import append_result, save_json, set_seed

TORCH_HEADS = {"linear", "mlp", "tanh", "vqc"}


# Parse command line options for one classifier head run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classical or quantum head on cached CNN features.")
    parser.add_argument("--head", choices=["linear", "mlp", "tanh", "svm", "vqc", "qkernel"], required=True)
    parser.add_argument("--features", default="resnet18_finetuned")
    parser.add_argument("--scheme", choices=["session", "date", "official", "ism"], default="session")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--feature-seed", type=int, default=0, help="Seed of the CNN run that produced the features.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of this head run.")
    parser.add_argument("--dim", type=int, default=4, help="PCA dimension, equal to the number of qubits.")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--topology", choices=["linear", "circular"], default="linear")
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-train", type=int, default=0, help="Cap on training samples for the quantum kernel.")
    return parser.parse_args()


# Load cached features and reduce them to dim components with scaler and PCA fitted on train only.
def load_reduced(args: argparse.Namespace) -> dict:
    fold_name = "official" if args.scheme == "official" else f"fold{args.fold}"
    path = OUTPUT_DIR / "features" / args.scheme / args.features / f"{fold_name}_seed{args.feature_seed}.npz"
    raw = np.load(path, allow_pickle=True)
    standard = StandardScaler().fit(raw["X_train"])
    pca = PCA(n_components=args.dim, random_state=0).fit(standard.transform(raw["X_train"]))
    reduce = lambda x: pca.transform(standard.transform(x))
    angle = MinMaxScaler(feature_range=(0.0, np.pi)).fit(reduce(raw["X_train"]))
    data = {"num_classes": len(raw["class_names"]), "fold_name": fold_name,
            "explained_variance": float(pca.explained_variance_ratio_.sum())}
    for split in ("train", "val", "test"):
        data[f"X_{split}"] = np.clip(angle.transform(reduce(raw[f"X_{split}"])), 0.0, np.pi).astype(np.float32)
        data[f"y_{split}"] = raw[f"y_{split}"]
    return data


# Build the requested torch head; all of them map dim inputs to class logits.
def build_torch_head(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.head == "linear":
        return nn.Linear(args.dim, num_classes)
    if args.head == "tanh":
        return nn.Sequential(nn.Linear(args.dim, args.dim), nn.Tanh(), nn.Linear(args.dim, num_classes))
    if args.head == "mlp":
        return nn.Sequential(nn.Linear(args.dim, args.hidden), nn.ReLU(), nn.Linear(args.hidden, num_classes))
    return VQCClassifier(args.dim, num_classes, args.depth, args.topology)


# Return class probabilities of a torch head for a feature matrix.
@torch.no_grad()
def torch_probs(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()


# Train a torch head with class-weighted cross-entropy and early stopping on validation macro-F1.
def fit_torch_head(args: argparse.Namespace, data: dict) -> tuple[nn.Module, int]:
    model = build_torch_head(args, data["num_classes"])
    counts = np.bincount(data["y_train"], minlength=data["num_classes"]).astype(float)
    weights = torch.tensor(len(data["y_train"]) / (data["num_classes"] * np.maximum(counts, 1.0)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_train, y_train = torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"]).long()
    best_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(y_train)).split(args.batch_size):
            loss = criterion(model(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_f1 = evaluate(torch_probs(model, data["X_val"]), data["y_val"], data["num_classes"])["macro_f1"]
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


# Fit an SVM on either an RBF kernel or a precomputed quantum fidelity kernel and return test probabilities.
def fit_svm(args: argparse.Namespace, data: dict) -> tuple[np.ndarray, np.ndarray, int]:
    x_train, y_train = data["X_train"], data["y_train"]
    if args.max_train and len(y_train) > args.max_train:
        keep = np.random.default_rng(args.seed).choice(len(y_train), args.max_train, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]
    svm = SVC(kernel="precomputed" if args.head == "qkernel" else "rbf", class_weight="balanced",
              probability=True, random_state=args.seed)
    if args.head == "qkernel":
        kernel = lambda a, b: fidelity_kernel(a, b, args.dim, args.depth, args.topology)
        svm.fit(kernel(x_train, x_train), y_train)
        val_probs = svm.predict_proba(kernel(data["X_val"], x_train))
        test_probs = svm.predict_proba(kernel(data["X_test"], x_train))
    else:
        svm.fit(x_train, y_train)
        val_probs, test_probs = svm.predict_proba(data["X_val"]), svm.predict_proba(data["X_test"])
    return align_probs(val_probs, svm.classes_, data), align_probs(test_probs, svm.classes_, data), len(svm.support_)


# Place SVM probability columns at their class index, filling classes absent from training with zero.
def align_probs(probs: np.ndarray, classes: np.ndarray, data: dict) -> np.ndarray:
    full = np.zeros((len(probs), data["num_classes"]))
    full[:, classes] = probs
    return full


# Train the chosen head, evaluate on the test fold and store metrics and a results row.
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_reduced(args)
    start = time.perf_counter()
    if args.head in TORCH_HEADS:
        model, best_epoch = fit_torch_head(args, data)
        train_seconds = time.perf_counter() - start
        val_probs = torch_probs(model, data["X_val"])
        start = time.perf_counter()
        test_probs = torch_probs(model, data["X_test"])
        infer_ms = 1000 * (time.perf_counter() - start) / len(data["y_test"])
        params, support = sum(p.numel() for p in model.parameters()), None
    else:
        val_probs, test_probs, support = fit_svm(args, data)
        train_seconds, best_epoch, params, infer_ms = time.perf_counter() - start, None, None, None

    quantum = args.head in {"vqc", "qkernel"}
    name = f"{args.head}_q{args.dim}_d{args.depth}_{args.topology}" if quantum else f"{args.head}_d{args.dim}"
    run_dir = OUTPUT_DIR / "heads" / args.scheme / args.features / name / f"{data['fold_name']}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "predictions.npz", val_probs=val_probs, val_labels=data["y_val"],
             test_probs=test_probs, test_labels=data["y_test"])
    metrics = evaluate(test_probs, data["y_test"], data["num_classes"])
    row = {"head": args.head, "name": name, "features": args.features, "scheme": args.scheme,
           "fold": data["fold_name"], "feature_seed": args.feature_seed, "seed": args.seed, "dim": args.dim,
           "qubits": args.dim if quantum else 0, "depth": args.depth if quantum else 0,
           "topology": args.topology if quantum else "", "explained_variance": data["explained_variance"],
           "params": params, "support_vectors": support, "best_epoch": best_epoch,
           "train_seconds": round(train_seconds, 2), "infer_ms_per_image": None if infer_ms is None else round(infer_ms, 4), **metrics}
    save_json({"args": vars(args), **row}, run_dir / "metrics.json")
    append_result(row, OUTPUT_DIR / "results" / "heads.csv")
    print(f"{name}: test accuracy {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f} "
          f"params {params} time {train_seconds:.1f}s")


if __name__ == "__main__":
    main()
