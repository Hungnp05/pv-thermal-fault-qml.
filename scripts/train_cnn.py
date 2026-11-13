import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pvf.data import PVFDataset, build_transform, class_names, fold_split, load_table, official_split
from pvf.metrics import evaluate
from pvf.models import build_cnn, count_parameters, set_backbone_trainable
from pvf.paths import OUTPUT_DIR
from pvf.utils import append_result, get_device, save_json, set_seed


# Parse command line options for one training run.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one CNN on one PVF-10 fold.")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--scheme", choices=["session", "date", "official", "ism"], default="session")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--version", choices=["ori", "110x60", "112x112", "ism"], default="ori")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of each split, for quick smoke tests.")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.scheme == "ism":
        args.version = "ism"
    return args


# Build train, validation and test data loaders for the chosen split scheme.
def build_loaders(args: argparse.Namespace) -> tuple[dict, dict, list[str]]:
    table = load_table(args.scheme)
    names = class_names(table)
    if args.scheme == "official":
        parts = official_split(table, args.seed)
    else:
        parts = fold_split(table, args.fold, args.scheme)
    if args.subset < 1.0:
        parts = {k: v.sample(frac=args.subset, random_state=args.seed).reset_index(drop=True) for k, v in parts.items()}
    loaders = {}
    for split, rows in parts.items():
        dataset = PVFDataset(rows, build_transform(args.image_size, split == "train"), args.version)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train")
    return loaders, parts, names


# Weight each class by inverse frequency so rare faults count as much as common ones.
def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


# Run one pass over the training data and return the mean loss.
def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = criterion(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


# Return raw logits and labels for every image in a loader.
@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images.to(device))
        logits.append(output.float().cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


# Convert logits to probabilities with a numerically stable softmax.
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# Train with a frozen backbone first, then fine-tune everything with early stopping on validation macro-F1.
def fit(model, loaders, criterion, args, device, checkpoint) -> dict:
    num_classes = criterion.weight.numel()
    set_backbone_trainable(model, False)
    head_optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=args.head_lr)
    for _ in range(args.head_epochs):
        train_epoch(model, loaders["train"], criterion, head_optimizer, device)

    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = {"macro_f1": -1.0, "epoch": -1}
    for epoch in range(args.epochs):
        loss = train_epoch(model, loaders["train"], criterion, optimizer, device)
        scheduler.step()
        logits, labels = predict_logits(model, loaders["val"], device)
        val = evaluate(softmax(logits), labels, num_classes)
        print(f"epoch {epoch:02d} loss {loss:.4f} val_acc {val['accuracy']:.4f} val_f1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "epoch": epoch}
            torch.save(model.state_dict(), checkpoint)
        elif epoch - best["epoch"] >= args.patience:
            break
    return best


# Train, evaluate on the held-out test fold and save predictions, metrics and a results row.
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    loaders, parts, names = build_loaders(args)
    fold_name = "official" if args.scheme == "official" else f"fold{args.fold}"
    run_dir = OUTPUT_DIR / "cnn" / args.scheme / f"{args.model}{args.tag}" / f"{fold_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_cnn(args.model, len(names)).to(device)
    weights = class_weights(parts["train"]["label"].to_numpy(), len(names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    start = time.perf_counter()
    best = fit(model, loaders, criterion, args, device, run_dir / "best.pt")
    train_seconds = time.perf_counter() - start

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    val_logits, val_labels = predict_logits(model, loaders["val"], device)
    start = time.perf_counter()
    test_logits, test_labels = predict_logits(model, loaders["test"], device)
    infer_ms = 1000 * (time.perf_counter() - start) / len(test_labels)
    np.savez(run_dir / "predictions.npz", val_logits=val_logits, val_labels=val_labels,
             test_logits=test_logits, test_labels=test_labels, test_keys=parts["test"]["key"].to_numpy())

    metrics = evaluate(softmax(test_logits), test_labels, len(names))
    row = {"model": args.model + args.tag, "scheme": args.scheme, "fold": fold_name, "seed": args.seed,
           "version": args.version, "image_size": args.image_size, "best_epoch": best["epoch"],
           "val_macro_f1": best["macro_f1"], "train_seconds": round(train_seconds, 1),
           "infer_ms_per_image": round(infer_ms, 3), **count_parameters(model), **metrics}
    save_json({"args": vars(args), "class_names": names, **row}, run_dir / "metrics.json")
    if args.subset == 1.0:
        append_result(row, OUTPUT_DIR / "results" / "cnn.csv")
    print(f"test accuracy {metrics['accuracy']:.4f} macro_f1 {metrics['macro_f1']:.4f} ece {metrics['ece']:.4f}")


if __name__ == "__main__":
    main()
