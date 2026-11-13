import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pvf.data import build_transform
from pvf.metadata import canonical_class
from pvf.metrics import evaluate
from pvf.models import build_cnn
from pvf.paths import OUTPUT_DIR
from pvf.utils import get_device, save_json

IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


# Parse command line options for predicting fault classes of new images.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PV fault classes for thermal images with a trained CNN.")
    parser.add_argument("inputs", nargs="+", help="Image files or folders (searched recursively).")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--scheme", default="session")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.7, help="Flag predictions below this confidence.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV file for the predictions.")
    parser.add_argument("--labels-from-folders", action="store_true",
                        help="Take the true class from each image's parent folder name and report metrics.")
    return parser.parse_args()


# Map each image's parent folder name to a class index, stopping if a folder is not a known class.
def folder_labels(paths: list[Path], names: list[str]) -> np.ndarray:
    folders = [canonical_class(p.parent.name) for p in paths]
    unknown = sorted(set(folders) - set(names))
    if unknown:
        raise SystemExit(f"Unknown class folders {unknown}. Valid names: {names}")
    return np.array([names.index(f) for f in folders])


# Print overall, per-class and selective metrics plus a confusion matrix, and save them next to the CSV.
def report_metrics(table: pd.DataFrame, probs: np.ndarray, labels: np.ndarray, names: list[str], out) -> None:
    metrics = evaluate(probs, labels, len(names))
    print("\nOverall:")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll"):
        print(f"  {key:18s} {metrics[key]:.4f}")
    confident = ~table["uncertain"].to_numpy()
    correct = table["prediction"].to_numpy() == np.array(names)[labels]
    accuracy_kept = correct[confident].mean() if confident.any() else float("nan")
    print(f"  coverage           {confident.mean():.4f}  (accuracy on confident images {accuracy_kept:.4f})")
    present = np.unique(labels)
    print("\nPer-class recall (classes present in the folders):")
    for i in present:
        print(f"  {names[i]:24s} {metrics[f'recall_class_{i}']:.4f}  (n={int((labels == i).sum())})")
    confusion = pd.crosstab(pd.Series(np.array(names)[labels], name="true"),
                            pd.Series(table["prediction"].to_numpy(), name="predicted"))
    print("\nConfusion matrix (counts):")
    print(confusion.to_string())
    if out:
        confusion.to_csv(out.with_name(f"{out.stem}_confusion.csv"))
        save_json(metrics, out.with_name(f"{out.stem}_metrics.json"))


# Collect every image path from the given files and folders.
def collect_images(inputs: list[str]) -> list[Path]:
    paths = []
    for item in map(Path, inputs):
        candidates = sorted(item.rglob("*")) if item.is_dir() else [item]
        paths += [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return paths


# Find the temperature that minimizes validation negative log-likelihood by a simple grid search.
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.25), np.log(8.0), 200)):
        scaled = logits / t
        log_probs = scaled - scaled.max(axis=1, keepdims=True)
        log_probs -= np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
        nll = -log_probs[np.arange(len(labels)), labels].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


# Load a trained CNN run, its class names and a temperature fitted on its validation logits.
def load_run(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, list[str], int, float]:
    run_dir = OUTPUT_DIR / "cnn" / args.scheme / args.model / f"fold{args.fold}_seed{args.seed}"
    info = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    names = info["class_names"]
    model = build_cnn(args.model, len(names), pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    model.eval()
    saved = np.load(run_dir / "predictions.npz")
    temperature = fit_temperature(saved["val_logits"], saved["val_labels"])
    return model, names, info["args"]["image_size"], temperature


# Return calibrated class probabilities for a list of image paths.
@torch.no_grad()
def predict(model, paths: list[Path], image_size: int, temperature: float, device, batch_size: int) -> np.ndarray:
    transform = build_transform(image_size, train=False)
    probs = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
        probs.append(torch.softmax(model(batch.to(device)).float() / temperature, dim=1).cpu().numpy())
    return np.concatenate(probs)


# Predict every input image, print a readable table and optionally save it as CSV.
def main() -> None:
    args = parse_args()
    paths = collect_images(args.inputs)
    if not paths:
        raise SystemExit("No images found.")
    device = get_device()
    model, names, image_size, temperature = load_run(args, device)
    probs = predict(model, paths, image_size, temperature, device, args.batch_size)
    top2 = np.argsort(-probs, axis=1)[:, :2]
    table = pd.DataFrame({
        "image": [str(p) for p in paths],
        "prediction": [names[i] for i in top2[:, 0]],
        "confidence": probs[np.arange(len(paths)), top2[:, 0]].round(4),
        "second_choice": [names[i] for i in top2[:, 1]],
        "second_confidence": probs[np.arange(len(paths)), top2[:, 1]].round(4),
    })
    table["uncertain"] = table["confidence"] < args.threshold
    labels = folder_labels(paths, names) if args.labels_from_folders else None
    if labels is not None:
        table.insert(1, "true", [names[i] for i in labels])
    print(f"model {args.model} fold {args.fold} seed {args.seed}, temperature {temperature:.3f}, "
          f"threshold {args.threshold}")
    print(table.to_string(index=False))
    print(f"\n{len(table)} images, {int(table['uncertain'].sum())} flagged as uncertain")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"saved {args.out}")
    if labels is not None:
        report_metrics(table, probs, labels, names, args.out)


if __name__ == "__main__":
    main()
