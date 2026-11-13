import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from pvf.data import IMAGENET_MEAN, IMAGENET_STD, LetterboxSquare, PVFDataset, fold_split, load_table
from pvf.metrics import evaluate
from pvf.models import DROP_RATE, build_cnn
from pvf.paths import OUTPUT_DIR
from pvf.utils import get_device, set_seed

SEVERITIES = {
    "gaussian_noise": [0.02, 0.04, 0.06, 0.08, 0.10],
    "blur": [0.3, 0.6, 0.9, 1.2, 1.5],
    "contrast": [0.8, 0.65, 0.5, 0.4, 0.3],
    "brightness": [0.05, 0.10, 0.15, 0.20, 0.25],
    "resolution": [0.8, 0.6, 0.5, 0.4, 0.3],
}


# Apply one corruption at one strength to an image tensor with values in [0, 1].
def corrupt(image: torch.Tensor, kind: str, level: float) -> torch.Tensor:
    if kind == "gaussian_noise":
        image = image + torch.randn_like(image) * level
    elif kind == "blur":
        size = 2 * int(np.ceil(3 * level)) + 1
        image = v2.functional.gaussian_blur(image, [size, size], [level, level])
    elif kind == "contrast":
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * level + mean
    elif kind == "brightness":
        image = image + level
    elif kind == "resolution":
        height, width = image.shape[-2:]
        small = [max(1, round(height * level)), max(1, round(width * level))]
        image = v2.functional.resize(v2.functional.resize(image, small, antialias=True), [height, width])
    return image.clamp(0.0, 1.0)


# Build the evaluation transform with an optional corruption applied at the original image resolution.
def corrupted_transform(image_size: int, kind: str | None, level: float) -> v2.Compose:
    steps = [LetterboxSquare(), v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)]
    if kind is not None:
        steps.append(v2.Lambda(lambda image: corrupt(image, kind, level)))
    steps += [v2.Resize((image_size, image_size), antialias=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


# Return pooled features and labels for every image, computed once and reused for all methods.
@torch.no_grad()
def extract(model, dataset, device, batch_size: int = 128) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    features, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=batch_size):
        features.append(model.forward_head(model.forward_features(images.to(device)), pre_logits=True).float())
        labels.append(targets.numpy())
    return torch.cat(features), np.concatenate(labels)


# Average class probabilities over several passes with dropout kept active before the classifier.
@torch.no_grad()
def mc_dropout_probs(classifier, features: torch.Tensor, drop_rate: float, passes: int) -> np.ndarray:
    total = torch.zeros(len(features), classifier.out_features, device=features.device)
    for _ in range(passes):
        total += torch.softmax(classifier(F.dropout(features, p=drop_rate, training=True)), dim=1)
    return (total / passes).cpu().numpy()


# Fit a temperature on validation logits by minimizing negative log-likelihood.
def fit_temperature(logits: torch.Tensor, labels: np.ndarray) -> float:
    logits = logits.detach()
    target = torch.from_numpy(labels).to(logits.device)
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp())


# Evaluate softmax, temperature scaling and MC dropout on clean and corrupted test images for one CNN run.
def evaluate_run(run_dir, device, args) -> list[dict]:
    info = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_args, names = info["args"], info["class_names"]
    parts = fold_split(load_table(run_args["scheme"]), run_args["fold"], run_args["scheme"])
    model = build_cnn(run_args["model"], len(names), pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    classifier = model.get_classifier()

    clean = corrupted_transform(run_args["image_size"], None, 0.0)
    val_features, val_labels = extract(model, PVFDataset(parts["val"], clean, run_args["version"]), device)
    temperature = fit_temperature(classifier(val_features), val_labels)
    test_set = PVFDataset(parts["test"], clean, run_args["version"])

    conditions = [("clean", 0, None, 0.0)]
    conditions += [(kind, i + 1, kind, level) for kind, levels in SEVERITIES.items() for i, level in enumerate(levels)]
    rows = []
    for name, severity, kind, level in conditions:
        set_seed(0)
        test_set.transform = corrupted_transform(run_args["image_size"], kind, level)
        features, labels = extract(model, test_set, device)
        with torch.no_grad():
            logits = classifier(features)
        methods = {
            "softmax": torch.softmax(logits, dim=1).cpu().numpy(),
            "temperature": torch.softmax(logits / temperature, dim=1).cpu().numpy(),
            "mc_dropout": mc_dropout_probs(classifier, features, DROP_RATE, args.passes),
        }
        for method, probs in methods.items():
            rows.append({"model": run_args["model"], "scheme": run_args["scheme"], "fold": run_args["fold"],
                         "seed": run_args["seed"], "corruption": name, "severity": severity, "method": method,
                         "temperature": temperature, **evaluate(probs, labels, len(names))})
    return rows


# Run the reliability and robustness evaluation for every finished CNN run of the chosen models.
def main() -> None:
    parser = argparse.ArgumentParser(description="MC dropout, temperature scaling and corruption robustness.")
    parser.add_argument("--scheme", default="session")
    parser.add_argument("--models", nargs="+", default=["resnet18"])
    parser.add_argument("--passes", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    rows = []
    for model in args.models:
        for run_dir in sorted((OUTPUT_DIR / "cnn" / args.scheme / model).glob("*/")):
            if (run_dir / "metrics.json").exists():
                print(f"evaluating {run_dir}", flush=True)
                rows += evaluate_run(run_dir, device, args)
    results = pd.DataFrame(rows)
    out = OUTPUT_DIR / "results" / f"reliability_{args.scheme}_{'_'.join(args.models)}.csv"
    results.to_csv(out, index=False)

    summary = results.groupby(["corruption", "severity", "method"])[["accuracy", "macro_f1", "ece", "aurc"]].mean()
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
