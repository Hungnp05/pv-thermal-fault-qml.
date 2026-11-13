import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pvf.data import PVFDataset, build_transform, class_names, fold_split, load_table, official_split
from pvf.models import build_cnn
from pvf.paths import OUTPUT_DIR
from pvf.utils import get_device


# Parse command line options for feature extraction.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache penultimate CNN features for one fold.")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--scheme", choices=["session", "date", "official", "ism"], default="session")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=["finetuned", "imagenet"], default="finetuned",
                        help="Use the fine-tuned checkpoint of this fold or the frozen ImageNet backbone.")
    parser.add_argument("--version", choices=["ori", "110x60", "112x112", "ism"], default="ori")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.scheme == "ism":
        args.version = "ism"
    return args


# Return the pooled feature vector before the classification layer for every image.
@torch.no_grad()
def extract(model, loader, device) -> np.ndarray:
    model.eval()
    features = []
    for images, _ in loader:
        output = model.forward_head(model.forward_features(images.to(device)), pre_logits=True)
        features.append(output.float().cpu().numpy())
    return np.concatenate(features)


# Extract features for train, validation and test splits and save them in one file.
def main() -> None:
    args = parse_args()
    device = get_device()
    table = load_table(args.scheme)
    names = class_names(table)
    parts = official_split(table, args.seed) if args.scheme == "official" else fold_split(table, args.fold, args.scheme)
    fold_name = "official" if args.scheme == "official" else f"fold{args.fold}"

    model = build_cnn(args.model, len(names), pretrained=args.source == "imagenet").to(device)
    if args.source == "finetuned":
        checkpoint = OUTPUT_DIR / "cnn" / args.scheme / args.model / f"{fold_name}_seed{args.seed}" / "best.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    arrays = {}
    for split, rows in parts.items():
        dataset = PVFDataset(rows, build_transform(args.image_size, train=False), args.version)
        arrays[f"X_{split}"] = extract(model, DataLoader(dataset, batch_size=args.batch_size), device)
        arrays[f"y_{split}"] = rows["label"].to_numpy()
        arrays[f"keys_{split}"] = rows["key"].to_numpy()

    out = OUTPUT_DIR / "features" / args.scheme / f"{args.model}_{args.source}" / f"{fold_name}_seed{args.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, class_names=np.array(names), **arrays)
    print(f"saved {out} with feature dim {arrays['X_train'].shape[1]}")


if __name__ == "__main__":
    main()
