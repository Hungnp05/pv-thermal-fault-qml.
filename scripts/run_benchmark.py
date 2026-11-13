import argparse
import itertools
import subprocess
import sys

from pvf.paths import OUTPUT_DIR, ROOT

QUANTUM_GRID = [(4, 1, "linear"), (4, 2, "linear"), (4, 3, "circular"),
                (8, 1, "linear"), (8, 2, "linear"), (8, 3, "circular")]


# Run one script with the venv Python and stop the batch if it fails.
def run(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *arguments]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


# Train every requested CNN on every fold and seed, skipping runs that already have metrics.
def run_cnns(args: argparse.Namespace) -> None:
    for model, fold, seed in itertools.product(args.models, args.folds, args.seeds):
        run_dir = OUTPUT_DIR / "cnn" / args.scheme / model / f"fold{fold}_seed{seed}"
        if (run_dir / "metrics.json").exists():
            continue
        run("train_cnn.py", ["--model", model, "--scheme", args.scheme, "--fold", str(fold), "--seed", str(seed)])


# Extract features and train every classical control and quantum configuration on each fold.
def run_heads(args: argparse.Namespace) -> None:
    features = f"{args.feature_model}_finetuned"
    for fold, seed in itertools.product(args.folds, args.seeds):
        feature_file = OUTPUT_DIR / "features" / args.scheme / features / f"fold{fold}_seed0.npz"
        if not feature_file.exists():
            run("extract_features.py", ["--model", args.feature_model, "--scheme", args.scheme, "--fold", str(fold)])
        common = ["--features", features, "--scheme", args.scheme, "--fold", str(fold), "--seed", str(seed)]
        jobs = [[head, dim, 1, "linear"] for head in ("linear", "mlp", "tanh", "svm") for dim in (4, 8)]
        jobs += [["vqc", dim, depth, topology] for dim, depth, topology in QUANTUM_GRID]
        jobs += [["qkernel", dim, depth, topology] for dim, depth, topology in QUANTUM_GRID]
        for head, dim, depth, topology in jobs:
            quantum = head in {"vqc", "qkernel"}
            name = f"{head}_q{dim}_d{depth}_{topology}" if quantum else f"{head}_d{dim}"
            run_dir = OUTPUT_DIR / "heads" / args.scheme / features / name / f"fold{fold}_seed{seed}"
            if (run_dir / "metrics.json").exists():
                continue
            run("train_head.py", common + ["--head", head, "--dim", str(dim), "--depth", str(depth),
                                           "--topology", topology])


# Parse options and run the requested stage of the benchmark.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of experiments, resuming where it stopped.")
    parser.add_argument("stage", choices=["cnn", "heads"])
    parser.add_argument("--scheme", choices=["session", "date", "ism"], default="session")
    parser.add_argument("--models", nargs="+", default=["mobilenetv3_small", "resnet18", "efficientnet_b0",
                                                        "densenet121", "resnet50", "coat_lite_small"])
    parser.add_argument("--feature-model", default="resnet18")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run_cnns(args) if args.stage == "cnn" else run_heads(args)


if __name__ == "__main__":
    main()
