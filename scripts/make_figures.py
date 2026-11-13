import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from pvf.paths import OUTPUT_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CNN_ORDER = ["coat_lite_small", "densenet121", "efficientnet_b0", "resnet50", "resnet18", "mobilenetv3_small"]
CNN_LABELS = {"coat_lite_small": "CoaT-Lite-S", "densenet121": "DenseNet121", "efficientnet_b0": "EfficientNet-B0",
              "resnet50": "ResNet50", "resnet18": "ResNet18", "mobilenetv3_small": "MobileNetV3-S"}
CORRUPTIONS = ["gaussian_noise", "blur", "contrast", "brightness", "resolution"]
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS = OUTPUT_DIR / "results"


# Apply one consistent, quiet style to every figure.
def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2,
    })


# Save a figure as both PNG and PDF and close it.
def save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# Plot ResNet18 macro-F1 under the three split protocols to show how leakage inflates results.
def figure_split_protocols() -> None:
    cnn = pd.read_csv(RESULTS / "cnn.csv")
    runs = cnn[cnn["model"] == "resnet18"]
    order = [("official", "Official split\n(leaky)"), ("session", "Session-grouped\n5-fold"),
             ("date", "Date-grouped\n5-fold")]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (scheme, label) in enumerate(order):
        values = runs.loc[runs["scheme"] == scheme, "macro_f1"].to_numpy()
        ax.bar(i, values.mean(), width=0.55, color=SERIES[0], zorder=2)
        ax.scatter(np.full(len(values), i), values, s=14, color=INK, zorder=3)
        ax.text(i + 0.31, values.mean(), f"{values.mean():.3f}", va="center", color=INK)
    ax.set_xticks(range(len(order)), [label for _, label in order])
    ax.set_xlim(-0.5, len(order) - 0.15)
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_title("ResNet18 by evaluation protocol", loc="left")
    save(fig, "fig1_split_protocols")


# Plot macro-F1 and ECE of every CNN on the session-grouped folds side by side.
def figure_cnn_benchmark() -> None:
    cnn = pd.read_csv(RESULTS / "cnn.csv")
    runs = cnn[cnn["scheme"] == "session"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, metric, title in ((axes[0], "macro_f1", "Macro-F1 (higher is better)"),
                              (axes[1], "ece", "ECE (lower is better)")):
        for i, model in enumerate(CNN_ORDER):
            values = runs.loc[runs["model"] == model, metric].to_numpy()
            ax.barh(i, values.mean(), height=0.55, color=SERIES[0], zorder=2)
            ax.scatter(values, np.full(len(values), i), s=12, color=INK, zorder=3)
            ax.text(values.max() * 1.02 + 0.002, i, f"{values.mean():.3f}", va="center", color=INK_MUTED)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(CNN_ORDER)), [CNN_LABELS[m] for m in CNN_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.5, 0.9)
    save(fig, "fig2_cnn_benchmark")


# Plot macro-F1 against corruption severity, one small panel per corruption, one line per CNN.
def figure_robustness() -> None:
    files = sorted(RESULTS.glob("reliability_session_*coat_lite_small*.csv"))
    data = pd.read_csv(files[-1])
    data = data[data["method"] == "softmax"]
    clean = data[data["corruption"] == "clean"].groupby("model")["macro_f1"].mean()
    fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(11.0, 2.9), sharey=True)
    for ax, corruption in zip(axes, CORRUPTIONS):
        subset = data[data["corruption"] == corruption]
        for i, model in enumerate(CNN_ORDER):
            curve = subset[subset["model"] == model].groupby("severity")["macro_f1"].mean()
            ax.plot([0, *curve.index], [clean[model], *curve.to_numpy()], color=SERIES[i],
                    marker=MARKERS[i], markersize=5, markeredgecolor=SURFACE, label=CNN_LABELS[model])
        ax.set_title(corruption.replace("_", " ").capitalize(), loc="left")
        ax.set_xticks(range(6))
        ax.set_xlabel("Severity (0 = clean)")
    axes[0].set_ylabel("Macro-F1")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "fig3_robustness")


# Summarize head runs as mean and standard deviation over folds and seeds.
def head_table(scheme: str, features: str) -> pd.DataFrame:
    heads = pd.read_csv(RESULTS / "heads.csv")
    runs = heads[(heads["scheme"] == scheme) & (heads["features"] == features)]
    return runs.groupby("name").agg(f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
                                    ece=("ece", "mean"), ece_sd=("ece", "std"), head=("head", "first"),
                                    dim=("dim", "first"))


# Plot macro-F1 and ECE of classical and quantum heads trained on the same reduced features.
def figure_heads(scheme: str = "session", features: str = "resnet18_finetuned") -> None:
    table = head_table(scheme, features)
    table = table[table["dim"] == 8].sort_values("f1")
    quantum = table["head"].isin(["vqc", "qkernel"])
    colors = np.where(quantum, SERIES[1], SERIES[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.32 * len(table) + 1.2), sharey=True)
    y = np.arange(len(table))
    for ax, metric, title in ((axes[0], "f1", "Macro-F1"), (axes[1], "ece", "ECE")):
        ax.errorbar(table[metric], y, xerr=table[f"{metric}_sd"], fmt="none", ecolor=INK_MUTED, elinewidth=1)
        ax.scatter(table[metric], y, c=colors, s=40, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_title(f"{title} (mean +- sd)", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, table.index)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=l)
               for c, l in ((SERIES[0], "Classical head"), (SERIES[1], "Quantum head"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, f"fig4_heads_{scheme}_{features}")


# Plot risk against coverage for ResNet18 on session-grouped and date-grouped folds.
def figure_risk_coverage() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (scheme, label) in enumerate((("session", "Session-grouped"), ("date", "Date-grouped"))):
        curves = []
        for path in sorted((OUTPUT_DIR / "cnn" / scheme / "resnet18").glob("*/predictions.npz")):
            data = np.load(path)
            logits = data["test_logits"]
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            order = np.argsort(-probs.max(axis=1))
            errors = (probs.argmax(axis=1) != data["test_labels"])[order]
            risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
            curves.append(np.interp(np.linspace(0.05, 1, 96), np.arange(1, len(errors) + 1) / len(errors), risk))
        ax.plot(np.linspace(0.05, 1, 96), np.mean(curves, axis=0), color=SERIES[i], label=label)
    ax.set_xlabel("Coverage (fraction of images answered)")
    ax.set_ylabel("Selective risk (error rate)")
    ax.set_title("ResNet18 risk-coverage, mean over folds", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig5_risk_coverage")


# Plot the row-normalized confusion matrix of one CNN pooled over all session-grouped folds.
def figure_confusion(model: str = "resnet18") -> None:
    labels, predictions = [], []
    run_dirs = sorted((OUTPUT_DIR / "cnn" / "session" / model).glob("*/"))
    names = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))["class_names"]
    for run_dir in run_dirs:
        data = np.load(run_dir / "predictions.npz")
        labels.append(data["test_labels"])
        predictions.append(data["test_logits"].argmax(axis=1))
    matrix = confusion_matrix(np.concatenate(labels), np.concatenate(predictions), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i, j] >= 0.05:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if matrix[i, j] > 0.5 else INK)
    pretty = [n.replace("_", " ") for n in names]
    ax.set_xticks(range(len(names)), pretty, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), pretty)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"{CNN_LABELS[model]}, row-normalized, all session folds", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, f"fig6_confusion_{model}")


# Plot each CNN's macro-F1 on session-grouped and date-grouped folds to show the domain gap.
def figure_domain_gap() -> None:
    cnn = pd.read_csv(RESULTS / "cnn.csv")
    means = cnn[cnn["scheme"].isin(["session", "date"])].groupby(["model", "scheme"])["macro_f1"].mean().unstack()
    models = [m for m in CNN_ORDER if m in means.index and not means.loc[m].isna().any()]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for i, model in enumerate(models):
        session, date = means.loc[model, "session"], means.loc[model, "date"]
        ax.plot([date, session], [i, i], color=GRID, linewidth=3, zorder=1)
        ax.text(session + 0.01, i, f"-{session - date:.3f}", va="center", color=INK_MUTED)
    for j, (scheme, label, marker) in enumerate((("session", "Session-grouped", "o"), ("date", "Date-grouped", "s"))):
        ax.scatter(means.loc[models, scheme], range(len(models)), color=SERIES[j], marker=marker, s=45,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
    ax.set_yticks(range(len(models)), [CNN_LABELS[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0.4, 0.88)
    ax.set_xlabel("Test macro-F1 (mean over folds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Domain gap: unseen capture dates", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig7_domain_gap")


# Build every figure that its input results allow.
def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper figures from saved results.")
    parser.add_argument("--head-features", nargs="+", default=["resnet18_finetuned"])
    args = parser.parse_args()
    set_style()
    figure_split_protocols()
    figure_cnn_benchmark()
    figure_robustness()
    for features in args.head_features:
        figure_heads("session", features)
    figure_heads("date", "resnet18_finetuned")
    figure_risk_coverage()
    figure_confusion("resnet18")
    figure_confusion("densenet121")
    figure_domain_gap()


if __name__ == "__main__":
    main()
