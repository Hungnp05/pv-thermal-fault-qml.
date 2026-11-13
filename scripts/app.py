from types import SimpleNamespace

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from predict import load_run
from pvf.data import LetterboxSquare, build_transform, fold_split, load_table
from pvf.gradcam import grad_cam
from pvf.paths import OUTPUT_DIR, VERSION_DIRS

DATASETS = {"session": "PVF-10 (session-grouped folds)", "ism": "InfraredSolarModules"}
SCHEMES = {"session": "PVF-10, session-grouped", "date": "PVF-10, date-grouped (unseen days)",
           "ism": "InfraredSolarModules"}
FIGURES = {
    "fig1_split_protocols": "Data leakage: ResNet18 under three evaluation protocols",
    "fig2_cnn_benchmark": "RQ1: CNN macro-F1 and ECE on session-grouped folds",
    "fig7_domain_gap": "RQ4 proxy: macro-F1 drop on unseen capture dates",
    "fig3_robustness": "RQ5: macro-F1 under five corruptions and five severities",
    "fig5_risk_coverage": "RQ5: selective prediction, risk against coverage",
    "fig6_confusion_resnet18": "Confusion matrix, ResNet18",
    "fig6_confusion_densenet121": "Confusion matrix, DenseNet121",
}
CLASSICAL, QUANTUM = "#2a78d6", "#eb6834"
DEVICE = torch.device("cpu")


# List the trained models of a dataset that have a checkpoint for every requested fold.
def available_models(scheme: str) -> list[str]:
    root = OUTPUT_DIR / "cnn" / scheme
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "fold0_seed0" / "best.pt").exists())


# Load one trained model with its class names, input size and validation-fitted temperature, once per session.
@st.cache_resource
def cached_run(scheme: str, model: str, fold: int):
    return load_run(SimpleNamespace(scheme=scheme, model=model, fold=fold, seed=0), DEVICE)


# Load the image table of one dataset, once per session.
@st.cache_data
def cached_table(scheme: str) -> pd.DataFrame:
    return load_table(scheme)


# Return calibrated class probabilities and a heatmap of the regions that drove the top prediction.
def diagnose(model, image: Image.Image, image_size: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    tensor = build_transform(image_size, train=False)(image)
    with torch.no_grad():
        probs = torch.softmax(model(tensor.unsqueeze(0)) / temperature, dim=1)[0].numpy()
    return probs, grad_cam(model, tensor, int(probs.argmax()))


# Blend the heatmap over the letterboxed input image.
def overlay(image: Image.Image, cam: np.ndarray, image_size: int) -> np.ndarray:
    base = np.asarray(LetterboxSquare()(image).resize((image_size, image_size)), dtype=np.float32) / 255
    heat = matplotlib.colormaps["inferno"](cam)[..., :3]
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)


# Pick a few random images from the test fold, which the selected model never saw during training.
def sample_test_images(scheme: str, fold: int, count: int, seed: int) -> list[tuple[str, Image.Image, str]]:
    version = "ism" if scheme == "ism" else "ori"
    test = fold_split(cached_table(scheme), fold, scheme)["test"].sample(count, random_state=seed)
    root = VERSION_DIRS[version]
    return [(row["key"], Image.open(root / row[f"path_{version}"]).convert("RGB"), row["label_name"])
            for _, row in test.iterrows()]


# Show the prediction, the heatmap and the top three classes for one image.
def show_result(title: str, image: Image.Image, truth: str | None, run, threshold: float) -> None:
    model, names, image_size, temperature = run
    probs, cam = diagnose(model, image, image_size, temperature)
    top = np.argsort(-probs)[:3]
    st.markdown(f"**{title}**")
    left, middle, right = st.columns([1, 1, 2])
    left.image(LetterboxSquare()(image).resize((image_size, image_size)), caption="Input", width="stretch")
    middle.image(overlay(image, cam, image_size), caption="HiResCAM", width="stretch")
    prediction, confidence = names[top[0]], float(probs[top[0]])
    if confidence < threshold:
        right.warning(f"Uncertain: {prediction.replace('_', ' ')} ({confidence:.1%}). Send to a technician.")
    else:
        right.success(f"{prediction.replace('_', ' ')} ({confidence:.1%})")
    if truth is not None:
        verdict = "correct" if truth == prediction else "wrong"
        right.caption(f"True class: {truth.replace('_', ' ')} ({verdict})")
    ranking = pd.DataFrame({"class": [names[i].replace("_", " ") for i in top], "probability": probs[top]})
    right.dataframe(ranking, hide_index=True, width="stretch", column_config={
        "probability": st.column_config.ProgressColumn("probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.divider()


# Render the diagnosis tab: choose a model, upload images or draw test images, and inspect predictions.
def tab_diagnosis() -> None:
    with st.sidebar:
        st.header("Model")
        scheme = st.selectbox("Dataset", [s for s in DATASETS if available_models(s)], format_func=DATASETS.get)
        model_name = st.selectbox("Backbone", available_models(scheme))
        folds = [f for f in range(5) if (OUTPUT_DIR / "cnn" / scheme / model_name / f"fold{f}_seed0" / "best.pt").exists()]
        fold = st.selectbox("Fold", folds)
        threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05,
                              help="Predictions below this confidence are flagged for manual review.")
    run = cached_run(scheme, model_name, fold)
    st.caption(f"Temperature fitted on validation fold: {run[3]:.3f}. Inference runs on CPU.")
    source = st.radio("Images", ["Random test images", "Upload"], horizontal=True)
    if source == "Upload":
        files = st.file_uploader("Thermal module images", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                                 accept_multiple_files=True)
        for file in files or []:
            show_result(file.name, Image.open(file).convert("RGB"), None, run, threshold)
    else:
        count = st.number_input("How many", 1, 12, 4)
        if "sample_seed" not in st.session_state:
            st.session_state.sample_seed = 0
        if st.button("Draw new images"):
            st.session_state.sample_seed += 1
        st.caption(f"Drawn from test fold {fold}, never seen by this model during training.")
        for key, image, truth in sample_test_images(scheme, fold, count, st.session_state.sample_seed):
            show_result(key, image, truth, run, threshold)


# Render the results tab with the CNN summary table and the saved paper figures.
def tab_results() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "results" / "cnn_summary.csv")
    options = [s for s in SCHEMES if s in set(summary["scheme"])]
    scheme = st.selectbox("Evaluation", options, format_func=SCHEMES.get, key="results_scheme")
    table = summary[summary["scheme"] == scheme].drop(columns="scheme").sort_values("macro_f1", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Mean +- standard deviation over folds. ECE, Brier, NLL and AURC: lower is better.")
    available = {k: v for k, v in FIGURES.items() if (OUTPUT_DIR / "figures" / f"{k}.png").exists()}
    choice = st.selectbox("Figure", list(available), format_func=available.get)
    st.image(str(OUTPUT_DIR / "figures" / f"{choice}.png"), width="stretch")


# Render the quantum versus classical tab from the head results on identical reduced features.
def tab_quantum() -> None:
    heads = pd.read_csv(OUTPUT_DIR / "results" / "heads.csv")
    heads = heads.drop_duplicates(["name", "features", "scheme", "fold", "seed"], keep="last")
    left, middle, right = st.columns(3)
    options = [s for s in SCHEMES if s in set(heads["scheme"])]
    scheme = left.selectbox("Evaluation", options, format_func=SCHEMES.get, key="quantum_scheme")
    features = middle.selectbox("Features", sorted(heads.loc[heads["scheme"] == scheme, "features"].unique()))
    dim = right.radio("Qubits / PCA dimension", [4, 8], index=1, horizontal=True)
    runs = heads[(heads["scheme"] == scheme) & (heads["features"] == features) & (heads["dim"] == dim)]
    table = runs.groupby("name").agg(family=("head", "first"), runs=("seed", "size"), macro_f1=("macro_f1", "mean"),
                                     f1_sd=("macro_f1", "std"), ece=("ece", "mean"), params=("params", "first"),
                                     train_seconds=("train_seconds", "mean")).reset_index()
    table["type"] = np.where(table["family"].isin(["vqc", "qkernel"]), "Quantum", "Classical")
    chart = alt.Chart(table).mark_point(size=120, filled=True).encode(
        x=alt.X("macro_f1:Q", title="Macro-F1 (higher is better)", scale=alt.Scale(zero=False)),
        y=alt.Y("ece:Q", title="ECE (lower is better)", scale=alt.Scale(zero=False)),
        color=alt.Color("type:N", scale=alt.Scale(domain=["Classical", "Quantum"], range=[CLASSICAL, QUANTUM])),
        shape=alt.Shape("type:N", scale=alt.Scale(domain=["Classical", "Quantum"], range=["circle", "square"])),
        tooltip=["name", "type", alt.Tooltip("macro_f1:Q", format=".3f"), alt.Tooltip("ece:Q", format=".3f"),
                 "params", alt.Tooltip("train_seconds:Q", format=".1f")],
    ).properties(height=380)
    st.altair_chart(chart, width="stretch")
    st.dataframe(table.drop(columns="family").sort_values("macro_f1", ascending=False).round(4),
                 hide_index=True, width="stretch")
    st.caption("All heads use the same PCA-reduced features and folds. Quantum training time is CPU simulator "
               "time and is not comparable with GPU time. The quantum kernel uses simulated statevectors.")


# Lay out the page with one tab per purpose.
def main() -> None:
    st.set_page_config(page_title="PV Thermal Fault Diagnosis", layout="wide")
    st.title("PV Thermal Fault Diagnosis")
    st.caption("Classical deep learning and hybrid quantum-classical models on PVF-10 and InfraredSolarModules")
    diagnosis, results, quantum = st.tabs(["Diagnosis", "Experiment results", "Quantum vs classical"])
    with diagnosis:
        tab_diagnosis()
    with results:
        tab_results()
    with quantum:
        tab_quantum()


if __name__ == "__main__":
    main()
