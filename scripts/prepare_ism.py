import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from pvf.paths import ISM_METADATA_CSV, OUTPUT_DIR, VERSION_DIRS


# Turn a class name such as "Hot-Spot-Multi" into "hot_spot_multi".
def canonical_ism_class(name: str) -> str:
    return re.sub(r"[-\s]+", "_", name.strip().lower())


# Read every image once and return a hash of its pixels, so exact duplicates share one value.
def pixel_hashes(paths: pd.Series) -> list[str]:
    root = VERSION_DIRS["ism"]
    return [hashlib.md5(np.asarray(Image.open(root / p)).tobytes()).hexdigest() for p in paths]


# Build the InfraredSolarModules table with stratified folds that keep exact duplicates together.
def main() -> None:
    parser = argparse.ArgumentParser(description="Index InfraredSolarModules and assign cross-validation folds.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = json.loads((VERSION_DIRS["ism"] / "module_metadata.json").read_text(encoding="utf-8"))
    table = pd.DataFrame([{"key": f"ism/{k}", "path_ism": v["image_filepath"],
                           "label_name": canonical_ism_class(v["anomaly_class"])} for k, v in meta.items()])
    table = table.sort_values("key", key=lambda s: s.str.split("/").str[1].astype(int)).reset_index(drop=True)
    names = sorted(table["label_name"].unique())
    table["label"] = table["label_name"].map(names.index)
    table["dup_group"] = pixel_hashes(table["path_ism"])

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    table["fold_ism"] = -1
    for fold, (_, test_index) in enumerate(splitter.split(table, table["label"], table["dup_group"])):
        table.loc[test_index, "fold_ism"] = fold
    table.to_csv(ISM_METADATA_CSV, index=False)

    sizes = table.groupby("dup_group")["key"].transform("size")
    conflicts = (table.groupby("dup_group")["label"].nunique() > 1).sum()
    counts = table.pivot_table(index="label_name", columns="fold_ism", values="key", aggfunc="count", fill_value=0)
    counts.loc["total"] = counts.sum()
    report = [f"Images: {len(table)}  classes: {len(names)}",
              f"Exact duplicate groups: {table.loc[sizes > 1, 'dup_group'].nunique()} "
              f"covering {int((sizes > 1).sum())} images, groups with conflicting labels: {conflicts}",
              "", counts.to_string()]
    out = OUTPUT_DIR / "audit" / "ism_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
