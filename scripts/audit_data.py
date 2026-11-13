import argparse
from collections import defaultdict

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from pvf.metadata import scan_version
from pvf.paths import METADATA_CSV, OUTPUT_DIR, VERSION_DIRS


# Build one metadata table that links the same image across all three dataset versions.
def build_metadata() -> pd.DataFrame:
    tables = {name: pd.DataFrame(scan_version(path)) for name, path in VERSION_DIRS.items()}
    base = tables["ori"].rename(columns={"rel_path": "path_ori"})
    for name in ("110x60", "112x112"):
        other = tables[name][["key", "rel_path", "official_split"]].rename(
            columns={"rel_path": f"path_{name}", "official_split": f"official_split_{name}"})
        base = base.merge(other, on="key", how="outer", indicator=True)
        missing = base[base["_merge"] != "both"]
        if len(missing):
            raise RuntimeError(f"{len(missing)} images do not match between ori and {name}")
        base = base.drop(columns="_merge")
    labels = sorted(base["label_name"].unique())
    base["label"] = base["label_name"].map({name: index for index, name in enumerate(labels)})
    return base.sort_values("key").reset_index(drop=True)


# Add image size and a perceptual hash computed from the original crop.
def add_image_features(meta: pd.DataFrame) -> pd.DataFrame:
    widths, heights, hashes = [], [], []
    for rel_path in meta["path_ori"]:
        with Image.open(VERSION_DIRS["ori"] / rel_path) as image:
            widths.append(image.width)
            heights.append(image.height)
            hashes.append(str(imagehash.phash(image.convert("L"))))
    return meta.assign(width=widths, height=heights, phash=hashes)


# Find pairs of images from different frames whose hashes differ by at most max_distance bits.
def near_duplicate_pairs(meta: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in meta["phash"]])
    frames = meta["frame_id"].to_numpy()
    pairs = []
    for i in range(len(bits)):
        distance = (bits[i + 1:] != bits[i]).sum(axis=1)
        for offset in np.flatnonzero(distance <= max_distance):
            j = i + 1 + offset
            if frames[i] != frames[j]:
                pairs.append((meta.at[i, "key"], meta.at[j, "key"], int(distance[offset])))
    return pd.DataFrame(pairs, columns=["key_a", "key_b", "distance"])


# Write a plain text summary of class balance, grouping structure and split leakage.
def write_report(meta: pd.DataFrame, pairs: pd.DataFrame, path) -> str:
    lines = [f"Images: {len(meta)}", f"Source frames: {meta['frame_id'].nunique()}",
             f"Capture dates: {meta['date'].nunique()}", ""]
    counts = meta.pivot_table(index="label_name", columns="official_split", values="key", aggfunc="count")
    counts["frames"] = meta.groupby("label_name")["frame_id"].nunique()
    lines += ["Images per class:", counts.to_string(), ""]
    per_date = meta.groupby("date").agg(images=("key", "count"), frames=("frame_id", "nunique"),
                                        classes=("label_name", "nunique"))
    lines += ["Images per capture date:", per_date.to_string(), ""]
    train_frames = set(meta.loc[meta["official_split"] == "train", "frame_id"])
    test_frames = set(meta.loc[meta["official_split"] == "test", "frame_id"])
    test_images = meta[meta["official_split"] == "test"]
    leaked = test_images["frame_id"].isin(train_frames).sum()
    lines += ["Official split leakage:",
              f"  test frames also in train: {len(test_frames & train_frames)} / {len(test_frames)}",
              f"  test images from a train frame: {leaked} / {len(test_images)}", ""]
    mixed = meta.groupby("frame_id")["label_name"].nunique()
    lines += [f"Frames containing more than one class: {(mixed > 1).sum()} / {len(mixed)}", ""]
    lines += ["Near-duplicate pairs across different frames (by pHash distance):"]
    for distance, group in pairs.groupby("distance"):
        lines.append(f"  distance {distance}: {len(group)} pairs")
    lines.append(f"  identical pHash: {meta['phash'].duplicated(keep=False).sum()} images")
    same_split = (meta["official_split"] == meta["official_split_112x112"]).sum()
    lines.insert(3, f"Images with the same official split in ori and 112x112: {same_split} / {len(meta)}")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


# Run the full audit and save metadata, near-duplicate pairs and the report.
def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the PVF-10 dataset.")
    parser.add_argument("--max-distance", type=int, default=4)
    args = parser.parse_args()

    audit_dir = OUTPUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    meta = add_image_features(build_metadata())
    meta.to_csv(METADATA_CSV, index=False)
    pairs = near_duplicate_pairs(meta, args.max_distance)
    pairs.to_csv(audit_dir / "near_duplicates.csv", index=False)
    print(write_report(meta, pairs, audit_dir / "report.txt"))


if __name__ == "__main__":
    main()
