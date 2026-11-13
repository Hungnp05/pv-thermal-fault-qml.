import re
from pathlib import Path

FILENAME_PATTERN = re.compile(r"^(DJI_(\d{8})(\d{6})_(\d+))_T_(\d+)$")
CLASS_PREFIX_PATTERN = re.compile(r"^\d+\s*")


# Convert a class folder name like "05hot cell" into a canonical name like "hot_cell".
def canonical_class(folder_name: str) -> str:
    name = CLASS_PREFIX_PATTERN.sub("", folder_name).strip().lower()
    return re.sub(r"\s+", "_", name)


# Split a drone crop filename into frame id, capture date, capture time and crop index.
def parse_filename(path: Path) -> dict:
    match = FILENAME_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    frame_id, date, time, _, crop = match.groups()
    return {"frame_id": frame_id, "date": date, "time": time, "crop_index": int(crop)}


# List every image of one dataset version with its class, official split and filename fields.
def scan_version(version_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(version_dir.glob("*/*/*")):
        if not path.is_file():
            continue
        split, class_folder = path.parent.parent.name, path.parent.name
        row = {
            "key": f"{canonical_class(class_folder)}/{path.stem}",
            "rel_path": path.relative_to(version_dir).as_posix(),
            "label_name": canonical_class(class_folder),
            "official_split": split,
        }
        row.update(parse_filename(path))
        rows.append(row)
    return rows
