import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from pvf.paths import ISM_METADATA_CSV, METADATA_CSV, SPLITS_CSV, VERSION_DIRS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Load the image table for a split scheme: InfraredSolarModules for "ism", PVF-10 otherwise.
def load_table(scheme: str = "session") -> pd.DataFrame:
    if scheme == "ism":
        return pd.read_csv(ISM_METADATA_CSV)
    meta = pd.read_csv(METADATA_CSV, dtype={"date": str, "time": str})
    splits = pd.read_csv(SPLITS_CSV, dtype={"date": str})
    columns = ["key", "session_id", "group_id", "fold_session", "fold_date"]
    return meta.merge(splits[columns], on="key", validate="one_to_one")


# Return the sorted class names so that index i matches label i.
def class_names(table: pd.DataFrame) -> list[str]:
    return table.drop_duplicates("label").sort_values("label")["label_name"].tolist()


# Reproduce the authors' split: official test set, validation drawn at random from official train.
def official_split(table: pd.DataFrame, seed: int, val_fraction: float = 0.1) -> dict:
    train = table[table["official_split"] == "train"]
    val = train.groupby("label", group_keys=False).sample(frac=val_fraction, random_state=seed)
    return {
        "train": train.drop(val.index).reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": table[table["official_split"] == "test"].reset_index(drop=True),
    }


# Split the table into train, validation and test parts for one cross-validation fold.
def fold_split(table: pd.DataFrame, fold: int, scheme: str = "session", n_folds: int = 5) -> dict:
    column = f"fold_{scheme}"
    val_fold = (fold + 1) % n_folds
    return {
        "train": table[~table[column].isin([fold, val_fold])].reset_index(drop=True),
        "val": table[table[column] == val_fold].reset_index(drop=True),
        "test": table[table[column] == fold].reset_index(drop=True),
    }


class LetterboxSquare:
    # Pad an image with black borders so it becomes square without distorting the aspect ratio.
    def __call__(self, image: Image.Image) -> Image.Image:
        side = max(image.size)
        canvas = Image.new("RGB", (side, side))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return canvas


# Build the image preprocessing pipeline, with light augmentation for training.
def build_transform(image_size: int, train: bool) -> v2.Compose:
    steps = [LetterboxSquare(), v2.PILToTensor(), v2.Resize((image_size, image_size), antialias=True)]
    if train:
        steps += [v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


class PVFDataset(Dataset):
    # Keep all images of the given rows in memory because the dataset is small.
    def __init__(self, rows: pd.DataFrame, transform, version: str = "ori"):
        root = VERSION_DIRS[version]
        self.images = [Image.open(root / path).convert("RGB") for path in rows[f"path_{version}"]]
        self.labels = rows["label"].to_numpy()
        self.transform = transform

    # Return the number of images.
    def __len__(self) -> int:
        return len(self.labels)

    # Return one transformed image tensor and its integer label.
    def __getitem__(self, index: int):
        return self.transform(self.images[index]), int(self.labels[index])
