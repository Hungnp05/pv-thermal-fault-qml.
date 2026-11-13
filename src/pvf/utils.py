import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Seed every random number generator used in training for reproducible runs.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Return the GPU if available, otherwise the CPU.
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Save a dictionary as pretty-printed JSON.
def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# Append one result row to a CSV file in append mode so parallel runs do not overwrite each other.
def append_result(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
