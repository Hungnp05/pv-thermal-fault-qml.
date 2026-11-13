from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "outputs"
METADATA_CSV = DATA_DIR / "metadata.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"
ISM_METADATA_CSV = DATA_DIR / "ism_metadata.csv"

VERSION_DIRS = {
    "ori": RAW_DIR / "PVF_10_Ori",
    "110x60": RAW_DIR / "PVF_10_110x60",
    "112x112": RAW_DIR / "PVF_10_112x112",
    "ism": RAW_DIR / "ism" / "InfraredSolarModules",
}
