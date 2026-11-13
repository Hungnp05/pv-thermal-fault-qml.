import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pvf.paths import METADATA_CSV, OUTPUT_DIR, SPLITS_CSV


# Assign a flight session id by cutting the capture timeline wherever the gap exceeds gap_seconds.
def session_ids(meta: pd.DataFrame, gap_seconds: int) -> pd.Series:
    times = pd.to_datetime(meta["date"] + meta["time"], format="%Y%m%d%H%M%S")
    unique_times = times.drop_duplicates().sort_values()
    new_session = unique_times.diff().dt.total_seconds().fillna(np.inf) > gap_seconds
    session_of_time = pd.Series(new_session.cumsum().to_numpy(), index=unique_times.to_numpy())
    return times.map(session_of_time).astype(int)


# Merge sessions that share an exact duplicate image so both copies always land in the same fold.
def merge_duplicate_groups(groups: pd.Series, phash: pd.Series) -> pd.Series:
    parent = {g: g for g in groups.unique()}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for _, members in groups.groupby(phash):
        roots = {find(g) for g in members}
        first = min(roots)
        for root in roots:
            parent[root] = first
    return groups.map(find)


# Split images into stratified folds where every group stays inside a single fold.
def grouped_folds(labels: pd.Series, groups: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(labels), -1)
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        folds[test_index] = fold
    return folds


# Summarize class counts per fold for one fold column.
def fold_summary(splits: pd.DataFrame, column: str) -> str:
    table = splits.pivot_table(index="label_name", columns=column, values="key", aggfunc="count", fill_value=0)
    table.loc["total"] = table.sum()
    return f"{column}:\n{table.to_string()}\n"


# Build session-grouped folds for the main benchmark and date-grouped folds for domain shift.
def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-free cross-validation folds.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gap-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(METADATA_CSV, dtype={"date": str, "time": str})
    sessions = session_ids(meta, args.gap_seconds)
    groups = merge_duplicate_groups(sessions, meta["phash"])
    splits = pd.DataFrame({
        "key": meta["key"],
        "label_name": meta["label_name"],
        "session_id": sessions,
        "group_id": groups,
        "date": meta["date"],
        "fold_session": grouped_folds(meta["label"], groups, args.folds, args.seed),
        "fold_date": grouped_folds(meta["label"], meta["date"], args.folds, args.seed),
    })
    splits.to_csv(SPLITS_CSV, index=False)

    report = [f"Sessions: {sessions.nunique()}  groups after duplicate merge: {groups.nunique()}", "",
              fold_summary(splits, "fold_session"), fold_summary(splits, "fold_date"),
              "Dates per date fold:"]
    for fold, dates in splits.groupby("fold_date")["date"].unique().items():
        report.append(f"  fold {fold}: {', '.join(sorted(dates))}")
    text = "\n".join(report)
    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit" / "splits_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
