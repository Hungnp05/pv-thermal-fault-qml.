import argparse

import pandas as pd

from pvf.paths import OUTPUT_DIR

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "ece", "brier", "nll", "aurc", "train_seconds"]


# Format each metric as mean plus or minus standard deviation over folds and seeds.
def summarize(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = results.groupby(group_columns)
    table = pd.DataFrame({"runs": grouped.size()})
    for metric in METRICS:
        mean, std = grouped[metric].mean(), grouped[metric].std().fillna(0.0)
        table[metric] = mean.map("{:.4f}".format) + " +- " + std.map("{:.4f}".format)
    return table.reset_index()


# Print and save summary tables for the CNN and head result files that exist.
def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.parse_args()
    groups = {"cnn": ["scheme", "model"], "heads": ["scheme", "features", "name"]}
    for name, columns in groups.items():
        path = OUTPUT_DIR / "results" / f"{name}.csv"
        if not path.exists():
            continue
        table = summarize(pd.read_csv(path), columns)
        table.to_csv(OUTPUT_DIR / "results" / f"{name}_summary.csv", index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
