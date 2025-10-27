import os
import itertools
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
import matplotlib.patches as mpatches


CSV_PATH = "student_lifestyle_dataset.csv"
OUT_DIR = os.path.join("plots", "knn")


def make_3d_plot(df, features, out_path):
    """Create and save a 3D scatter for the given three features.

    Points are colored by Stress_Level (Low/Moderate/High).
    """
    if len(features) != 3:
        raise ValueError("features must be a list of three column names")

    for f in features:
        if f not in df.columns:
            raise KeyError(f"Feature '{f}' not found in dataset columns: {df.columns.tolist()}")

    color_map = {"Low": "green", "Moderate": "yellow", "High": "red"}
    colors = df["Stress_Level"].map(color_map).fillna("gray")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df[features[0]].values
    y = df[features[1]].values
    z = df[features[2]].values

    ax.scatter(x, y, z, c=colors, edgecolor='k', s=30, alpha=0.9)

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    patches = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
    ax.legend(handles=patches, title='Stress_Level', loc='upper left')

    plt.title('3D scatter: ' + ' / '.join(features))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_triplets(df, out_dir):
    # Determine numeric columns to consider (excluding Student_ID if present)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "Student_ID" in numeric_cols:
        numeric_cols.remove("Student_ID")

    triplets = list(itertools.combinations(numeric_cols, 3))
    created = []
    for trip in triplets:
        filename = f"3d_features_{'_'.join(trip)}.png"
        out_path = os.path.join(out_dir, filename)
        make_3d_plot(df, list(trip), out_path)
        created.append(out_path)
    return created


def main():
    parser = argparse.ArgumentParser(description="Create 3D feature scatter plots colored by Stress_Level.")
    parser.add_argument("--features", type=str,
                        help="Comma-separated list of three feature names to plot (e.g. Study_Hours_Per_Day,Sleep_Hours_Per_Day,Physical_Activity_Hours_Per_Day)")
    parser.add_argument("--all", action="store_true", help="Generate 3D plots for all numeric triplets")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    if args.features:
        features = [f.strip() for f in args.features.split(",")]
        if len(features) != 3:
            raise SystemExit("Provide exactly three features with --features")
        out_file = os.path.join(OUT_DIR, f"3d_features_{'_'.join(features)}.png")
        make_3d_plot(df, features, out_file)
        print(f"Saved: {out_file}")
    elif args.all:
        created = generate_all_triplets(df, OUT_DIR)
        print(f"Created {len(created)} files under {OUT_DIR}")
    else:
        # Default: create the previously used triplet
        default = ["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "Physical_Activity_Hours_Per_Day"]
        out_file = os.path.join(OUT_DIR, f"3d_features_{'_'.join(default)}.png")
        make_3d_plot(df, default, out_file)
        print(f"Saved default 3D plot to: {out_file}")


if __name__ == '__main__':
    main()

