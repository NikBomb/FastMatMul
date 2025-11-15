#!/usr/bin/env python3
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GFLOP/s vs. matrix size from the benchmark CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="baseline_naive.csv",
        help="CSV file produced by fast_mat_mult (default: baseline_naive.csv)",
    )
    parser.add_argument(
        "--title",
        default="Naive Matrix Multiplication Performance",
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if data.empty:
        raise SystemExit("CSV file is empty")

    if "algorithm" not in data.columns:
        data["algorithm"] = "unknown"

    plt.figure(figsize=(8, 5))
    for algorithm, group in data.groupby("algorithm"):
        sorted_group = group.sort_values("size")
        plt.plot(sorted_group["size"], sorted_group["gflops"], marker="o", label=algorithm)
    plt.title(args.title)
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("GFLOP/s")
    plt.grid(True, linestyle="--", alpha=0.6)
    if data["algorithm"].nunique() > 1 or data["algorithm"].iloc[0] != "unknown":
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
