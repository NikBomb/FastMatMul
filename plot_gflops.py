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
        default="gflops_data.csv",
        help="CSV file produced by fast_mat_mult (default: gflops_data.csv)",
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

    plt.figure(figsize=(8, 5))
    plt.plot(data["size"], data["gflops"], marker="o")
    plt.title(args.title)
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("GFLOP/s")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
