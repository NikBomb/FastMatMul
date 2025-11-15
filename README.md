# FastMatMult

Simple benchmark for a naive dense matrix multiplication kernel in C++. The program multiplies two random square matrices for a set of sizes, captures the fastest runtime per size, and writes time/GFLOP data to a CSV file that can be plotted from Python.

## Build

### CMake

```bash
cmake -S . -B build
cmake --build build
```

### Single-file build

```bash
g++ -O3 -std=c++17 -march=native -DNDEBUG main.cpp -o fast_mat_mult
```

## Usage

```
./fast_mat_mult [--output file.csv] [--repetitions count] [--sizes n1,n2,...] [--seed value] [--algo naive|goto|both] [--verify]
```

- `--output` / `-o`: CSV file to be generated (default `gflops_data.csv`)
- `--repetitions` / `-r`: how many runs per matrix size (best run is recorded, default `3`)
- `--sizes` / `-s`: comma-separated list of matrix dimensions (default `64,128,256,384,512`)
- `--seed`: RNG seed for reproducible inputs
- `--algo`: choose the naive implementation, the (future) optimized Goto-style path, or run `both` for side-by-side data
- `--verify`: optional correctness check that runs naive vs Goto once per size (not timed)

Each CSV record has `size,algorithm,time_seconds,gflops`. Example execution:

```
./fast_mat_mult --sizes 64,128,256,512 --repetitions 5 --algo both --verify
```

## Plotting in Python

Install dependencies (e.g., `pip install -r requirements.txt`), then run:

```bash
python plot_gflops.py baseline_naive.csv  # or pass any CSV produced by the benchmark
```

The script plots GFLOP/s versus matrix size for every algorithm found in the CSV and opens an interactive window; use `--title` to customize the figure title.
