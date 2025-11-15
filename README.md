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
./fast_mat_mult [--output file.csv] [--repetitions count] [--sizes n1,n2,...] [--seed value]
```

- `--output` / `-o`: CSV file to be generated (default `gflops_data.csv`)
- `--repetitions` / `-r`: how many runs per matrix size (best run is recorded, default `3`)
- `--sizes` / `-s`: comma-separated list of matrix dimensions (default `64,128,256,384,512`)
- `--seed`: RNG seed for reproducible inputs

Each CSV record has `size,time_seconds,gflops`. Example execution:

```
./fast_mat_mult --sizes 64,128,256,512,768,1024 --repetitions 5
```

## Plotting in Python

Install dependencies (e.g., `pip install -r requirements.txt`), then run:

```bash
python plot_gflops.py gflops_data.csv
```

The script opens an interactive window visualizing GFLOP/s versus matrix size; use `--title` to customize the figure title.
