# FastMatMult Roadmap

## Aim

Implement a high-performance matrix multiplication kernel inspired by Kazushige Goto’s algorithm. The core objective is to replace the current naive `O(n³)` implementation with a cache-aware, SIMD-friendly version that dramatically improves GFLOP/s while retaining correctness and benchmark/plotting capabilities.

## Baseline Snapshot

- **Hardware**: AMD Ryzen 5 8600G (6C/12T), AVX2/AVX-512 capable. Cache sizes per `lscpu`: L1d 192 KiB (6), L2 6 MiB (6 × 1 MiB), L3 16 MiB shared.
- **Naive performance** (`baseline_naive.csv`, 3 repetitions, best run recorded):
  - 128×128 → 3.02 GFLOP/s (1.39 ms)
  - 256×256 → 1.39 GFLOP/s (24.2 ms)
  - 512×512 → 0.39 GFLOP/s (0.687 s)
  - 768×768 → 0.89 GFLOP/s (1.02 s)
  - 1024×1024 → 0.36 GFLOP/s (5.95 s)

These figures form the reference point for evaluating improvements.

## Major Workstreams

1. **Baseline Assessment**
   - Capture reference performance numbers and verify existing correctness tests on a representative set of matrix sizes.
   - Record CPU/cache characteristics for the target platform (register count, SIMD width, L2/L3 sizes).

2. **Blocking Strategy Design**
   - Choose cache block sizes (`mc`, `kc`, `nc`) that maximize reuse of packed panels in L2/L3.
   - Select micro-kernel dimensions (`mr`, `nr`) that keep operands in registers and align with SIMD vector widths.

3. **Packing Routines**
   - Implement functions to pack `mc × kc` blocks of `A` and `kc × nc` blocks of `B` into contiguous, aligned buffers.
   - Handle edge blocks (non-multiples) cleanly, possibly via padding or specialized kernels.

4. **Micro-Kernel Implementation**
   - Write a highly tuned `mr × nr` kernel using scalar loops first, then optimize with intrinsics (e.g., AVX/AVX2).
   - Keep accumulators in registers and store to `C` only after the inner `kc` loop is complete.

5. **Macro-Kernel / Control Flow**
   - Compose the outer loops over `nc`, `kc`, `mc`, invoking packing routines and the micro-kernel to cover the entire matrix.
   - Incorporate double-buffering or prefetch hints if beneficial.

6. **Integration & Benchmarking**
   - Expose the new implementation behind a CLI flag or replace the current kernel once stable.
   - Extend benchmark reporting to compare naive vs tuned implementation, updating CSV/plotting workflows.

## Testing Strategy

- **Correctness Tests**
  - For each development milestone, compare the optimized kernel output to the naive reference across varying sizes (including edge cases such as small sizes, odd dimensions, and non-multiple blocks).
  - Automate via a test harness that generates random matrices, runs both implementations, and asserts element-wise differences within a small epsilon.

- **Performance Regression Tests**
  - Maintain benchmark scripts that log GFLOP/s over time; flag regressions if performance drops beyond a threshold on key matrix sizes.

- **Continuous Validation**
  - Integrate the tests into the build system (e.g., CTest) so they are easy to execute (`cmake --build build && ctest`).
  - Optionally add CI hooks to run correctness tests and lightweight benchmarks on every change.

This plan should keep development structured: establish a reliable baseline, incrementally add Goto-style optimizations, and constantly validate both accuracy and speed gains.
