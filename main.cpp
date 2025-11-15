#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/goto_matmul.h"
#include "src/naive_matmul.h"

namespace {

enum class AlgorithmKind {
    Naive,
    Goto,
};

enum class AlgorithmMode {
    Naive,
    Goto,
    Both,
};

struct Config {
    std::string output_file = "gflops_data.csv";
    int repetitions = 3;
    std::vector<int> sizes = {64, 128, 256, 384, 512};
    unsigned seed = std::random_device{}();
    AlgorithmMode algorithm_mode = AlgorithmMode::Naive;
    bool verify = false;
};

struct RunResult {
    int size;
    AlgorithmKind algorithm;
    double seconds;
    double gflops;
};

const char* algorithm_name(AlgorithmKind algo) {
    switch (algo) {
        case AlgorithmKind::Naive:
            return "naive";
        case AlgorithmKind::Goto:
            return "goto";
    }
    return "unknown";
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name
              << " [--output <path>] [--repetitions <count>] [--sizes "
                 "n1,n2,...] [--seed <value>] [--algo naive|goto|both] [--verify]\n";
    std::cout << "  --output / -o       Output CSV file (default: gflops_data.csv)\n";
    std::cout << "  --repetitions / -r  Number of runs per size (default: 3)\n";
    std::cout << "  --sizes / -s        Comma separated list of square matrix sizes\n";
    std::cout << "  --seed              Seed for RNG used to fill input matrices\n";
    std::cout << "  --algo              Algorithm to use: naive, goto, or both (default: naive)\n";
    std::cout << "  --verify            Run both algorithms once to compare results (not timed)\n";
    std::cout << "  --help / -h         Show this message\n";
}

std::vector<int> parse_sizes(const std::string& arg) {
    std::vector<int> sizes;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        try {
            int value = std::stoi(token);
            if (value <= 0) {
                throw std::invalid_argument("Matrix size must be positive");
            }
            sizes.push_back(value);
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid matrix size: " + token);
        }
    }
    if (sizes.empty()) {
        throw std::invalid_argument("No matrix sizes provided");
    }
    return sizes;
}

Config parse_arguments(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--output requires a value");
            }
            cfg.output_file = argv[++i];
        } else if (arg == "--repetitions" || arg == "-r") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--repetitions requires a value");
            }
            cfg.repetitions = std::stoi(argv[++i]);
            if (cfg.repetitions <= 0) {
                throw std::invalid_argument("Repetitions must be positive");
            }
        } else if (arg == "--sizes" || arg == "-s") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--sizes requires a value");
            }
            cfg.sizes = parse_sizes(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--seed requires a value");
            }
            cfg.seed = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--algo") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("--algo requires a value");
            }
            std::string algo = argv[++i];
            if (algo == "naive") {
                cfg.algorithm_mode = AlgorithmMode::Naive;
            } else if (algo == "goto") {
                cfg.algorithm_mode = AlgorithmMode::Goto;
            } else if (algo == "both") {
                cfg.algorithm_mode = AlgorithmMode::Both;
            } else {
                throw std::invalid_argument("Unknown algorithm: " + algo);
            }
        } else if (arg == "--verify") {
            cfg.verify = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    return cfg;
}

void fill_random(std::vector<double>& matrix, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : matrix) {
        value = dist(rng);
    }
}

RunResult benchmark_size(int n, int repetitions, const std::vector<double>& A, const std::vector<double>& B,
                         AlgorithmKind algorithm, const BlockParams& block_params) {
    const std::size_t total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<double> C(total_elements);

    double best_seconds = std::numeric_limits<double>::max();

    for (int rep = 0; rep < repetitions; ++rep) {
        auto start = std::chrono::high_resolution_clock::now();
        if (algorithm == AlgorithmKind::Naive) {
            naive_matmul(A.data(), B.data(), C.data(), n);
        } else {
            goto_matmul(A.data(), B.data(), C.data(), n, block_params);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        best_seconds = std::min(best_seconds, elapsed.count());
    }

    const double operations = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    const double gflops = (operations / best_seconds) / 1e9;

    return RunResult{n, algorithm, best_seconds, gflops};
}

void write_results(const std::string& path, const std::vector<RunResult>& results) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }
    out << "size,algorithm,time_seconds,gflops\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        out << result.size << ',' << algorithm_name(result.algorithm) << ',' << result.seconds << ','
            << result.gflops << '\n';
    }
}

bool verify_algorithms(int n, const std::vector<double>& A, const std::vector<double>& B, const BlockParams& params) {
    const std::size_t total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<double> C_naive(total_elements);
    std::vector<double> C_goto(total_elements);

    naive_matmul(A.data(), B.data(), C_naive.data(), n);
    goto_matmul(A.data(), B.data(), C_goto.data(), n, params);

    const double epsilon = 1e-9;
    for (std::size_t idx = 0; idx < total_elements; ++idx) {
        double ref = C_naive[idx];
        double val = C_goto[idx];
        double diff = std::abs(ref - val);
        double scale = std::max(1.0, std::max(std::abs(ref), std::abs(val)));
        if (diff > epsilon * scale) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    try {
        cfg = parse_arguments(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::mt19937 rng(cfg.seed);
    BlockParams block_params;

    std::vector<AlgorithmKind> algorithms_to_run;
    switch (cfg.algorithm_mode) {
        case AlgorithmMode::Naive:
            algorithms_to_run.push_back(AlgorithmKind::Naive);
            break;
        case AlgorithmMode::Goto:
            algorithms_to_run.push_back(AlgorithmKind::Goto);
            break;
        case AlgorithmMode::Both:
            algorithms_to_run.push_back(AlgorithmKind::Naive);
            algorithms_to_run.push_back(AlgorithmKind::Goto);
            break;
    }

    std::vector<RunResult> results;
    results.reserve(cfg.sizes.size() * algorithms_to_run.size());

    std::cout << "Running matrix multiplication benchmarks\n";
    std::cout << "Output file: " << cfg.output_file << '\n';
    std::cout << "Repetitions per size: " << cfg.repetitions << '\n';
    std::cout << "Algorithms:";
    for (AlgorithmKind algo : algorithms_to_run) {
        std::cout << ' ' << algorithm_name(algo);
    }
    std::cout << '\n';
    if (cfg.verify) {
        std::cout << "Verification: enabled (naive vs goto)\n";
    }

    for (int size : cfg.sizes) {
        const std::size_t total_elements = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
        std::vector<double> A(total_elements);
        std::vector<double> B(total_elements);
        fill_random(A, rng);
        fill_random(B, rng);

        std::cout << "Size " << size << 'x' << size << '\n';
        for (AlgorithmKind algo : algorithms_to_run) {
            RunResult res = benchmark_size(size, cfg.repetitions, A, B, algo, block_params);
            results.push_back(res);
            std::cout << "  [" << algorithm_name(algo) << "] " << res.gflops << " GFLOP/s (best of "
                      << cfg.repetitions << " runs, " << res.seconds << " s)\n";
        }

        if (cfg.verify) {
            bool ok = verify_algorithms(size, A, B, block_params);
            std::cout << "  Verification: " << (ok ? "passed" : "FAILED") << '\n';
            if (!ok) {
                std::cerr << "Verification failed for size " << size << '\n';
                return EXIT_FAILURE;
            }
        }
    }

    try {
        write_results(cfg.output_file, results);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Benchmark data written to " << cfg.output_file << '\n';
    std::cout << "You can load it in Python with pandas or numpy to plot GFLOPs vs size.\n";
    return EXIT_SUCCESS;
}
