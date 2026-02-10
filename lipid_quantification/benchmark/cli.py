from __future__ import annotations

import argparse
from pathlib import Path

from lipid_quantification.benchmark.benchmark import Benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark grid of experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="benchmarks",
        help="Where TrainingRun writes per-run directories.",
    )
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default="benchmarks",
        help="Where Benchmark writes aggregated CSVs.",
    )
    args = parser.parse_args()

    bench = Benchmark.from_yaml(
        Path(args.config),
        runs_root=args.runs_root,
        benchmark_root=args.benchmark_root,
    )
    result = bench.run()

    print(f"Benchmark directory: {result.benchmark_dir}")
    print(f"Internal CSV:        {result.internal_csv}")
    print(f"External CSV:        {result.external_csv}")
    print(f"Runs executed:       {result.n_runs}")


if __name__ == "__main__":
    main()