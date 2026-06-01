from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    dataset: str
    partition: str = "iid"
    feature_column: str | None = None
    optimizer: str = "fedavg"


SCENARIOS = {
    "native": Scenario(dataset="federated_csv"),
    "chuc_iid": Scenario(dataset="chuc"),
    "chuc_label_skew": Scenario(dataset="chuc", partition="label_skew", optimizer="fedprox"),
    "chuc_killip_skew": Scenario(
        dataset="chuc",
        partition="feature_skew",
        feature_column="killip_class",
        optimizer="fedprox",
    ),
    "chuc_st_skew": Scenario(
        dataset="chuc",
        partition="feature_skew",
        feature_column="st_segment_elevation",
        optimizer="fedprox",
    ),
    "support2_iid": Scenario(dataset="support2", feature_column="dzclass"),
    "support2_label_skew": Scenario(
        dataset="support2",
        partition="label_skew",
        feature_column="dzclass",
        optimizer="fedprox",
    ),
    "support2_dzclass_skew": Scenario(
        dataset="support2",
        partition="feature_skew",
        feature_column="dzclass",
        optimizer="fedprox",
    ),
    "support2_cancer_skew": Scenario(
        dataset="support2",
        partition="feature_skew",
        feature_column="ca",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed overhead experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--repeat-inference", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overhead_multiseed"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    scenarios = [SCENARIOS[name.strip()] for name in args.scenarios.split(",") if name.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    if args.threads_per_job < 1:
        raise ValueError("--threads-per-job must be at least 1")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(run_overhead, args, scenario, seed)
            for scenario in scenarios
            for seed in seeds
        ]
        for future in futures:
            future.result()

    subprocess.run(
        [
            sys.executable,
            "experiments/summarize_overhead.py",
            "--input-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )


def run_overhead(args: argparse.Namespace, scenario: Scenario, seed: int) -> None:
    if args.skip_existing and overhead_csv_path(args.output_dir, scenario, seed).exists():
        return
    command = [
        sys.executable,
        "experiments/run_overhead_analysis.py",
        "--config",
        str(args.config),
        "--dataset",
        scenario.dataset,
        "--partition",
        scenario.partition,
        "--optimizer",
        scenario.optimizer,
        "--seed",
        str(seed),
        "--repeat-inference",
        str(args.repeat_inference),
        "--output-dir",
        str(args.output_dir),
    ]
    if scenario.feature_column is not None:
        command += ["--feature-column", scenario.feature_column]
    print("\n$", " ".join(command), flush=True)
    env = os.environ.copy()
    thread_count = str(args.threads_per_job)
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[variable] = thread_count
    for attempt in range(args.retries + 1):
        result = subprocess.run(command, check=False, env=env)
        if result.returncode == 0:
            return
        if attempt == args.retries:
            raise subprocess.CalledProcessError(result.returncode, command)
        time.sleep(2.0)


def overhead_csv_path(output_dir: Path, scenario: Scenario, seed: int) -> Path:
    if scenario.dataset != "federated_csv" and scenario.partition == "feature_skew":
        name = f"{scenario.dataset}_{scenario.feature_column}_skew"
    elif scenario.dataset != "federated_csv":
        name = f"{scenario.dataset}_{scenario.partition}"
    else:
        name = "native"
    return output_dir / f"overhead_{name}_{scenario.optimizer}_seed_{seed}.csv"


if __name__ == "__main__":
    main()
