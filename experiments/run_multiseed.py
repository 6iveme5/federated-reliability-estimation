from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    dataset: str
    partition: str = "iid"
    feature_column: str | None = None
    n_clients: int = 4


SCENARIOS = {
    "native": Scenario(name="native", dataset="federated_csv", partition="native"),
    "chuc_iid": Scenario(name="chuc_iid", dataset="chuc", partition="iid"),
    "chuc_label_skew": Scenario(
        name="chuc_label_skew",
        dataset="chuc",
        partition="label_skew",
    ),
    "chuc_killip_skew": Scenario(
        name="chuc_killip_skew",
        dataset="chuc",
        partition="feature_skew",
        feature_column="killip_class",
    ),
    "chuc_st_skew": Scenario(
        name="chuc_st_skew",
        dataset="chuc",
        partition="feature_skew",
        feature_column="st_segment_elevation",
    ),
    "support2_iid": Scenario(name="support2_iid", dataset="support2", partition="iid"),
    "support2_label_skew": Scenario(
        name="support2_label_skew",
        dataset="support2",
        partition="label_skew",
    ),
    "support2_dzclass_skew": Scenario(
        name="support2_dzclass_skew",
        dataset="support2",
        partition="feature_skew",
        feature_column="dzclass",
    ),
    "support2_cancer_skew": Scenario(
        name="support2_cancer_skew",
        dataset="support2",
        partition="feature_skew",
        feature_column="ca",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate multi-seed experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--optimizers", default="fedavg,fedprox,fedadam")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multiseed"))
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-surrogate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    scenarios = [SCENARIOS[name.strip()] for name in args.scenarios.split(",") if name.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_transfer:
        for scenario in scenarios:
            for seed in seeds:
                run_transfer(args, scenario, seed)

    if not args.skip_surrogate:
        for scenario in scenarios:
            for seed in seeds:
                run_surrogate(args, scenario, seed)

    aggregate_outputs(args.output_dir, scenarios, seeds)


def run_transfer(args: argparse.Namespace, scenario: Scenario, seed: int) -> None:
    output_dir = args.output_dir / scenario.name / f"seed_{seed}" / "transfer"
    command = [
        sys.executable,
        "experiments/run_transfer_analysis.py",
        "--config",
        str(args.config),
        "--dataset",
        scenario.dataset,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if scenario.dataset in {"chuc", "support2"}:
        command += ["--partition", scenario.partition, "--n-clients", str(scenario.n_clients)]
        if scenario.feature_column is not None:
            command += ["--feature-column", scenario.feature_column]
    run_command(command)


def run_surrogate(args: argparse.Namespace, scenario: Scenario, seed: int) -> None:
    output_dir = args.output_dir / scenario.name / f"seed_{seed}" / "surrogate"
    command = [
        sys.executable,
        "experiments/run_surrogate_learning.py",
        "--config",
        str(args.config),
        "--dataset",
        scenario.dataset,
        "--seed",
        str(seed),
        "--optimizers",
        args.optimizers,
        "--output-dir",
        str(output_dir),
    ]
    if scenario.dataset in {"chuc", "support2"}:
        command += ["--partition", scenario.partition, "--n-clients", str(scenario.n_clients)]
        if scenario.feature_column is not None:
            command += ["--feature-column", scenario.feature_column]
    run_command(command)


def run_command(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def aggregate_outputs(output_dir: Path, scenarios: list[Scenario], seeds: list[int]) -> None:
    transfer = load_transfer_outputs(output_dir, scenarios, seeds)
    surrogate = load_surrogate_outputs(output_dir, scenarios, seeds)

    if not transfer.empty:
        transfer.to_csv(output_dir / "all_transfer_metrics.csv", index=False)
        transfer_summary = summarize_metrics(
            transfer,
            group_cols=["scenario", "dataset", "partition_detail", "method", "score"],
            metric_cols=["pearson", "spearman", "mae", "rmse", "r2"],
        )
        transfer_summary.to_csv(output_dir / "summary_transfer_mean_std.csv", index=False)
    else:
        transfer_summary = pd.DataFrame()

    if not surrogate.empty:
        surrogate.to_csv(output_dir / "all_surrogate_metrics.csv", index=False)
        metric_cols = [
            column
            for column in surrogate.columns
            if column
            in {"pearson", "spearman", "mae", "rmse", "r2"}
            or column.endswith("_auroc")
            or column.endswith("_auprc")
            or column.endswith("_risk_at_5")
        ]
        surrogate_summary = summarize_metrics(
            surrogate,
            group_cols=["scenario", "dataset", "partition_detail", "optimizer"],
            metric_cols=metric_cols,
        )
        surrogate_summary.to_csv(output_dir / "summary_surrogate_mean_std.csv", index=False)
        best = surrogate_summary.loc[surrogate_summary.groupby("scenario")["r2_mean"].idxmax()]
        best.to_csv(output_dir / "summary_surrogate_best_by_r2_mean.csv", index=False)
    else:
        surrogate_summary = pd.DataFrame()
        best = pd.DataFrame()

    write_markdown_summary(output_dir, transfer_summary, surrogate_summary, best)


def load_transfer_outputs(
    output_dir: Path,
    scenarios: list[Scenario],
    seeds: list[int],
) -> pd.DataFrame:
    frames = []
    for scenario in scenarios:
        for seed in seeds:
            path = transfer_csv_path(output_dir, scenario, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df.insert(0, "scenario", scenario.name)
            df["partition_detail"] = partition_detail(scenario)
            df["seed"] = seed
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_surrogate_outputs(
    output_dir: Path,
    scenarios: list[Scenario],
    seeds: list[int],
) -> pd.DataFrame:
    frames = []
    for scenario in scenarios:
        for seed in seeds:
            path = surrogate_csv_path(output_dir, scenario, seed)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df.insert(0, "scenario", scenario.name)
            df["partition_detail"] = partition_detail(scenario)
            df["seed"] = seed
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def transfer_csv_path(output_dir: Path, scenario: Scenario, seed: int) -> Path:
    partition = "iid" if scenario.dataset == "federated_csv" else scenario.partition
    return (
        output_dir
        / scenario.name
        / f"seed_{seed}"
        / "transfer"
        / f"transfer_metrics_{scenario.dataset}_{partition}.csv"
    )


def surrogate_csv_path(output_dir: Path, scenario: Scenario, seed: int) -> Path:
    partition = "iid" if scenario.dataset == "federated_csv" else scenario.partition
    return (
        output_dir
        / scenario.name
        / f"seed_{seed}"
        / "surrogate"
        / f"surrogate_metrics_{scenario.dataset}_{partition}.csv"
    )


def partition_detail(scenario: Scenario) -> str:
    if scenario.dataset == "federated_csv":
        return "native"
    if scenario.feature_column is None:
        return scenario.partition
    return f"{scenario.partition}_{scenario.feature_column}"


def summarize_metrics(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[metric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().iloc[:, :1].rename(columns={metric_cols[0]: "n_seeds"})
    return pd.concat([mean, std, count], axis=1).reset_index()


def write_markdown_summary(
    output_dir: Path,
    transfer_summary: pd.DataFrame,
    surrogate_summary: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    sections = ["# Multi-Seed Experiment Summary\n"]
    if not transfer_summary.empty:
        selected = transfer_summary[
            (transfer_summary["method"].eq("gmm"))
            | (
                transfer_summary["method"].eq("rhh")
                & transfer_summary["score"].eq("s_knn_global")
            )
            | transfer_summary["method"].eq("embedding")
        ]
        sections.append("## Transfer Selected Scores\n")
        sections.append(selected.to_markdown(index=False, floatfmt=".4f"))
    if not surrogate_summary.empty:
        sections.append("\n\n## Surrogate All Optimizers\n")
        sections.append(surrogate_summary.to_markdown(index=False, floatfmt=".4f"))
    if not best.empty:
        sections.append("\n\n## Best Surrogate Optimizer by Mean R2\n")
        sections.append(best.to_markdown(index=False, floatfmt=".4f"))
    (output_dir / "MULTISEED_SUMMARY.md").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
