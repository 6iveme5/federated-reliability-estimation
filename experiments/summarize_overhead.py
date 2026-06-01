from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize overhead analysis outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/overhead"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/overhead"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in sorted(args.input_dir.glob("overhead_*_seed_*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No overhead CSV files found in {args.input_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["scenario"] = df.apply(scenario_name, axis=1)
    df = df[
        [
            "scenario",
            "dataset",
            "partition",
            "feature_column",
            "optimizer",
            "seed",
            "n_clients",
            "n_teacher_samples",
            "teacher_time_sec",
            "teacher_time_per_sample_ms",
            "surrogate_train_time_sec",
            "surrogate_inference_time_sec",
            "surrogate_inference_per_sample_ms",
            "model_size_mb",
            "total_comm_mb",
            "break_even_inference_batches",
        ]
    ].sort_values("scenario")
    df.to_csv(args.output_dir / "overhead_summary.csv", index=False)
    summary = summarize_overhead(df)
    summary.to_csv(args.output_dir / "overhead_mean_std.csv", index=False)

    markdown = [
        "# Overhead Summary\n",
        "## Multi-Seed Mean and Standard Deviation\n",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "\n\n## Individual Runs\n",
        df.to_markdown(index=False, floatfmt=".4f"),
        "\n\n## Notes\n",
        "- `teacher_time_sec` measures local task-model fitting plus hash-teacher generation.",
        "- `surrogate_train_time_sec` measures federated surrogate training.",
        "- `surrogate_inference_time_sec` is median inference time over repeated full-batch runs.",
        "- Communication is estimated as model upload plus download for each client and round.",
    ]
    (args.output_dir / "OVERHEAD_SUMMARY.md").write_text("\n".join(markdown), encoding="utf-8")

    plot_compute(summary, args.output_dir)
    plot_communication(summary, args.output_dir)
    print(df.to_string(index=False))
    print(f"\nSaved overhead summary to: {args.output_dir / 'OVERHEAD_SUMMARY.md'}")


def scenario_name(row: pd.Series) -> str:
    if row["dataset"] == "federated_csv":
        return "Multi-hospital"
    prefix = str(row["dataset"]).upper()
    if row["partition"] == "iid":
        return f"{prefix}-IID"
    if row["partition"] == "label_skew":
        return f"{prefix}-label"
    if row["feature_column"] == "killip_class":
        return "CHUC-Killip"
    if row["feature_column"] == "st_segment_elevation":
        return "CHUC-ST"
    if row["feature_column"] == "dzclass":
        return "SUPPORT2-disease"
    if row["feature_column"] == "ca":
        return "SUPPORT2-cancer"
    return f"{prefix}-feature"


def summarize_overhead(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "scenario",
        "dataset",
        "partition",
        "feature_column",
        "optimizer",
        "n_clients",
    ]
    metric_cols = [
        "n_teacher_samples",
        "teacher_time_sec",
        "teacher_time_per_sample_ms",
        "surrogate_train_time_sec",
        "surrogate_inference_time_sec",
        "surrogate_inference_per_sample_ms",
        "model_size_mb",
        "total_comm_mb",
        "break_even_inference_batches",
    ]
    grouped = df.groupby(group_cols, dropna=False)[metric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=0).add_suffix("_std")
    count = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, count], axis=1).reset_index().sort_values("scenario")


def plot_compute(df: pd.DataFrame, output_dir: Path) -> None:
    labels = df["scenario"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(x - width, df["teacher_time_sec_mean"], width, label="Teacher generation")
    ax.bar(x, df["surrogate_train_time_sec_mean"], width, label="Surrogate training")
    ax.bar(
        x + width,
        df["surrogate_inference_time_sec_mean"],
        width,
        label="Surrogate inference",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Wall time (seconds, log scale)")
    ax.set_title("Reliability Computation Overhead")
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "overhead_compute_time")


def plot_communication(df: pd.DataFrame, output_dir: Path) -> None:
    labels = df["scenario"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x, df["total_comm_mb_mean"], color="#4c78a8")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Estimated total communication (MB)")
    ax.set_title("Federated Surrogate Communication Overhead")
    save_figure(fig, output_dir / "overhead_communication")


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
