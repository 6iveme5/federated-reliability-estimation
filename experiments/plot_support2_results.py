from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "support2_iid": "SUPPORT2-IID",
    "support2_label_skew": "SUPPORT2-label",
    "support2_dzclass_skew": "SUPPORT2-disease",
    "support2_cancer_skew": "SUPPORT2-cancer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SUPPORT2 experiment results.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/support2_multiseed_5"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures_support2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    transfer = pd.read_csv(args.input_dir / "summary_transfer_mean_std.csv")
    surrogate = pd.read_csv(args.input_dir / "summary_surrogate_mean_std.csv")
    best = pd.read_csv(args.input_dir / "summary_surrogate_best_by_r2_mean.csv")

    plot_transfer_heatmap(transfer, args.output_dir)
    plot_baseline_auroc(best, args.output_dir)
    plot_optimizer_r2(surrogate, args.output_dir)
    plot_surrogate_vs_teacher(best, args.output_dir)


def plot_transfer_heatmap(transfer: pd.DataFrame, output_dir: Path) -> None:
    selected = transfer[
        (transfer["method"].eq("gmm"))
        | ((transfer["method"].eq("rhh")) & transfer["score"].eq("s_knn_global"))
        | transfer["method"].eq("embedding")
    ].copy()
    methods = [("gmm", "GMM"), ("rhh", "RHH-kNN"), ("embedding", "Embedding")]
    scenarios = list(SCENARIO_LABELS)
    matrix = np.full((len(scenarios), len(methods)), np.nan)
    for i, scenario in enumerate(scenarios):
        for j, (method, _) in enumerate(methods):
            row = selected[selected["scenario"].eq(scenario) & selected["method"].eq(method)]
            if not row.empty:
                matrix[i, j] = float(row["spearman_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(methods)), [label for _, label in methods])
    ax.set_yticks(range(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("SUPPORT2 CL-to-FL Reliability Transfer")
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Spearman")
    save_figure(fig, output_dir / "support2_transfer_spearman")


def plot_baseline_auroc(best: pd.DataFrame, output_dir: Path) -> None:
    methods = [
        "centroid_fl",
        "pmax_fl",
        "entropy_fl",
        "margin_fl",
        "feature_knn",
        "teacher",
        "surrogate",
    ]
    labels = {
        "centroid_fl": "Centroid(FL)",
        "pmax_fl": "pmax(FL)",
        "entropy_fl": "Entropy(FL)",
        "margin_fl": "Margin(FL)",
        "feature_knn": "Feature-kNN",
        "teacher": "Teacher",
        "surrogate": "Surrogate",
    }
    scenarios = list(SCENARIO_LABELS)
    rows = best.set_index("scenario")
    matrix = np.full((len(scenarios), len(methods)), np.nan)
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            matrix[i, j] = rows.loc[scenario, f"{method}_auroc_mean"]

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    image = ax.imshow(matrix, cmap="magma", vmin=0.45, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=25, ha="right")
    ax.set_yticks(range(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("SUPPORT2 Error Detection AUROC")
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="AUROC")
    save_figure(fig, output_dir / "support2_baseline_auroc_heatmap")


def plot_optimizer_r2(surrogate: pd.DataFrame, output_dir: Path) -> None:
    scenarios = list(SCENARIO_LABELS)
    optimizers = ["fedavg", "fedprox"]
    colors = {"fedavg": "#4c78a8", "fedprox": "#f58518"}
    x = np.arange(len(scenarios))
    width = 0.28
    rows = surrogate.set_index(["scenario", "optimizer"])

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for idx, optimizer in enumerate(optimizers):
        means = [rows.loc[(scenario, optimizer), "r2_mean"] for scenario in scenarios]
        stds = [rows.loc[(scenario, optimizer), "r2_std"] for scenario in scenarios]
        ax.bar(
            x + (idx - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=colors[optimizer],
            label=optimizer_name(optimizer),
        )
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Teacher approximation R2")
    ax.set_title("SUPPORT2 Surrogate Approximation")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, ncols=2)
    save_figure(fig, output_dir / "support2_surrogate_optimizer_r2")


def plot_surrogate_vs_teacher(best: pd.DataFrame, output_dir: Path) -> None:
    scenarios = list(SCENARIO_LABELS)
    rows = best.set_index("scenario")
    x = np.arange(len(scenarios))
    width = 0.24
    methods = [
        ("pmax_fl_auroc_mean", "pmax(FL)", "#9ecae9"),
        ("teacher_auroc_mean", "Teacher", "#f58518"),
        ("surrogate_auroc_mean", "Surrogate", "#4c78a8"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for idx, (column, label, color) in enumerate(methods):
        means = [rows.loc[scenario, column] for scenario in scenarios]
        ax.bar(x + (idx - 1) * width, means, width, color=color, label=label)
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel("Misclassification detection AUROC")
    ax.set_title("SUPPORT2 Reliability Utility")
    ax.legend(frameon=False, ncols=3)
    save_figure(fig, output_dir / "support2_error_detection_auroc")


def optimizer_name(optimizer: str) -> str:
    return {"fedavg": "FedAvg", "fedprox": "FedProx"}[optimizer]


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
