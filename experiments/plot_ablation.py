from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "native": "Multi-hospital",
    "chuc_iid": "CHUC-IID",
    "chuc_label_skew": "CHUC-label",
    "chuc_killip_skew": "CHUC-Killip",
    "chuc_st_skew": "CHUC-ST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot surrogate ablation results.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/surrogate_ablation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures_ablation"))
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
    summary = pd.read_csv(args.input_dir / "surrogate_ablation_summary.csv")
    plot_hash_dim(summary, args.output_dir)
    plot_model_size(summary, args.output_dir)


def plot_hash_dim(summary: pd.DataFrame, output_dir: Path) -> None:
    df = summary[summary["ablation"].eq("hash_dim")].copy()
    scenarios = list(SCENARIO_LABELS)
    settings = ["hash_64", "hash_128", "hash_256"]
    colors = {"hash_64": "#9ecae9", "hash_128": "#4c78a8", "hash_256": "#f58518"}
    plot_grouped_bars(
        df=df,
        scenarios=scenarios,
        settings=settings,
        colors=colors,
        metric="r2",
        ylabel="Teacher approximation R2",
        title="Projection Dimension Ablation",
        output_path=output_dir / "hash_dim_r2",
    )
    plot_grouped_bars(
        df=df,
        scenarios=scenarios,
        settings=settings,
        colors=colors,
        metric="surrogate_auroc",
        ylabel="Misclassification detection AUROC",
        title="Projection Dimension Ablation",
        output_path=output_dir / "hash_dim_auroc",
        ylim=(0.6, 0.9),
    )


def plot_model_size(summary: pd.DataFrame, output_dir: Path) -> None:
    df = summary[summary["ablation"].eq("model_size")].copy()
    scenarios = list(SCENARIO_LABELS)
    settings = ["model_small", "model_current"]
    colors = {"model_small": "#9ecae9", "model_current": "#4c78a8"}
    plot_grouped_bars(
        df=df,
        scenarios=scenarios,
        settings=settings,
        colors=colors,
        metric="r2",
        ylabel="Teacher approximation R2",
        title="Surrogate Size Ablation",
        output_path=output_dir / "model_size_r2",
    )
    plot_grouped_bars(
        df=df,
        scenarios=scenarios,
        settings=settings,
        colors=colors,
        metric="surrogate_auroc",
        ylabel="Misclassification detection AUROC",
        title="Surrogate Size Ablation",
        output_path=output_dir / "model_size_auroc",
        ylim=(0.6, 0.9),
    )


def plot_grouped_bars(
    df: pd.DataFrame,
    scenarios: list[str],
    settings: list[str],
    colors: dict[str, str],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    rows = df.set_index(["scenario", "setting"])
    x = np.arange(len(scenarios))
    width = min(0.75 / len(settings), 0.26)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for idx, setting in enumerate(settings):
        means = [rows.loc[(scenario, setting), f"{metric}_mean"] for scenario in scenarios]
        stds = [rows.loc[(scenario, setting), f"{metric}_std"] for scenario in scenarios]
        ax.bar(
            x + (idx - (len(settings) - 1) / 2) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=colors[setting],
            label=setting_label(setting),
        )
    ax.set_xticks(x, [SCENARIO_LABELS[scenario] for scenario in scenarios], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncols=len(settings))
    save_figure(fig, output_path)


def setting_label(setting: str) -> str:
    return {
        "hash_64": "64",
        "hash_128": "128",
        "hash_256": "256",
        "model_small": "Small",
        "model_current": "Current",
    }[setting]


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
