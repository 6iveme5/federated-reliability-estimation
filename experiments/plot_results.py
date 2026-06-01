from __future__ import annotations

import argparse
import json
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

METHOD_LABELS = {
    "gmm": "GMM",
    "rhh": "RHH-kNN",
    "embedding": "Embedding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from experiment outputs.")
    parser.add_argument("--multiseed-dir", type=Path, default=Path("outputs/multiseed_5"))
    parser.add_argument("--full-dir", type=Path, default=Path("outputs/full"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
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

    transfer_path = args.multiseed_dir / "summary_transfer_mean_std.csv"
    surrogate_path = args.multiseed_dir / "summary_surrogate_mean_std.csv"
    best_path = args.multiseed_dir / "summary_surrogate_best_by_r2_mean.csv"

    transfer = pd.read_csv(transfer_path) if transfer_path.exists() else pd.DataFrame()
    surrogate = pd.read_csv(surrogate_path) if surrogate_path.exists() else pd.DataFrame()
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    if not transfer.empty:
        plot_transfer_heatmap(transfer, args.output_dir)
    if not surrogate.empty:
        plot_optimizer_r2_bars(surrogate, args.output_dir)
    if not best.empty:
        plot_auroc_comparison(best, args.output_dir)
        plot_baseline_auroc_comparison(best, args.output_dir)
    plot_convergence_curves(args.full_dir, args.output_dir)
    plot_transfer_scatters(args.full_dir, args.output_dir)
    if not surrogate.empty and not best.empty:
        plot_native_focus_figures(surrogate, best, args.full_dir, args.output_dir)


def plot_transfer_heatmap(transfer: pd.DataFrame, output_dir: Path) -> None:
    selected = transfer[
        (transfer["method"].eq("gmm"))
        | ((transfer["method"].eq("rhh")) & transfer["score"].eq("s_knn_global"))
        | transfer["method"].eq("embedding")
    ].copy()
    selected["method_key"] = selected["method"].where(
        ~selected["method"].eq("rhh"),
        "rhh",
    )
    scenarios = ["native", "chuc_iid", "chuc_label_skew", "chuc_killip_skew", "chuc_st_skew"]
    methods = ["gmm", "rhh", "embedding"]
    matrix = np.full((len(scenarios), len(methods)), np.nan)
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            row = selected[(selected["scenario"].eq(scenario)) & selected["method_key"].eq(method)]
            if not row.empty:
                matrix[i, j] = float(row["spearman_mean"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.set_yticks(range(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("CL-to-FL Reliability Transfer (Spearman)")
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Spearman")
    save_figure(fig, output_dir / "transfer_spearman_heatmap")


def plot_optimizer_r2_bars(surrogate: pd.DataFrame, output_dir: Path) -> None:
    scenarios = ["native", "chuc_iid", "chuc_label_skew", "chuc_killip_skew", "chuc_st_skew"]
    optimizers = ["fedavg", "fedprox", "fedadam"]
    colors = {"fedavg": "#4c78a8", "fedprox": "#f58518", "fedadam": "#54a24b"}
    width = 0.24
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for offset, optimizer in enumerate(optimizers):
        rows = surrogate.set_index(["scenario", "optimizer"])
        means = [rows.loc[(scenario, optimizer), "r2_mean"] for scenario in scenarios]
        stds = [rows.loc[(scenario, optimizer), "r2_std"] for scenario in scenarios]
        ax.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=stds,
            label=optimizer_name(optimizer),
            color=colors[optimizer],
            capsize=3,
        )
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Teacher approximation R2")
    ax.set_title("Federated Surrogate Approximation Across Optimizers")
    ax.legend(frameon=False, ncols=3)
    ax.axhline(0, color="black", linewidth=0.8)
    save_figure(fig, output_dir / "surrogate_optimizer_r2")


def plot_auroc_comparison(best: pd.DataFrame, output_dir: Path) -> None:
    scenarios = ["native", "chuc_iid", "chuc_label_skew", "chuc_killip_skew", "chuc_st_skew"]
    rows = best.set_index("scenario")
    pmax_mean = "pmax_fl_auroc_mean" if "pmax_fl_auroc_mean" in best.columns else "pmax_auroc_mean"
    pmax_std = "pmax_fl_auroc_std" if "pmax_fl_auroc_std" in best.columns else "pmax_auroc_std"
    pmax_label = "pmax(FL)" if pmax_mean.startswith("pmax_fl") else "pmax"
    methods = [
        (pmax_mean, pmax_std, pmax_label),
        ("teacher_auroc_mean", "teacher_auroc_std", "Teacher"),
        ("surrogate_auroc_mean", "surrogate_auroc_std", "Surrogate"),
    ]
    colors = ["#9ecae9", "#f58518", "#4c78a8"]
    width = 0.24
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for offset, (mean_col, std_col, label) in enumerate(methods):
        means = [rows.loc[scenario, mean_col] for scenario in scenarios]
        stds = [rows.loc[scenario, std_col] for scenario in scenarios]
        ax.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=stds,
            label=label,
            color=colors[offset],
            capsize=3,
        )
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Misclassification detection AUROC")
    ax.set_title("Reliability Utility for Error Detection")
    ax.legend(frameon=False, ncols=3)
    save_figure(fig, output_dir / "error_detection_auroc")


def plot_baseline_auroc_comparison(best: pd.DataFrame, output_dir: Path) -> None:
    scenarios = ["native", "chuc_iid", "chuc_label_skew", "chuc_killip_skew", "chuc_st_skew"]
    methods = [
        "pmax",
        "pmax_fl",
        "entropy",
        "entropy_fl",
        "margin",
        "margin_fl",
        "centroid",
        "centroid_fl",
        "feature_knn",
        "lof",
        "teacher",
        "surrogate",
    ]
    available = [method for method in methods if f"{method}_auroc_mean" in best.columns]
    if len(available) <= 3:
        return

    rows = best.set_index("scenario")
    matrix = np.full((len(scenarios), len(available)), np.nan)
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(available):
            matrix[i, j] = rows.loc[scenario, f"{method}_auroc_mean"]

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    image = ax.imshow(matrix, cmap="magma", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(available)), [method_label(method) for method in available], rotation=25, ha="right")
    ax.set_yticks(range(len(scenarios)), [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("Error Detection AUROC Across Reliability Baselines")
    for i in range(len(scenarios)):
        for j in range(len(available)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="AUROC")
    save_figure(fig, output_dir / "baseline_auroc_heatmap")


def plot_convergence_curves(full_dir: Path, output_dir: Path) -> None:
    scenarios = {
        "Native": full_dir / "surrogate_native" / "surrogate_history_federated_csv_iid.json",
        "CHUC-label": full_dir
        / "surrogate_chuc_label_skew"
        / "surrogate_history_chuc_label_skew.json",
    }
    colors = {"fedavg": "#4c78a8", "fedprox": "#f58518", "fedadam": "#54a24b"}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.0, 3.0), sharey=True)
    for ax, (title, path) in zip(np.atleast_1d(axes), scenarios.items(), strict=True):
        if not path.exists():
            ax.set_visible(False)
            continue
        history = json.loads(path.read_text(encoding="utf-8"))
        for optimizer, records in history.items():
            rounds = [record["round"] for record in records]
            r2 = [record["r2"] for record in records]
            ax.plot(rounds, r2, label=optimizer_name(optimizer), color=colors[optimizer])
        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Teacher approximation R2")
    axes[-1].legend(frameon=False)
    fig.suptitle("Surrogate Training Convergence")
    save_figure(fig, output_dir / "surrogate_convergence_r2")


def plot_transfer_scatters(full_dir: Path, output_dir: Path) -> None:
    specs = [
        (
            "Native RHH-kNN",
            full_dir / "transfer_native" / "transfer_scores_federated_csv_iid.csv",
            "rhh",
            "s_knn_global",
        ),
        (
            "Native Embedding",
            full_dir / "transfer_native" / "transfer_scores_federated_csv_iid.csv",
            "embedding",
            "knn",
        ),
        (
            "CHUC Label-Skew RHH-kNN",
            full_dir / "transfer_chuc_label_skew" / "transfer_scores_chuc_label_skew.csv",
            "rhh",
            "s_knn_global",
        ),
    ]
    existing = [spec for spec in specs if spec[1].exists()]
    if not existing:
        return

    fig, axes = plt.subplots(1, len(existing), figsize=(3.2 * len(existing), 3.0), sharex=True, sharey=True)
    for ax, (title, path, method, score) in zip(np.atleast_1d(axes), existing, strict=True):
        df = pd.read_csv(path)
        df = df[df["method"].eq(method) & df["score"].eq(score)]
        if len(df) > 3000:
            df = df.sample(3000, random_state=42)
        ax.scatter(df["cl_score"], df["fl_score"], s=8, alpha=0.35, edgecolors="none")
        low = min(float(df["cl_score"].min()), float(df["fl_score"].min()))
        high = max(float(df["cl_score"].max()), float(df["fl_score"].max()))
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("CL reliability")
        corr = df["cl_score"].corr(df["fl_score"], method="spearman")
        ax.text(0.04, 0.92, f"Spearman={corr:.2f}", transform=ax.transAxes)
    axes[0].set_ylabel("FL reliability")
    fig.suptitle("Point-wise CL-vs-FL Reliability")
    save_figure(fig, output_dir / "transfer_scatter_examples")


def plot_native_focus_figures(
    surrogate: pd.DataFrame,
    best: pd.DataFrame,
    full_dir: Path,
    output_dir: Path,
) -> None:
    plot_native_optimizer_r2(surrogate, output_dir)
    plot_native_auroc(best, output_dir)
    plot_native_transfer_scatter(full_dir, output_dir)


def plot_native_optimizer_r2(surrogate: pd.DataFrame, output_dir: Path) -> None:
    native = surrogate[surrogate["scenario"].eq("native")].set_index("optimizer")
    optimizers = ["fedavg", "fedprox", "fedadam"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    x = np.arange(len(optimizers))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(
        x,
        [native.loc[opt, "r2_mean"] for opt in optimizers],
        yerr=[native.loc[opt, "r2_std"] for opt in optimizers],
        color=colors,
        capsize=3,
    )
    ax.set_xticks(x, [optimizer_name(opt) for opt in optimizers])
    ax.set_ylabel("Teacher approximation R2")
    ax.set_title("Multi-hospital Surrogate Approximation")
    ax.axhline(0, color="black", linewidth=0.8)
    save_figure(fig, output_dir / "native_surrogate_optimizer_r2")


def plot_native_auroc(best: pd.DataFrame, output_dir: Path) -> None:
    native = best[best["scenario"].eq("native")].iloc[0]
    pmax_mean = "pmax_fl_auroc_mean" if "pmax_fl_auroc_mean" in best.columns else "pmax_auroc_mean"
    pmax_std = "pmax_fl_auroc_std" if "pmax_fl_auroc_std" in best.columns else "pmax_auroc_std"
    pmax_label = "pmax(FL)" if pmax_mean.startswith("pmax_fl") else "pmax"
    labels = [pmax_label, "Teacher", "Surrogate"]
    means = [
        native[pmax_mean],
        native["teacher_auroc_mean"],
        native["surrogate_auroc_mean"],
    ]
    stds = [
        native[pmax_std],
        native["teacher_auroc_std"],
        native["surrogate_auroc_std"],
    ]
    colors = ["#9ecae9", "#f58518", "#4c78a8"]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar(labels, means, yerr=stds, color=colors, capsize=3)
    ax.set_ylim(0.55, 0.9)
    ax.set_ylabel("Misclassification detection AUROC")
    ax.set_title("Multi-hospital Error Detection")
    save_figure(fig, output_dir / "native_error_detection_auroc")


def plot_native_transfer_scatter(full_dir: Path, output_dir: Path) -> None:
    path = full_dir / "transfer_native" / "transfer_scores_federated_csv_iid.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    specs = [("RHH-kNN", "rhh", "s_knn_global"), ("Embedding", "embedding", "knn")]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharex=True, sharey=True)
    for ax, (title, method, score) in zip(axes, specs, strict=True):
        subset = df[df["method"].eq(method) & df["score"].eq(score)]
        if len(subset) > 3000:
            subset = subset.sample(3000, random_state=42)
        ax.scatter(subset["cl_score"], subset["fl_score"], s=8, alpha=0.35, edgecolors="none")
        low = min(float(subset["cl_score"].min()), float(subset["fl_score"].min()))
        high = max(float(subset["cl_score"].max()), float(subset["fl_score"].max()))
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        corr = subset["cl_score"].corr(subset["fl_score"], method="spearman")
        ax.text(0.04, 0.92, f"Spearman={corr:.2f}", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("CL reliability")
    axes[0].set_ylabel("FL reliability")
    fig.suptitle("Multi-hospital CL-vs-FL Reliability")
    save_figure(fig, output_dir / "native_transfer_scatter")


def optimizer_name(optimizer: str) -> str:
    return {"fedavg": "FedAvg", "fedprox": "FedProx", "fedadam": "FedAdam"}[optimizer]


def method_label(method: str) -> str:
    return {
        "pmax": "pmax",
        "pmax_fl": "pmax(FL)",
        "entropy": "Entropy",
        "entropy_fl": "Entropy(FL)",
        "margin": "Margin",
        "margin_fl": "Margin(FL)",
        "centroid": "Centroid",
        "centroid_fl": "Centroid(FL)",
        "feature_knn": "Feature-kNN",
        "lof": "LOF",
        "teacher": "Teacher",
        "surrogate": "Surrogate",
    }.get(method, method)


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
