from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedrel_journal.config import load_config
from fedrel_journal.data import FEATURE_COLUMNS, TARGET_COLUMN, load_chuc_federated, load_federated_csv_dir
from fedrel_journal.overhead import model_nbytes
from fedrel_journal.surrogate.model import SurrogateMLP


SCENARIOS = [
    ("native", "federated_csv", "native", None),
    ("chuc_iid", "chuc", "iid", None),
    ("chuc_label", "chuc", "label_skew", None),
    ("chuc_killip", "chuc", "feature_skew", "killip_class"),
    ("chuc_st", "chuc", "feature_skew", "st_segment_elevation"),
]

SCENARIO_LABELS = {
    "native": "Multi-hospital",
    "chuc_iid": "CHUC-IID",
    "chuc_label": "CHUC-label",
    "chuc_killip": "CHUC-Killip",
    "chuc_st": "CHUC-ST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reliability communication overhead.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/communication"))
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario_name, dataset_name, partition, feature_column in SCENARIOS:
        dataset = load_dataset(dataset_name, partition, feature_column, config, seed)
        n_clients = len(dataset.clients)
        n_samples = int(sum(len(client.y) for client in dataset.clients))
        input_dim = int(dataset.clients[0].x.shape[1])
        model = SurrogateMLP(input_dim=input_dim)
        model_bytes = model_nbytes(model)
        rounds = int(config["federated"]["rounds"])

        rows.extend(
            zero_comm_rows(
                scenario_name=scenario_name,
                dataset=dataset_name,
                partition=partition,
                n_clients=n_clients,
                n_samples=n_samples,
                input_dim=input_dim,
            )
        )
        for optimizer in ["fedavg", "fedprox", "fedadam"]:
            upload_bytes = model_bytes * n_clients * rounds
            download_bytes = model_bytes * n_clients * rounds
            server_state_bytes = 2 * model_bytes if optimizer == "fedadam" else 0
            rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "dataset": dataset_name,
                    "partition": partition,
                    "method": f"surrogate_{optimizer}",
                    "communication_type": "federated_surrogate_training",
                    "n_clients": n_clients,
                    "n_samples": n_samples,
                    "input_dim": input_dim,
                    "rounds": rounds,
                    "model_size_mb": model_bytes / mb(),
                    "upload_mb": upload_bytes / mb(),
                    "download_mb": download_bytes / mb(),
                    "total_comm_mb": (upload_bytes + download_bytes) / mb(),
                    "per_round_comm_mb": (2 * model_bytes * n_clients) / mb(),
                    "server_state_mb": server_state_bytes / mb(),
                    "deployment_comm_mb": 0.0,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "communication_overhead.csv", index=False)
    write_summary(df, args.output_dir)
    plot_surrogate_comm(df, args.output_dir)
    plot_method_comm_native(df, args.output_dir)

    print(df.to_string(index=False))
    print(f"\nSaved communication analysis to: {args.output_dir}")


def load_dataset(dataset_name: str, partition: str, feature_column: str | None, config: dict, seed: int):
    if dataset_name == "chuc":
        return load_chuc_federated(
            Path(config["data"]["chuc_path"]),
            n_clients=4,
            strategy=partition,
            feature_column=feature_column,
            seed=seed,
        )
    return load_federated_csv_dir(
        Path(config["data"]["federated_dir"]),
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )


def zero_comm_rows(
    scenario_name: str,
    dataset: str,
    partition: str,
    n_clients: int,
    n_samples: int,
    input_dim: int,
) -> list[dict]:
    methods = [
        "pmax",
        "entropy",
        "margin",
        "centroid",
        "feature_knn",
        "lof",
        "hash_teacher_direct",
    ]
    return [
        {
            "scenario": scenario_name,
            "scenario_label": SCENARIO_LABELS[scenario_name],
            "dataset": dataset,
            "partition": partition,
            "method": method,
            "communication_type": "local_or_direct_reliability",
            "n_clients": n_clients,
            "n_samples": n_samples,
            "input_dim": input_dim,
            "rounds": 0,
            "model_size_mb": 0.0,
            "upload_mb": 0.0,
            "download_mb": 0.0,
            "total_comm_mb": 0.0,
            "per_round_comm_mb": 0.0,
            "server_state_mb": 0.0,
            "deployment_comm_mb": 0.0,
        }
        for method in methods
    ]


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    surrogate = df[df["communication_type"].eq("federated_surrogate_training")].copy()
    selected = surrogate[
        surrogate["method"].isin(["surrogate_fedavg", "surrogate_fedprox", "surrogate_fedadam"])
    ]
    lines = [
        "# Communication Overhead Summary\n",
        "Direct confidence, density, kNN, LOF, and hash-teacher scores are treated as local/direct reliability computations with no extra federated training communication. The surrogate rows estimate model upload and download during federated surrogate training.\n",
        "## Federated Surrogate Communication\n",
        selected[
            [
                "scenario_label",
                "method",
                "n_clients",
                "rounds",
                "model_size_mb",
                "upload_mb",
                "download_mb",
                "total_comm_mb",
                "per_round_comm_mb",
                "server_state_mb",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "\n\n## Notes\n",
        "- FedAvg and FedProx have the same communication pattern in this implementation.",
        "- FedAdam has the same client communication, plus server-side optimizer state memory.",
        "- Deployment communication for the trained surrogate is zero under local inference.",
    ]
    (output_dir / "COMMUNICATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def plot_surrogate_comm(df: pd.DataFrame, output_dir: Path) -> None:
    surrogate = df[df["method"].eq("surrogate_fedavg")].copy()
    labels = surrogate["scenario_label"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x, surrogate["total_comm_mb"], color="#4c78a8")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Total communication (MB)")
    ax.set_title("Federated Surrogate Training Communication")
    save_figure(fig, output_dir / "surrogate_training_communication")


def plot_method_comm_native(df: pd.DataFrame, output_dir: Path) -> None:
    native = df[df["scenario"].eq("native")].copy()
    method_order = [
        "pmax",
        "entropy",
        "margin",
        "centroid",
        "feature_knn",
        "lof",
        "hash_teacher_direct",
        "surrogate_fedavg",
    ]
    native = native[native["method"].isin(method_order)]
    native["method"] = pd.Categorical(native["method"], categories=method_order, ordered=True)
    native = native.sort_values("method")

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(native))
    ax.bar(x, native["total_comm_mb"], color="#4c78a8")
    ax.set_xticks(x, native["method"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Total communication (MB)")
    ax.set_title("Multi-hospital Reliability Communication Comparison")
    save_figure(fig, output_dir / "native_method_communication")


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_no_suffix.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(path_no_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def mb() -> float:
    return 1024.0 * 1024.0


if __name__ == "__main__":
    main()
