from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedrel_journal.config import load_config
from fedrel_journal.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_chuc_federated,
    load_federated_csv_dir,
)
from fedrel_journal.metrics import approximation_metrics, error_detection_metrics, risk_at_fraction
from fedrel_journal.surrogate.model import SurrogateMLP
from fedrel_journal.surrogate.training import (
    FederatedSurrogateConfig,
    build_hash_teacher_surrogate_clients,
    predict_surrogate,
    train_federated_surrogate,
)
from fedrel_journal.task import FederatedTaskConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    dataset: str
    partition: str = "iid"
    feature_column: str | None = None
    n_clients: int = 4
    optimizer: str = "fedavg"


SCENARIOS = {
    "native": Scenario(name="native", dataset="federated_csv", partition="native"),
    "chuc_iid": Scenario(name="chuc_iid", dataset="chuc", partition="iid"),
    "chuc_label_skew": Scenario(
        name="chuc_label_skew",
        dataset="chuc",
        partition="label_skew",
        optimizer="fedprox",
    ),
    "chuc_killip_skew": Scenario(
        name="chuc_killip_skew",
        dataset="chuc",
        partition="feature_skew",
        feature_column="killip_class",
        optimizer="fedprox",
    ),
    "chuc_st_skew": Scenario(
        name="chuc_st_skew",
        dataset="chuc",
        partition="feature_skew",
        feature_column="st_segment_elevation",
        optimizer="fedprox",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run surrogate hash/model-size ablations.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--hash-dims", default="64,128,256")
    parser.add_argument("--model-sizes", default="small:32,16;current:64,32")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    scenarios = [SCENARIOS[name.strip()] for name in args.scenarios.split(",") if name.strip()]
    hash_dims = [int(item.strip()) for item in args.hash_dims.split(",") if item.strip()]
    model_sizes = parse_model_sizes(args.model_sizes)
    current_hidden = dict(model_sizes).get("current", (64, 32))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for scenario in scenarios:
        for seed in seeds:
            dataset = load_dataset(scenario, config, seed)
            client_cache = {}
            for n_hash in hash_dims:
                clients = get_surrogate_clients(
                    cache=client_cache,
                    dataset=dataset,
                    config=config,
                    n_hash=n_hash,
                    seed=seed,
                    device=device,
                )
                rows.append(
                    run_one_setting(
                        scenario=scenario,
                        seed=seed,
                        n_hash=n_hash,
                        hidden_dims=current_hidden,
                        setting=f"hash_{n_hash}",
                        ablation="hash_dim",
                        clients=clients,
                        config=config,
                        device=device,
                    )
                )

            clients = get_surrogate_clients(
                cache=client_cache,
                dataset=dataset,
                config=config,
                n_hash=int(config["reliability"]["hash_dim"]),
                seed=seed,
                device=device,
            )
            for size_name, hidden_dims in model_sizes:
                rows.append(
                    run_one_setting(
                        scenario=scenario,
                        seed=seed,
                        n_hash=int(config["reliability"]["hash_dim"]),
                        hidden_dims=hidden_dims,
                        setting=f"model_{size_name}",
                        ablation="model_size",
                        clients=clients,
                        config=config,
                        device=device,
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "surrogate_ablation_metrics.csv", index=False)
    summary = summarize_ablation(df)
    summary.to_csv(args.output_dir / "surrogate_ablation_summary.csv", index=False)
    write_markdown(args.output_dir, summary)
    print(summary.to_string(index=False))


def parse_model_sizes(value: str) -> list[tuple[str, tuple[int, int]]]:
    sizes = []
    for item in value.split(";"):
        if not item.strip():
            continue
        name, dims_text = item.split(":", maxsplit=1)
        dims = tuple(int(dim.strip()) for dim in dims_text.split(",") if dim.strip())
        if len(dims) != 2:
            raise ValueError(f"Model size {item!r} must have exactly two dimensions")
        sizes.append((name.strip(), dims))
    return sizes


def load_dataset(scenario: Scenario, config: dict, seed: int):
    if scenario.dataset == "chuc":
        return load_chuc_federated(
            Path(config["data"]["chuc_path"]),
            n_clients=scenario.n_clients,
            strategy=scenario.partition,
            feature_column=scenario.feature_column,
            seed=seed,
        )
    return load_federated_csv_dir(
        Path(config["data"]["federated_dir"]),
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )


def get_surrogate_clients(
    cache: dict[int, list],
    dataset,
    config: dict,
    n_hash: int,
    seed: int,
    device: str,
):
    if n_hash not in cache:
        cache[n_hash] = build_hash_teacher_surrogate_clients(
            dataset.clients,
            k=int(config["reliability"]["neighbors"]),
            n_hash=n_hash,
            seed=seed,
            include_federated_confidence=True,
            federated_task_config=FederatedTaskConfig(
                rounds=int(config["federated"]["rounds"]),
                local_epochs=int(config["federated"]["local_epochs"]),
                batch_size=int(config["federated"]["batch_size"]),
                learning_rate=float(config["federated"]["learning_rate"]),
                weight_decay=float(config["federated"]["weight_decay"]),
                seed=seed,
                device=device,
            ),
        )
    return cache[n_hash]


def run_one_setting(
    scenario: Scenario,
    seed: int,
    n_hash: int,
    hidden_dims: tuple[int, int],
    setting: str,
    ablation: str,
    clients,
    config: dict,
    device: str,
) -> dict[str, float | int | str]:
    train_config = FederatedSurrogateConfig(
        optimizer=scenario.optimizer,
        rounds=int(config["federated"]["rounds"]),
        local_epochs=int(config["federated"]["local_epochs"]),
        batch_size=int(config["federated"]["batch_size"]),
        learning_rate=float(config["federated"]["learning_rate"]),
        weight_decay=float(config["federated"]["weight_decay"]),
        prox_mu=float(config["federated"].get("prox_mu", 0.01)),
        server_learning_rate=float(config["federated"].get("server_learning_rate", 0.01)),
        server_beta1=float(config["federated"].get("server_beta1", 0.9)),
        server_beta2=float(config["federated"].get("server_beta2", 0.99)),
        server_tau=float(config["federated"].get("server_tau", 1e-3)),
        seed=seed,
        device=device,
    )
    model = SurrogateMLP(input_dim=clients[0].x.shape[1], hidden_dims=hidden_dims)
    result = train_federated_surrogate(clients, config=train_config, model=model)
    row = evaluate_setting(result.model, clients, device)
    row.update(
        {
            "scenario": scenario.name,
            "dataset": scenario.dataset,
            "partition_detail": partition_detail(scenario),
            "seed": seed,
            "optimizer": scenario.optimizer,
            "ablation": ablation,
            "setting": setting,
            "n_hash": n_hash,
            "hidden_dims": f"{hidden_dims[0]},{hidden_dims[1]}",
        }
    )
    return row


def evaluate_setting(model, clients, device: str) -> dict[str, float]:
    x_all = np.vstack([client.x for client in clients]).astype(np.float32)
    teacher = np.concatenate([client.teacher_reliability for client in clients])
    surrogate = predict_surrogate(model, x_all, device=device)
    local_errors = np.concatenate([client.errors for client in clients if client.errors is not None])
    method_scores = collect_method_scores(clients)
    method_errors = collect_method_errors(clients)
    method_scores["teacher"] = teacher
    method_scores["surrogate"] = surrogate

    row = approximation_metrics(teacher, surrogate)
    for method, scores in method_scores.items():
        errors = method_errors.get(method, local_errors)
        detection = error_detection_metrics(errors, scores)
        row[f"{method}_auroc"] = detection["auroc"]
        row[f"{method}_auprc"] = detection["auprc"]
        row[f"{method}_risk_at_5"] = risk_at_fraction(errors, scores, fraction=0.05)
    return row


def collect_method_scores(clients) -> dict[str, np.ndarray]:
    names = sorted({name for client in clients for name in client.baselines})
    return {
        name: np.concatenate([client.baselines[name] for client in clients])
        for name in names
    }


def collect_method_errors(clients) -> dict[str, np.ndarray]:
    names = sorted({name for client in clients for name in client.baseline_errors})
    return {
        name: np.concatenate([client.baseline_errors[name] for client in clients])
        for name in names
    }


def summarize_ablation(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pearson",
        "spearman",
        "mae",
        "rmse",
        "r2",
        "surrogate_auroc",
        "teacher_auroc",
        "pmax_fl_auroc",
        "centroid_fl_auroc",
    ]
    grouped = df.groupby(
        ["scenario", "partition_detail", "optimizer", "ablation", "setting", "n_hash", "hidden_dims"],
        dropna=False,
    )[metric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().iloc[:, :1].rename(columns={metric_cols[0]: "n_seeds"})
    return pd.concat([mean, std, count], axis=1).reset_index()


def write_markdown(output_dir: Path, summary: pd.DataFrame) -> None:
    sections = ["# Surrogate Ablation Summary\n"]
    sections.append(summary.to_markdown(index=False, floatfmt=".4f"))
    (output_dir / "SURROGATE_ABLATION_SUMMARY.md").write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def partition_detail(scenario: Scenario) -> str:
    if scenario.dataset == "federated_csv":
        return "native"
    if scenario.feature_column is None:
        return scenario.partition
    return f"{scenario.partition}_{scenario.feature_column}"


if __name__ == "__main__":
    main()
