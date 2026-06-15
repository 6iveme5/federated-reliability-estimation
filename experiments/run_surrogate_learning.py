from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from fedrel_journal.config import load_config
from fedrel_journal.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_chuc_federated,
    load_federated_csv_dir,
    load_support2_federated,
)
from fedrel_journal.metrics import error_detection_metrics, risk_at_fraction
from fedrel_journal.surrogate.training import (
    FederatedSurrogateConfig,
    build_hash_teacher_surrogate_clients,
    predict_surrogate,
    train_federated_surrogate,
)
from fedrel_journal.surrogate.model import SurrogateMLP
from fedrel_journal.task import FederatedTaskConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated surrogate reliability learning.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dataset",
        choices=["federated_csv", "chuc", "support2"],
        default="federated_csv",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--chuc-path", type=Path, default=None)
    parser.add_argument("--support2-path", type=Path, default=None)
    parser.add_argument("--partition", choices=["iid", "label_skew", "feature_skew"], default="iid")
    parser.add_argument("--feature-column", default="killip_class")
    parser.add_argument("--n-clients", type=int, default=4)
    parser.add_argument("--optimizers", default="fedavg,fedprox,fedadam")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--surrogate-hidden-dims",
        default="64,32",
        help="Comma-separated hidden dimensions for the surrogate MLP.",
    )
    parser.add_argument(
        "--include-fl-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add pmax_fl/entropy_fl/margin_fl from a FedAvg task classifier.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = experiment_seed(args, config)
    output_dir = args.output_dir or Path(config["outputs"]["root"]) / "surrogate"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args, config, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    protocol = build_hash_teacher_surrogate_clients(
        dataset.clients,
        k=int(config["reliability"]["neighbors"]),
        n_hash=int(config["reliability"]["hash_dim"]),
        seed=seed,
        include_federated_confidence=args.include_fl_confidence,
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

    optimizers = [item.strip().lower() for item in args.optimizers.split(",") if item.strip()]
    hidden_dims = parse_hidden_dims(args.surrogate_hidden_dims)
    rows = []
    histories = {}
    for optimizer_name in optimizers:
        train_config = FederatedSurrogateConfig(
            optimizer=optimizer_name,
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
        model = SurrogateMLP(input_dim=protocol.train_clients[0].x.shape[1], hidden_dims=hidden_dims)
        result = train_federated_surrogate(
            protocol.train_clients,
            eval_clients=protocol.eval_clients,
            config=train_config,
            model=model,
        )
        histories[optimizer_name] = result.history
        rows.append(evaluation_row(args, optimizer_name, result.model, protocol.eval_clients, device))

        model_path = output_dir / f"surrogate_{args.dataset}_{args.partition}_{optimizer_name}.pt"
        torch.save(
            {
                "model_state_dict": result.model.state_dict(),
                "input_dim": int(protocol.train_clients[0].x.shape[1]),
                "hidden_dims": list(hidden_dims),
                "optimizer": optimizer_name,
                "dataset": args.dataset,
                "partition": args.partition,
            },
            model_path,
        )
        del result
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"surrogate_metrics_{args.dataset}_{args.partition}.csv"
    history_path = output_dir / f"surrogate_history_{args.dataset}_{args.partition}.json"
    df.to_csv(csv_path, index=False)
    history_path.write_text(json.dumps(histories, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved surrogate metrics to: {csv_path}")
    print(f"Saved round histories to: {history_path}")


def experiment_seed(args: argparse.Namespace, config: dict) -> int:
    return int(args.seed if args.seed is not None else config["experiment"]["seed"])


def parse_hidden_dims(value: str) -> tuple[int, int]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(dims) != 2:
        raise ValueError("--surrogate-hidden-dims must contain exactly two integers")
    return dims


def load_dataset(args: argparse.Namespace, config: dict, seed: int):
    if args.dataset == "chuc":
        chuc_path = args.chuc_path or Path(config["data"]["chuc_path"])
        return load_chuc_federated(
            chuc_path,
            n_clients=args.n_clients,
            strategy=args.partition,
            feature_column=args.feature_column,
            seed=seed,
        )
    if args.dataset == "support2":
        support2_path = args.support2_path or Path(config["data"]["support2_path"])
        return load_support2_federated(
            support2_path,
            n_clients=args.n_clients,
            strategy=args.partition,
            feature_column=args.feature_column,
            seed=seed,
        )

    data_dir = args.data_dir or Path(config["data"]["federated_dir"])
    return load_federated_csv_dir(
        data_dir,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )


def evaluation_row(args, optimizer_name, model, surrogate_clients, device):
    x_all = np.vstack([client.x for client in surrogate_clients]).astype(np.float32)
    teacher = np.concatenate([client.teacher_reliability for client in surrogate_clients])
    errors = np.concatenate(
        [client.errors for client in surrogate_clients if client.errors is not None]
    )
    surrogate = predict_surrogate(model, x_all, device=device)

    from fedrel_journal.metrics import approximation_metrics

    approx = approximation_metrics(teacher, surrogate)
    method_scores = collect_method_scores(surrogate_clients)
    method_errors = collect_method_errors(surrogate_clients)
    method_scores["teacher"] = teacher
    method_scores["surrogate"] = surrogate

    row = {
        "dataset": args.dataset,
        "partition": args.partition if args.dataset in {"chuc", "support2"} else "native",
        "n_clients": args.n_clients if args.dataset in {"chuc", "support2"} else "",
        "seed": args.seed if args.seed is not None else "",
        "optimizer": optimizer_name,
        "pearson": approx["pearson"],
        "spearman": approx["spearman"],
        "mae": approx["mae"],
        "rmse": approx["rmse"],
        "r2": approx["r2"],
    }
    row.update(task_classifier_metrics(surrogate_clients))
    for method, scores in method_scores.items():
        method_error_labels = method_errors.get(method, errors)
        detection = error_detection_metrics(method_error_labels, scores)
        row[f"{method}_auroc"] = detection["auroc"]
        row[f"{method}_auprc"] = detection["auprc"]
        for percentage in [1, 5, 10]:
            row[f"{method}_risk_at_{percentage}"] = risk_at_fraction(
                method_error_labels,
                scores,
                fraction=percentage / 100.0,
            )
    return row


def task_classifier_metrics(surrogate_clients) -> dict[str, float]:
    y_true = np.concatenate(
        [client.task_y_true for client in surrogate_clients if client.task_y_true is not None]
    )
    pred = np.concatenate(
        [client.task_pred for client in surrogate_clients if client.task_pred is not None]
    )
    proba = np.vstack(
        [client.task_proba for client in surrogate_clients if client.task_proba is not None]
    )
    proba_pos = positive_class_probability(proba)
    task_auroc = float("nan")
    if len(np.unique(y_true)) >= 2:
        task_auroc = float(roc_auc_score(y_true, proba_pos))
    return {
        "task_auroc": task_auroc,
        "task_accuracy": float(accuracy_score(y_true, pred)),
        "task_f1": float(f1_score(y_true, pred, zero_division=0)),
        "task_error_rate": float(np.mean(pred != y_true)),
    }


def positive_class_probability(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def collect_method_scores(surrogate_clients) -> dict[str, np.ndarray]:
    baseline_names = sorted(
        {
            name
            for client in surrogate_clients
            for name in client.baselines
        }
    )
    return {
        name: np.concatenate([client.baselines[name] for client in surrogate_clients])
        for name in baseline_names
    }


def collect_method_errors(surrogate_clients) -> dict[str, np.ndarray]:
    error_names = sorted(
        {
            name
            for client in surrogate_clients
            for name in client.baseline_errors
        }
    )
    return {
        name: np.concatenate([client.baseline_errors[name] for client in surrogate_clients])
        for name in error_names
    }


if __name__ == "__main__":
    main()
