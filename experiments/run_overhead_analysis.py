from __future__ import annotations

import argparse
import json
import time
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
    load_support2_federated,
)
from fedrel_journal.overhead import model_nbytes
from fedrel_journal.surrogate.training import (
    FederatedSurrogateConfig,
    build_hash_teacher_surrogate_clients,
    predict_surrogate,
    train_federated_surrogate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure reliability surrogate overhead.")
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
    parser.add_argument("--optimizer", choices=["fedavg", "fedprox", "fedadam"], default="fedavg")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat-inference", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    output_dir = args.output_dir or Path(config["outputs"]["root"]) / "overhead"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args, config, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_start = time.perf_counter()
    surrogate_clients = build_hash_teacher_surrogate_clients(
        dataset.clients,
        k=int(config["reliability"]["neighbors"]),
        n_hash=int(config["reliability"]["hash_dim"]),
        seed=seed,
    )
    teacher_time = time.perf_counter() - teacher_start

    train_config = FederatedSurrogateConfig(
        optimizer=args.optimizer,
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

    train_start = time.perf_counter()
    result = train_federated_surrogate(surrogate_clients, config=train_config)
    train_time = time.perf_counter() - train_start

    x_all = np.vstack([client.x for client in surrogate_clients]).astype(np.float32)
    inference_times = []
    for _ in range(args.repeat_inference):
        start = time.perf_counter()
        predict_surrogate(result.model, x_all, device=device)
        inference_times.append(time.perf_counter() - start)
    inference_time = float(np.median(inference_times))

    model_bytes = model_nbytes(result.model)
    n_clients = len(surrogate_clients)
    rounds = int(config["federated"]["rounds"])
    total_download_bytes = model_bytes * n_clients * rounds
    total_upload_bytes = model_bytes * n_clients * rounds
    if args.optimizer == "fedadam":
        server_state_bytes = 2 * model_bytes
    else:
        server_state_bytes = 0

    n_teacher_samples = int(sum(len(client.teacher_reliability) for client in surrogate_clients))
    rows = [
        {
            "dataset": args.dataset,
            "partition": args.partition if args.dataset != "federated_csv" else "native",
            "feature_column": args.feature_column if args.dataset != "federated_csv" else "",
            "optimizer": args.optimizer,
            "seed": seed,
            "n_clients": n_clients,
            "n_teacher_samples": n_teacher_samples,
            "teacher_time_sec": teacher_time,
            "teacher_time_per_sample_ms": teacher_time / max(n_teacher_samples, 1) * 1000.0,
            "surrogate_train_time_sec": train_time,
            "surrogate_inference_time_sec": inference_time,
            "surrogate_inference_per_sample_ms": inference_time / max(len(x_all), 1) * 1000.0,
            "model_size_bytes": model_bytes,
            "model_size_mb": model_bytes / (1024.0 * 1024.0),
            "total_upload_mb": total_upload_bytes / (1024.0 * 1024.0),
            "total_download_mb": total_download_bytes / (1024.0 * 1024.0),
            "total_comm_mb": (total_upload_bytes + total_download_bytes) / (1024.0 * 1024.0),
            "server_state_mb": server_state_bytes / (1024.0 * 1024.0),
            "break_even_inference_batches": train_time / max(teacher_time - inference_time, 1e-12),
        }
    ]

    df = pd.DataFrame(rows)
    csv_path = output_dir / overhead_filename(args, seed, ".csv")
    json_path = output_dir / overhead_filename(args, seed, ".json")
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows[0], indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved overhead CSV to: {csv_path}")


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


def overhead_filename(args: argparse.Namespace, seed: int, suffix: str) -> str:
    if args.dataset != "federated_csv" and args.partition == "feature_skew":
        scenario = f"{args.dataset}_{args.feature_column}_skew"
    elif args.dataset != "federated_csv":
        scenario = f"{args.dataset}_{args.partition}"
    else:
        scenario = "native"
    return f"overhead_{scenario}_{args.optimizer}_seed_{seed}{suffix}"


if __name__ == "__main__":
    main()
