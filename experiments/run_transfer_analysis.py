from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fedrel_journal.config import load_config
from fedrel_journal.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_chuc_federated,
    load_federated_csv_dir,
    load_support2_federated,
)
from fedrel_journal.methods.embedding import transfer_embedding_scores
from fedrel_journal.methods.gmm import transfer_gmm_scores
from fedrel_journal.methods.rhh import transfer_rhh_scores
from fedrel_journal.metrics import approximation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CL-to-FL reliability transfer analysis.")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = experiment_seed(args, config)
    output_dir = args.output_dir or Path(config["outputs"]["root"]) / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args, config, seed)
    clients = [client.x for client in dataset.clients]
    x_query = dataset.x_central
    if x_query is None:
        raise ValueError("Loaded dataset did not provide x_central")

    rows = []
    score_rows = []
    gmm_cl, gmm_fl = transfer_gmm_scores(
        clients,
        x_query,
        n_components=int(config.get("reliability", {}).get("gmm_components", 8)),
        seed=seed,
    )
    rows.append(metric_row(args, "gmm", "density", gmm_cl, gmm_fl))
    score_rows.append(score_frame(args, "gmm", "density", gmm_cl, gmm_fl))

    rhh_cl, rhh_fl = transfer_rhh_scores(
        clients,
        x_query,
        bits=int(config["reliability"]["hash_dim"]),
        neighbors=int(config["reliability"]["neighbors"]),
        seed=seed,
    )
    for key in sorted(rhh_cl):
        rows.append(metric_row(args, "rhh", key, rhh_cl[key], rhh_fl[key]))
        score_rows.append(score_frame(args, "rhh", key, rhh_cl[key], rhh_fl[key]))

    emb_cl, emb_fl = transfer_embedding_scores(
        clients,
        x_query,
        embedding_dim=int(config.get("reliability", {}).get("embedding_dim", 8)),
        neighbors=int(config["reliability"]["neighbors"]),
        seed=seed,
    )
    rows.append(metric_row(args, "embedding", "knn", emb_cl, emb_fl))
    score_rows.append(score_frame(args, "embedding", "knn", emb_cl, emb_fl))

    df = pd.DataFrame(rows)
    score_df = pd.concat(score_rows, ignore_index=True)
    csv_path = output_dir / f"transfer_metrics_{args.dataset}_{args.partition}.csv"
    json_path = output_dir / f"transfer_metrics_{args.dataset}_{args.partition}.json"
    score_path = output_dir / f"transfer_scores_{args.dataset}_{args.partition}.csv"
    df.to_csv(csv_path, index=False)
    score_df.to_csv(score_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved transfer metrics to: {csv_path}")
    print(f"Saved transfer scores to: {score_path}")


def experiment_seed(args: argparse.Namespace, config: dict) -> int:
    return int(args.seed if args.seed is not None else config["experiment"]["seed"])


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


def metric_row(
    args: argparse.Namespace,
    method: str,
    score_name: str,
    reference,
    predicted,
) -> dict[str, float | str | int]:
    metrics = approximation_metrics(reference, predicted)
    return {
        "dataset": args.dataset,
        "partition": args.partition if args.dataset in {"chuc", "support2"} else "native",
        "n_clients": args.n_clients if args.dataset in {"chuc", "support2"} else "",
        "seed": args.seed if args.seed is not None else "",
        "method": method,
        "score": score_name,
        **metrics,
    }


def score_frame(
    args: argparse.Namespace,
    method: str,
    score_name: str,
    reference,
    predicted,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": args.dataset,
            "partition": args.partition if args.dataset in {"chuc", "support2"} else "native",
            "n_clients": args.n_clients if args.dataset in {"chuc", "support2"} else "",
            "seed": args.seed if args.seed is not None else "",
            "method": method,
            "score": score_name,
            "sample_index": range(len(reference)),
            "cl_score": reference,
            "fl_score": predicted,
        }
    )


if __name__ == "__main__":
    main()
