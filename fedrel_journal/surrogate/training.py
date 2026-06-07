from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedrel_journal.data import ClientDataset
from fedrel_journal.federated.fedavg import fedavg_state_dicts
from fedrel_journal.methods.baselines import (
    centroid_reliability,
    confidence_baselines,
    feature_knn_classaware_reliability,
    lof_reliability,
)
from fedrel_journal.methods.hash_teacher import classaware_hash_reliability
from fedrel_journal.metrics import approximation_metrics, minmax01
from fedrel_journal.surrogate.model import SurrogateMLP
from fedrel_journal.task import (
    FederatedTaskConfig,
    TaskClientSplit,
    predict_binary_classifier,
    train_federated_binary_classifier,
)

FederatedOptimizer = Literal["fedavg", "fedprox", "fedadam"]


@dataclass
class SurrogateClientDataset:
    client_id: int
    name: str
    x: np.ndarray
    teacher_reliability: np.ndarray
    errors: np.ndarray | None = None
    baselines: dict[str, np.ndarray] = field(default_factory=dict)
    baseline_errors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SurrogateProtocolDatasets:
    train_clients: list[SurrogateClientDataset]
    eval_clients: list[SurrogateClientDataset]


@dataclass
class FederatedSurrogateConfig:
    optimizer: FederatedOptimizer = "fedavg"
    rounds: int = 20
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    prox_mu: float = 1e-2
    server_learning_rate: float = 0.01
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass
class FedAdamState:
    m: dict[str, torch.Tensor] = field(default_factory=dict)
    v: dict[str, torch.Tensor] = field(default_factory=dict)
    t: int = 0


@dataclass
class FederatedSurrogateResult:
    model: nn.Module
    history: list[dict[str, float]]
    optimizer: str
    n_clients: int
    n_samples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_hash_teacher_surrogate_clients(
    clients: list[ClientDataset],
    validation_fraction: float = 0.2,
    surrogate_train_fraction: float = 0.5,
    k: int = 10,
    n_hash: int = 128,
    seed: int = 42,
    include_federated_confidence: bool = False,
    federated_task_config: FederatedTaskConfig | None = None,
) -> SurrogateProtocolDatasets:
    if not 0.0 < surrogate_train_fraction < 1.0:
        raise ValueError("surrogate_train_fraction must be between 0 and 1")
    splits = [
        _client_split(
            client,
            validation_fraction=validation_fraction,
            seed=seed + client.client_id,
        )
        for client in clients
    ]
    if include_federated_confidence:
        config = federated_task_config or FederatedTaskConfig(seed=seed)
        task_model = train_federated_binary_classifier(splits, config=config)
    else:
        config = None

    split_queries: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for split in splits:
        x_train, x_holdout, y_train, y_holdout = (
            split.x_train,
            split.x_val,
            split.y_train,
            split.y_val,
        )
        x_surrogate_train, x_surrogate_eval, y_surrogate_train, y_surrogate_eval = _train_validation_split(
            x_holdout,
            y_holdout,
            validation_fraction=1.0 - surrogate_train_fraction,
            seed=seed + 10_000 + split.client_id,
        )
        pred_train, proba_train = fit_predict_local_task_model(x_train, y_train, x_surrogate_train)
        pred_eval, proba_eval = fit_predict_local_task_model(x_train, y_train, x_surrogate_eval)
        split_queries[split.client_id] = {
            "train": (x_surrogate_train, y_surrogate_train, pred_train, proba_train),
            "eval": (x_surrogate_eval, y_surrogate_eval, pred_eval, proba_eval),
        }

    train_centroid_fl = {}
    eval_centroid_fl = {}
    train_fl_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    eval_fl_outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if include_federated_confidence:
        assert config is not None
        for split in splits:
            x_surrogate_train, _, _, _ = split_queries[split.client_id]["train"]
            x_surrogate_eval, _, _, _ = split_queries[split.client_id]["eval"]
            train_fl_outputs[split.client_id] = predict_binary_classifier(
                task_model,
                x_surrogate_train,
                device=config.device,
            )
            eval_fl_outputs[split.client_id] = predict_binary_classifier(
                task_model,
                x_surrogate_eval,
                device=config.device,
            )
        train_centroid_fl = federated_centroid_reliability(
            splits,
            {
                client_id: (split_queries[client_id]["train"][0], pred)
                for client_id, (pred, _) in train_fl_outputs.items()
            },
        )
        eval_centroid_fl = federated_centroid_reliability(
            splits,
            {
                client_id: (split_queries[client_id]["eval"][0], pred)
                for client_id, (pred, _) in eval_fl_outputs.items()
            },
        )

    train_clients = []
    eval_clients = []
    for split in splits:
        x_surrogate_train, y_surrogate_train, pred_train, proba_train = split_queries[split.client_id]["train"]
        x_surrogate_eval, y_surrogate_eval, pred_eval, proba_eval = split_queries[split.client_id]["eval"]
        teacher_train = classaware_hash_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_train,
            pred_train,
            k=k,
            n_hash=n_hash,
            seed=seed,
        )
        teacher_eval = classaware_hash_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_eval,
            pred_eval,
            k=k,
            n_hash=n_hash,
            seed=seed,
        )
        train_baselines = confidence_baselines(proba_train)
        train_baselines["centroid"] = centroid_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_train,
            pred_train,
        )
        train_baselines["feature_knn"] = feature_knn_classaware_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_train,
            pred_train,
            k=k,
        )
        train_baselines["lof"] = lof_reliability(split.x_train, x_surrogate_train, k=k)

        eval_baselines = confidence_baselines(proba_eval)
        eval_baselines["centroid"] = centroid_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_eval,
            pred_eval,
        )
        eval_baselines["feature_knn"] = feature_knn_classaware_reliability(
            split.x_train,
            split.y_train,
            x_surrogate_eval,
            pred_eval,
            k=k,
        )
        eval_baselines["lof"] = lof_reliability(split.x_train, x_surrogate_eval, k=k)

        train_baseline_errors = {}
        eval_baseline_errors = {}
        if include_federated_confidence:
            pred_fl_train, proba_fl_train = train_fl_outputs[split.client_id]
            pred_fl_eval, proba_fl_eval = eval_fl_outputs[split.client_id]
            for name, values in confidence_baselines(proba_fl_train).items():
                baseline_name = f"{name}_fl"
                train_baselines[baseline_name] = values
                train_baseline_errors[baseline_name] = (pred_fl_train != y_surrogate_train).astype(int)
            for name, values in confidence_baselines(proba_fl_eval).items():
                baseline_name = f"{name}_fl"
                eval_baselines[baseline_name] = values
                eval_baseline_errors[baseline_name] = (pred_fl_eval != y_surrogate_eval).astype(int)
            train_baselines["centroid_fl"] = train_centroid_fl[split.client_id]
            train_baseline_errors["centroid_fl"] = (pred_fl_train != y_surrogate_train).astype(int)
            eval_baselines["centroid_fl"] = eval_centroid_fl[split.client_id]
            eval_baseline_errors["centroid_fl"] = (pred_fl_eval != y_surrogate_eval).astype(int)

        train_clients.append(
            SurrogateClientDataset(
                client_id=split.client_id,
                name=split.name,
                x=x_surrogate_train.astype(np.float32),
                teacher_reliability=teacher_train.astype(np.float32),
                errors=(pred_train != y_surrogate_train).astype(int),
                baselines={key: value.astype(np.float32) for key, value in train_baselines.items()},
                baseline_errors={
                    key: value.astype(int) for key, value in train_baseline_errors.items()
                },
            )
        )
        eval_clients.append(
            SurrogateClientDataset(
                client_id=split.client_id,
                name=split.name,
                x=x_surrogate_eval.astype(np.float32),
                teacher_reliability=teacher_eval.astype(np.float32),
                errors=(pred_eval != y_surrogate_eval).astype(int),
                baselines={key: value.astype(np.float32) for key, value in eval_baselines.items()},
                baseline_errors={
                    key: value.astype(int) for key, value in eval_baseline_errors.items()
                },
            )
        )
    return SurrogateProtocolDatasets(train_clients=train_clients, eval_clients=eval_clients)


def fit_predict_local_task_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y_train)
    if len(classes) < 2:
        pred = np.full(len(x_query), classes[0], dtype=int)
        proba = np.ones((len(x_query), 1), dtype=np.float32)
        return pred, proba

    classifier = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_query)
    proba = classifier.predict_proba(x_query)
    return pred, proba


def train_federated_surrogate(
    surrogate_clients: list[SurrogateClientDataset],
    eval_clients: list[SurrogateClientDataset] | None = None,
    input_dim: int | None = None,
    config: FederatedSurrogateConfig | None = None,
    model: nn.Module | None = None,
) -> FederatedSurrogateResult:
    if not surrogate_clients:
        raise ValueError("surrogate_clients must not be empty")
    config = config or FederatedSurrogateConfig()
    if config.optimizer not in {"fedavg", "fedprox", "fedadam"}:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    set_seed(config.seed)
    input_dim = input_dim or surrogate_clients[0].x.shape[1]
    global_model = model or SurrogateMLP(input_dim=input_dim)
    global_model = global_model.to(config.device)
    fedadam_state = FedAdamState()
    history: list[dict[str, float]] = []

    for round_idx in range(1, config.rounds + 1):
        local_states = []
        local_weights = []
        global_state = copy.deepcopy(global_model.state_dict())
        for client in surrogate_clients:
            local_state = local_train_one_round(
                global_model,
                client.x,
                client.teacher_reliability,
                config,
                global_state=global_state,
            )
            local_states.append(local_state)
            local_weights.append(len(client.x))

        averaged_state = fedavg_state_dicts(local_states, local_weights)
        if config.optimizer == "fedadam":
            next_state = fedadam_update(
                current_state=global_model.state_dict(),
                averaged_state=averaged_state,
                state=fedadam_state,
                server_lr=config.server_learning_rate,
                beta1=config.server_beta1,
                beta2=config.server_beta2,
                tau=config.server_tau,
            )
        else:
            next_state = averaged_state
        global_model.load_state_dict(next_state)

        round_metrics = evaluate_surrogate_approximation(
            global_model,
            eval_clients or surrogate_clients,
            config.device,
        )
        round_metrics["round"] = float(round_idx)
        history.append(round_metrics)

    return FederatedSurrogateResult(
        model=global_model,
        history=history,
        optimizer=config.optimizer,
        n_clients=len(surrogate_clients),
        n_samples=sum(len(client.x) for client in surrogate_clients),
    )


def local_train_one_round(
    global_model: nn.Module,
    x_local: np.ndarray,
    y_local: np.ndarray,
    config: FederatedSurrogateConfig,
    global_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    local_model = copy.deepcopy(global_model).to(config.device)
    local_model.train()
    global_float_params = {
        name: value.detach().clone().to(config.device)
        for name, value in global_state.items()
        if torch.is_floating_point(value)
    }

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(x_local, dtype=torch.float32),
        torch.tensor(y_local, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for _ in range(config.local_epochs):
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()
            pred = local_model(xb)
            loss = criterion(pred, yb)
            if config.optimizer == "fedprox":
                loss = loss + fedprox_penalty(local_model, global_float_params, config.prox_mu)
            loss.backward()
            optimizer.step()

    return {key: value.detach().cpu() for key, value in local_model.state_dict().items()}


def fedprox_penalty(
    model: nn.Module,
    global_float_params: dict[str, torch.Tensor],
    prox_mu: float,
) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, local_value in model.named_parameters():
        if name not in global_float_params:
            continue
        global_value = global_float_params[name]
        penalty = penalty + torch.sum((local_value - global_value) ** 2)
    return 0.5 * prox_mu * penalty


def fedadam_update(
    current_state: dict[str, torch.Tensor],
    averaged_state: dict[str, torch.Tensor],
    state: FedAdamState,
    server_lr: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
) -> dict[str, torch.Tensor]:
    state.t += 1
    next_state = copy.deepcopy(current_state)
    for key, current_value in current_state.items():
        if not torch.is_floating_point(current_value):
            next_state[key] = averaged_state[key]
            continue

        pseudo_grad = current_value.detach().cpu() - averaged_state[key].detach().cpu()
        if key not in state.m:
            state.m[key] = torch.zeros_like(pseudo_grad)
            state.v[key] = torch.zeros_like(pseudo_grad)
        state.m[key] = beta1 * state.m[key] + (1.0 - beta1) * pseudo_grad
        state.v[key] = beta2 * state.v[key] + (1.0 - beta2) * (pseudo_grad**2)
        m_hat = state.m[key] / (1.0 - beta1**state.t)
        v_hat = state.v[key] / (1.0 - beta2**state.t)
        next_state[key] = (
            current_value.detach().cpu() - server_lr * m_hat / (torch.sqrt(v_hat) + tau)
        )
    return next_state


def predict_surrogate(model: nn.Module, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        return model(x_tensor).detach().cpu().numpy()


def evaluate_surrogate_approximation(
    model: nn.Module,
    surrogate_clients: list[SurrogateClientDataset],
    device: str = "cpu",
) -> dict[str, float]:
    x_all = np.vstack([client.x for client in surrogate_clients]).astype(np.float32)
    y_all = np.concatenate([client.teacher_reliability for client in surrogate_clients]).astype(
        np.float32
    )
    pred_all = predict_surrogate(model, x_all, device=device)
    return approximation_metrics(y_all, pred_all)


def _train_validation_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, counts = np.unique(y, return_counts=True)
    stratify = y if len(counts) > 1 and np.min(counts) >= 2 else None
    return train_test_split(
        x,
        y,
        test_size=validation_fraction,
        stratify=stratify,
        random_state=seed,
    )


def _client_split(
    client: ClientDataset,
    validation_fraction: float,
    seed: int,
) -> TaskClientSplit:
    x_train, x_val, y_train, y_val = _train_validation_split(
        client.x,
        client.y,
        validation_fraction,
        seed,
    )
    return TaskClientSplit(
        client_id=client.client_id,
        name=client.name,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )


def federated_centroid_reliability(
    splits: list[TaskClientSplit],
    federated_predictions: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, np.ndarray]:
    class_sums: dict[int, np.ndarray] = {}
    class_counts: dict[int, int] = {}
    for split in splits:
        for label in np.unique(split.y_train):
            label_int = int(label)
            x_label = split.x_train[split.y_train == label]
            if len(x_label) == 0:
                continue
            class_sums[label_int] = class_sums.get(label_int, np.zeros(split.x_train.shape[1])) + (
                x_label.sum(axis=0)
            )
            class_counts[label_int] = class_counts.get(label_int, 0) + len(x_label)

    centroids = {
        label: class_sums[label] / max(class_counts[label], 1)
        for label in class_sums
    }
    x_global = np.vstack([split.x_train for split in splits])
    fallback = float(np.linalg.norm(x_global - x_global.mean(axis=0), axis=1).mean())

    distances_by_client: dict[int, np.ndarray] = {}
    all_distances = []
    for split in splits:
        x_query, pred = federated_predictions[split.client_id]
        distances = []
        for x_i, pred_i in zip(x_query, pred, strict=True):
            centroid = centroids.get(int(pred_i))
            distance = fallback if centroid is None else float(np.linalg.norm(x_i - centroid))
            distances.append(distance)
        distances_arr = np.asarray(distances, dtype=np.float64)
        distances_by_client[split.client_id] = distances_arr
        all_distances.append(distances_arr)

    all_scores = minmax01(-np.concatenate(all_distances)).astype(np.float32)
    scores_by_client = {}
    offset = 0
    for split in splits:
        n_items = len(distances_by_client[split.client_id])
        scores_by_client[split.client_id] = all_scores[offset : offset + n_items]
        offset += n_items
    return scores_by_client
