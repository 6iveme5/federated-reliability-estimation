from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedrel_journal.federated.fedavg import fedavg_state_dicts


@dataclass
class TaskClientSplit:
    client_id: int
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


@dataclass
class FederatedTaskConfig:
    rounds: int = 20
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"


class BinaryTaskModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_federated_binary_classifier(
    splits: list[TaskClientSplit],
    config: FederatedTaskConfig | None = None,
) -> BinaryTaskModel:
    if not splits:
        raise ValueError("splits must not be empty")
    config = config or FederatedTaskConfig()
    _set_seed(config.seed)

    input_dim = splits[0].x_train.shape[1]
    global_model = BinaryTaskModel(input_dim=input_dim).to(config.device)

    for round_idx in range(config.rounds):
        local_states = []
        local_weights = []
        global_state = copy.deepcopy(global_model.state_dict())
        for split in splits:
            local_state = _train_binary_classifier_one_round(
                global_model=global_model,
                global_state=global_state,
                x_local=split.x_train,
                y_local=split.y_train,
                config=config,
                round_idx=round_idx,
                client_id=split.client_id,
            )
            local_states.append(local_state)
            local_weights.append(len(split.x_train))
        global_model.load_state_dict(fedavg_state_dicts(local_states, local_weights))

    return global_model


def predict_binary_classifier(
    model: nn.Module,
    x: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        prob_pos = torch.sigmoid(model(x_tensor)).detach().cpu().numpy()
    prob_pos = np.asarray(prob_pos, dtype=np.float32).reshape(-1)
    proba = np.column_stack([1.0 - prob_pos, prob_pos]).astype(np.float32)
    pred = (prob_pos >= 0.5).astype(int)
    return pred, proba


def _train_binary_classifier_one_round(
    global_model: nn.Module,
    global_state: dict[str, torch.Tensor],
    x_local: np.ndarray,
    y_local: np.ndarray,
    config: FederatedTaskConfig,
    round_idx: int,
    client_id: int,
) -> dict[str, torch.Tensor]:
    local_model = copy.deepcopy(global_model).to(config.device)
    local_model.load_state_dict(global_state)
    local_model.train()

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    y_float = np.asarray(y_local, dtype=np.float32)
    pos = float(y_float.sum())
    neg = float(len(y_float) - pos)
    pos_weight = torch.tensor(
        neg / max(pos, 1.0),
        dtype=torch.float32,
        device=config.device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator()
    generator.manual_seed(config.seed + round_idx * 1009 + client_id)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_local, dtype=torch.float32),
            torch.tensor(y_float, dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    for _ in range(config.local_epochs):
        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()
            loss = criterion(local_model(xb), yb)
            loss.backward()
            optimizer.step()

    return {key: value.detach().cpu() for key, value in local_model.state_dict().items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
