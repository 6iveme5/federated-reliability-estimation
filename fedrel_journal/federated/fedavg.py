from __future__ import annotations

import copy

import torch


def fedavg_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[int] | list[float],
) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("FedAvg weights must sum to a positive value")

    averaged = copy.deepcopy(state_dicts[0])
    for key in averaged:
        averaged[key] = sum(
            state[key] * (float(weight) / total_weight)
            for state, weight in zip(state_dicts, weights, strict=True)
        )
    return averaged
