from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
from torch import nn


@dataclass
class Timer:
    records: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.records[name] = self.records.get(name, 0.0) + time.perf_counter() - start


def model_nbytes(model: nn.Module) -> int:
    return state_dict_nbytes(model.state_dict())


def state_dict_nbytes(state_dict: dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        total += value.numel() * value.element_size()
    return int(total)


def ndarray_nbytes(*arrays: np.ndarray) -> int:
    return int(sum(array.nbytes for array in arrays))
