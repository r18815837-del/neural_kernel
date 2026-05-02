from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *arrays):
        if len(arrays) == 0:
            raise ValueError("TensorDataset requires at least one array")

        length = len(arrays[0])
        for arr in arrays:
            if len(arr) != length:
                raise ValueError("All arrays must have the same length")

        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays[0])

    def __getitem__(self, index):
        items = tuple(arr[index] for arr in self.arrays)
        if len(items) == 1:
            return items[0]
        return items