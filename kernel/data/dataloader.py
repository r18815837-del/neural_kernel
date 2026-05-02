from __future__ import annotations

import math
import numpy as np

from kernel.data.collate import default_collate


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn=None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch = [self.dataset[int(i)] for i in batch_indices]
            yield self.collate_fn(batch)