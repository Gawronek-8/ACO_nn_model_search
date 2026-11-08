"""Container for the train/validation/test tensors the Evaluator works on.

Loading and preprocessing of the real datasets (Adult, Wine Quality, Letter Recognition) is a separate step
that ends by building a DataSplit. The Colony never sees the data, only the Evaluator does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch


@dataclass(frozen=True)
class DataSplit:
    """Features are float32 of shape (n, n_features), labels are int64 class ids 0 .. n_classes - 1."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            X, y = getattr(self, f"X_{name}"), getattr(self, f"y_{name}")
            if X.ndim != 2:
                raise ValueError(f"X_{name} must be 2D (samples, features), got shape {tuple(X.shape)}")
            if y.ndim != 1 or len(y) != len(X):
                raise ValueError(f"y_{name} must be 1D with one label per row of X_{name}")
            if X.shape[1] != self.X_train.shape[1]:
                raise ValueError(f"X_{name} has {X.shape[1]} features, X_train has {self.X_train.shape[1]}")
            if y.dtype != torch.long:
                raise ValueError(f"y_{name} must be int64 class ids, got {y.dtype}")
            if len(y) and y.min() < 0:
                raise ValueError(f"y_{name} contains negative class ids")

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @cached_property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_val.max(), self.y_test.max())) + 1

    @classmethod
    def from_numpy(cls, X_train, y_train, X_val, y_val, X_test, y_test) -> DataSplit:
        """Builds a split from array-likes, converting to float32 features and int64 labels."""

        def features(a) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.float32))

        def labels(a) -> torch.Tensor:
            return torch.as_tensor(np.asarray(a, dtype=np.int64))

        return cls(features(X_train), labels(y_train), features(X_val), labels(y_val), features(X_test), labels(y_test))

    def to(self, device: torch.device | str) -> DataSplit:
        return DataSplit(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))
