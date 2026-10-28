"""Training and scoring of the networks built by the ants.

Placeholder only: the real implementation is designed together with the dataset handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .utils import LayerConfig


@dataclass
class EvaluationResult:
    val_accuracy: float
    val_loss: float
    train_time: float
    n_params: int


class Evaluator:
    """Turns an ant's path into an MLP, trains it and scores it on the validation set."""

    def __init__(self):
        pass

    def evaluate_path(self, path: list[LayerConfig], reuse_weights: bool) -> EvaluationResult:
        """Full pipeline for one ant: build (or warm-start), train, score, store weights."""
        pass

    def build_model(self, path: list[LayerConfig]) -> nn.Module:
        pass

    def load_reusable_weights(self, path: list[LayerConfig], model: nn.Module) -> None:
        """Copies weights from the saved model with the longest matching path prefix."""
        pass

    def store_weights(self, path: list[LayerConfig], model: nn.Module, result: EvaluationResult) -> None:
        """Keeps the model if it is the best so far for any of its path prefixes."""
        pass

    def train(self, model: nn.Module) -> None:
        pass

    def evaluate(self, model: nn.Module) -> EvaluationResult:
        pass
