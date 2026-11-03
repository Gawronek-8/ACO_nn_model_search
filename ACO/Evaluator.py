"""Training and scoring of the networks built by the ants."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from .Data import DataSplit
from .utils import LayerConfig

# names must match SearchSpace.activation
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}

EVAL_BATCH_SIZE = 4096


class EvaluatorConfig(BaseModel):
    device: Literal["auto", "cpu", "gpu"] = "auto"  # auto = GPU when there is one, otherwise CPU
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    patience: int | None = Field(default=5, ge=1)  # epochs without validation loss improvement; None = train all epochs


@dataclass
class EvaluationResult:
    val_accuracy: float
    val_loss: float
    train_time: float
    n_params: int


@dataclass
class _StoredWeights:
    """Weights of the first `len(hidden)` hidden layers (and the output layer for a full path)."""

    val_accuracy: float
    hidden: list[dict[str, torch.Tensor]]
    head: dict[str, torch.Tensor] | None


def resolve_device(name: str) -> torch.device:
    """'cpu' is always CPU, 'gpu' is CUDA (or Apple MPS) and fails loudly when there is none, 'auto' picks what exists."""
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "gpu":
        raise RuntimeError("device='gpu' was requested but no CUDA or MPS device is available; use 'cpu' or 'auto'")
    return torch.device("cpu")


class Evaluator:
    """Turns an ant's path into an MLP, trains it and scores it on the validation set.

    The test set is only touched by `final_test`, which is meant for the winning architecture after the search.
    """

    def __init__(self, data: DataSplit, config: EvaluatorConfig | None = None):
        self.config = config or EvaluatorConfig()
        self.device = resolve_device(self.config.device)
        self.data = data.to(self.device)
        self.weights: dict[tuple[LayerConfig, ...], _StoredWeights] = {}  # best weights per path prefix

    def evaluate_path(self, path: list[LayerConfig], reuse_weights: bool) -> EvaluationResult:
        """Full pipeline for one ant: build (or warm-start), train, score, store weights."""
        model = self.build_model(path)
        if reuse_weights:
            self.load_reusable_weights(path, model)
        result = self._fit_and_score(model, "val")
        if reuse_weights:
            self.store_weights(path, model, result)
        return result

    def final_test(self, path: list[LayerConfig]) -> EvaluationResult:
        """Trains the architecture from scratch and scores it on the test set.

        The metric fields of the returned result (`val_accuracy`, `val_loss`) hold the TEST numbers.
        Validation is still used for early stopping.
        """
        return self._fit_and_score(self.build_model(path), "test")

    def build_model(self, path: list[LayerConfig]) -> nn.Module:
        """Linear -> activation -> (dropout) for every layer of the path, then the output layer."""
        layers: list[nn.Module] = []
        in_features = self.data.n_features
        for layer in path:
            if layer.activation not in ACTIVATIONS:
                raise ValueError(f"Unknown activation '{layer.activation}', expected one of {sorted(ACTIVATIONS)}")
            layers += [nn.Linear(in_features, layer.neurons), ACTIVATIONS[layer.activation]()]
            if layer.dropout > 0:
                layers.append(nn.Dropout(layer.dropout))
            in_features = layer.neurons
        layers.append(nn.Linear(in_features, self.data.n_classes))
        return nn.Sequential(*layers).to(self.device)

    def load_reusable_weights(self, path: list[LayerConfig], model: nn.Module) -> int:
        """Copies weights from the saved model with the longest matching path prefix.

        Returns how many hidden layers were warm-started. The output layer is copied only for an identical path.
        """
        linears = _linears(model)
        for k in range(len(path), 0, -1):
            stored = self.weights.get(tuple(path[:k]))
            if stored is None:
                continue
            for layer, state in zip(linears[:k], stored.hidden):
                layer.load_state_dict(state)
            if k == len(path) and stored.head is not None:
                linears[-1].load_state_dict(stored.head)
            return k
        return 0

    def store_weights(self, path: list[LayerConfig], model: nn.Module, result: EvaluationResult) -> None:
        """Keeps the model if it is the best so far for any of its path prefixes."""
        linears = _linears(model)
        states = [{name: tensor.detach().cpu().clone() for name, tensor in layer.state_dict().items()} for layer in linears]
        for k in range(1, len(path) + 1):
            prefix = tuple(path[:k])
            current = self.weights.get(prefix)
            if current is None or result.val_accuracy > current.val_accuracy:
                self.weights[prefix] = _StoredWeights(
                    result.val_accuracy, states[:k], states[-1] if k == len(path) else None
                )

    def train(self, model: nn.Module) -> None:
        """Adam + cross entropy on mini-batches; with `patience` set, stops on validation loss and keeps the best epoch."""
        config, data = self.config, self.data
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n_samples = len(data.X_train)
        best_loss, best_state, stale_epochs = math.inf, None, 0

        for _ in range(config.epochs):
            model.train()
            order = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, config.batch_size):
                batch = order[start : start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss_fn(model(data.X_train[batch]), data.y_train[batch]).backward()
                optimizer.step()

            if config.patience is None:
                continue
            val_loss, _ = self._score(model, data.X_val, data.y_val)
            if val_loss < best_loss:
                best_loss, best_state, stale_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    def evaluate(self, model: nn.Module, split: Literal["val", "test"] = "val") -> EvaluationResult:
        """Accuracy and loss on one split; `train_time` is filled in by the caller that measured it."""
        X, y = (self.data.X_val, self.data.y_val) if split == "val" else (self.data.X_test, self.data.y_test)
        loss, accuracy = self._score(model, X, y)
        return EvaluationResult(accuracy, loss, 0.0, sum(p.numel() for p in model.parameters()))

    def _fit_and_score(self, model: nn.Module, split: Literal["val", "test"]) -> EvaluationResult:
        start = time.perf_counter()
        self.train(model)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - start
        return replace(self.evaluate(model, split), train_time=train_time)

    @torch.no_grad()
    def _score(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        """Returns (mean cross entropy, accuracy) over the whole split."""
        model.eval()
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), EVAL_BATCH_SIZE):
            logits = model(X[start : start + EVAL_BATCH_SIZE])
            target = y[start : start + EVAL_BATCH_SIZE]
            total_loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
            correct += (logits.argmax(dim=1) == target).sum().item()
        return total_loss / len(X), correct / len(X)


def _linears(model: nn.Module) -> list[nn.Linear]:
    """Linear layers of a built model: the hidden ones first, the output layer last."""
    return [module for module in model if isinstance(module, nn.Linear)]  # pyright: ignore[reportGeneralTypeIssues]
