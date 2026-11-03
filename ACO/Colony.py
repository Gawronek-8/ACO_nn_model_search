"""Ant colony: configuration, the ant and the search loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .Evaluator import EvaluationResult, Evaluator
from .Observers import Observers, BaseObserver
from .Strategies import STRATEGIES, StrategyName
from .Strategies.base import AntContext, IterationContext, MoveContext, PheromoneStrategy, SearchContext
from .utils import END, PARAMS, LayerConfig, Step, seed_everything


class SearchSpace(BaseModel):
    """Options an ant can pick for every hidden layer, and how deep the network may be."""

    neurons: list[int] = Field(default=[16, 32, 64, 128, 256], min_length=1)
    activation: list[str] = Field(default=["relu", "tanh", "elu", "leaky_relu"], min_length=1)
    dropout: list[float] = Field(default=[0.0, 0.1, 0.3, 0.5], min_length=1)
    max_depth: int = Field(default=6, ge=1)
    min_depth: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _check(self) -> SearchSpace:
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth cannot be greater than max_depth")
        for param in PARAMS:
            options = self.options(param)
            if len(set(options)) != len(options):
                raise ValueError(f"Duplicate options in {param}")
        return self

    def options(self, param: str) -> list:
        return getattr(self, param)

    def value_index(self, param: str, value) -> int:
        return self.options(param).index(value)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.options(param)) for param in PARAMS)

    @property
    def n_configs(self) -> int:
        return int(np.prod(self.shape))

    def config_index(self, layer: LayerConfig) -> int:
        """Position of a full layer config in the product of all options."""
        indexes = [self.value_index(param, getattr(layer, param)) for param in PARAMS]
        return int(np.ravel_multi_index(indexes, self.shape))

    def config_at(self, index: int) -> LayerConfig:
        indexes = np.unravel_index(index, self.shape)
        return LayerConfig(**{param: self.options(param)[i] for param, i in zip(PARAMS, indexes)})


class PheromoneConfig(BaseModel):
    start: float = Field(default=0.1, gt=0)  # THE PHEROMONE
    local_decay: float = Field(default=0.1, ge=0, le=1)  # SMTH TO TONE THE LOCAL UPDATE
    evaporation: float = Field(default=0.1, ge=0, le=1)  # SMTH TO TONE THE GLOBAL UPDATE


class ColonyConfig(BaseModel):
    strategy: StrategyName = "param_dependent"
    n_ants: int = Field(default=10, ge=1)
    n_iterations: int = Field(default=6, ge=1)
    greediness: float = Field(default=0.5, ge=0, le=1)  # HOW MUCH TO EXPLORE/WYKORZYSTAĆ FORGOR THE WORD
    progressive_depth: bool = True
    local_update: Literal["per_ant", "per_move"] = "per_ant"
    global_update_ant: Literal["global_best", "iteration_best"] = "global_best"
    reuse_weights: bool = True
    seed: int = 42
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    pheromone: PheromoneConfig = Field(default_factory=PheromoneConfig)


@dataclass
class Ant:
    """One walk through the search space and the score of the network it describes."""
    iteration: int
    steps: list[Step] = field(default_factory=list)
    result: EvaluationResult | None = None

    @property
    def path(self) -> list[LayerConfig]:
        return [step for step in self.steps if step is not END]

    @property
    def score(self) -> float:
        if self.result is None:
            raise RuntimeError("The ant has not been evaluated yet")
        return self.result.val_accuracy

    def transitions(self) -> Iterator[tuple[int, LayerConfig | None, Step]]:
        prev = None
        for depth, step in enumerate(self.steps):
            yield depth, prev, step
            if isinstance(step, LayerConfig):
                prev = step


class Colony:
    """Runs the search: ants walk, get evaluated, and the strategy updates its pheromones."""

    def __init__(
        self,
        config: ColonyConfig | None = None,
        evaluator: Evaluator | None = None,
        strategy: PheromoneStrategy | None = None,
        observers: BaseObserver | Iterable[BaseObserver] | None = None,
    ):
        self.config = config or ColonyConfig()
        if evaluator is None:
            raise ValueError("Colony needs an Evaluator, build one from a DataSplit: Evaluator(data, EvaluatorConfig(...))")
        self.evaluator = evaluator
        self.observers = Observers()
        if observers is not None:
            self.register_observer(observers)
        self.strategy = strategy or STRATEGIES[self.config.strategy]()
        self.ants: list[Ant] = []
        self.best_ant: Ant | None = None

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        evaluator: Evaluator | None = None,
        strategy: PheromoneStrategy | None = None,
        observers: BaseObserver | Iterable[BaseObserver] | None = None,
    ) -> Colony:
        """Creates a colony from a JSON file; missing fields keep their defaults."""
        config = ColonyConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
        return cls(config, evaluator, strategy, observers)

    def run(self) -> Ant | None:
        config = self.config
        seed_everything(config.seed)
        rng = np.random.default_rng(config.seed)
        search_ctx = SearchContext(config.search_space, config, rng)
        self.strategy.on_search_start(search_ctx)
        self.observers.on_search_start(search_ctx, self.strategy.snapshot())
        

        for iteration in range(config.n_iterations):
            depth_limit = self.depth_limit(iteration)
            iteration_ctx = IterationContext(iteration, depth_limit)
            self.strategy.on_iteration_start(iteration_ctx) # HONESTLY NO IDEA WHY THIS EXIST, IT DOES NOTHING
            self.observers.on_iteration_start(iteration_ctx, self.strategy.snapshot())

            ants = [self._run_ant(iteration, depth_limit) for _ in range(config.n_ants)]
            iteration_best = max(ants, key=lambda ant: ant.score)
            if self.best_ant is None or iteration_best.score > self.best_ant.score:
                self.best_ant = iteration_best

            update_ant = self.best_ant if config.global_update_ant == "global_best" else iteration_best # PICK THE BEST AND OVERALL OR IN THE ITERATION TO UPDATED PHEROMONES
            global_ctx = IterationContext(iteration, depth_limit, tuple(ants), self.best_ant, update_ant)
            self.strategy.on_global_step_end(global_ctx)
            self.observers.on_global_step_end(global_ctx, self.strategy.snapshot())

        self.observers.on_search_end(self.best_ant, self.strategy.snapshot()) # pyright: ignore[reportArgumentType] <- this tell the linter to shut up
        return self.best_ant

    def register_observer(self, observers: BaseObserver | Iterable[BaseObserver]) -> None:
        """Adds one observer, a list of them or a whole Observers group."""
        match observers:
            case Observers() | list() | tuple():  
                self.observers.extend(observers)
            case BaseObserver():
                self.observers.append(observers)
            case _:
                raise TypeError(
                    f"Expected a BaseObserver, a list of them or Observers, got {type(observers).__name__}"
                )

    def depth_limit(self, iteration: int) -> int:
        space = self.config.search_space
        if not self.config.progressive_depth:
            return space.max_depth
        return min(max(space.min_depth, 1) + iteration, space.max_depth)

    def _run_ant(self, iteration: int, depth_limit: int) -> Ant:
        ant = self._walk(iteration, depth_limit)
        ant.result = self.evaluator.evaluate_path(ant.path, reuse_weights=self.config.reuse_weights)
        self.ants.append(ant)
        ant_ctx = AntContext(iteration, ant)
        self.strategy.on_local_step_end(ant_ctx)  # THIS DOES THE SAME AS ON_ANT_STEP_END IN THE _WALK FUNCTION
        self.observers.on_local_step_end(ant_ctx, self.strategy.snapshot())
        return ant

    def _walk(self, iteration: int, depth_limit: int) -> Ant:
        """Lets one ant pick layers until it chooses END or reaches the depth limit.
        Stopping at the limit adds no END step, so that forced stop gets no pheromone.
        """
        ant = Ant(iteration)
        prev = None
        for depth in range(depth_limit):
            ctx = MoveContext(
                depth=depth,
                prev=prev,
                steps=tuple(ant.steps),
                depth_limit=depth_limit,
                can_end=depth >= self.config.search_space.min_depth,
            )
            step = self.strategy.on_ant_movement(ctx) 
            ant.steps.append(step)
            self.strategy.on_ant_step_end(ctx, step)  # THIS DOES THE SAME AS ON_LOCAL_STEP_END IN THE _RUN_ANT FUNCTION
            self.observers.on_ant_step_end(ctx, step, self.strategy.snapshot())
            if step is END:
                break
            prev = step
        return ant
