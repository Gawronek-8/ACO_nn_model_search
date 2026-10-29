from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from ..Strategies.base import AntContext, IterationContext, MoveContext, SearchContext
from ..utils import Step
from .base import BaseObserver, Pheromones

if TYPE_CHECKING:
    from ..Colony import Ant


class Observers(BaseObserver):

    def __init__(self, observers: Iterable[BaseObserver] = ()):
        self._observers = list(observers)

    def __len__(self) -> int:
        return len(self._observers)

    def __iter__(self) -> Iterator[BaseObserver]:
        return iter(self._observers)

    def __getitem__(self, key: int) -> BaseObserver:
        return self._observers[key]

    def append(self, observer: BaseObserver):
        self._observers.append(observer)

    def extend(self, observers: Iterable[BaseObserver]):
        self._observers.extend(list(observers))

    def on_search_start(self, ctx: SearchContext, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_search_start(ctx, pheromones)

    def on_iteration_start(self, ctx: IterationContext, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_iteration_start(ctx, pheromones)

    def on_ant_step_end(self, ctx: MoveContext, step: Step, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_ant_step_end(ctx, step, pheromones)

    def on_local_step_end(self, ctx: AntContext, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_local_step_end(ctx, pheromones)

    def on_global_step_end(self, ctx: IterationContext, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_global_step_end(ctx, pheromones)

    def on_search_end(self, best_ant: Ant, pheromones: Pheromones) -> None:
        for observer in self._observers:
            observer.on_search_end(best_ant, pheromones)
