from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from ..Strategies.base import AntContext, IterationContext, MoveContext, SearchContext
from ..utils import Step

if TYPE_CHECKING:
    from ..Colony import Ant

# copies of the strategy's pheromone tables by name, taken right after the strategy handled the event
Pheromones = dict[str, np.ndarray]


class BaseObserver(ABC):

    def on_search_start(self, ctx: SearchContext, pheromones: Pheromones) -> None:
        pass

    def on_iteration_start(self, ctx: IterationContext, pheromones: Pheromones) -> None:
        pass

    def on_ant_step_end(self, ctx: MoveContext, step: Step, pheromones: Pheromones) -> None:
        pass

    def on_local_step_end(self, ctx: AntContext, pheromones: Pheromones) -> None:
        pass

    def on_global_step_end(self, ctx: IterationContext, pheromones: Pheromones) -> None:
        pass

    def on_search_end(self, best_ant: Ant, pheromones: Pheromones) -> None:
        pass
