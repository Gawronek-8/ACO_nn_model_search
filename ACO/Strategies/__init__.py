from typing import Literal

from .base import PheromoneStrategy
from .config_dependent import ConfigDependentStrategy
from .independent import IndependentStrategy
from .param_dependent import ParamDependentStrategy
from .random_search import RandomSearchStrategy

StrategyName = Literal["independent", "param_dependent", "config_dependent", "random"]

STRATEGIES: dict[str, type[PheromoneStrategy]] = {
    "independent": IndependentStrategy,
    "param_dependent": ParamDependentStrategy,
    "config_dependent": ConfigDependentStrategy,
    "random": RandomSearchStrategy,
}
