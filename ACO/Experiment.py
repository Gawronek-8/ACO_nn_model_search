"""Running one experiment: a dataset, a colony configuration and a list of seeds.

Every seed gets a fresh Evaluator (so reused weights never leak between runs) and its own folder:

    <out_dir>/<name>/<dataset>_<strategy>_s<seed>/ants.csv      every evaluated network, in order
    <out_dir>/<name>/<dataset>_<strategy>_s<seed>/summary.json  best network, its test score, config
    <out_dir>/<name>/summary_<dataset>_<strategy>.csv           one row per seed

The budget of a run is the number of evaluated models, n_ants * n_iterations, and it is the same for
every strategy, Random Search included. Training time is recorded next to it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from .Colony import Ant, Colony, ColonyConfig
from .Data import DataSplit
from .Datasets import DATASETS, load_dataset
from .Evaluator import Evaluator, EvaluatorConfig
from .Observers import BaseObserver, PheromoneRecorder
from .Strategies.base import IterationContext
from .Tracking import MLflowTracker, flatten_params

Log = Callable[[str], None]


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    dataset: str = "wine"
    seeds: list[int] = Field(default=[0], min_length=1)  # colony and training seeds, one run each
    data_seed: int = 42  # seed of the train/val/test split, shared by every run
    min_class_count: int = Field(default=0, ge=0)  # drop classes rarer than this (0 keeps all)
    colony: ColonyConfig = Field(default_factory=ColonyConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)

    @field_validator("dataset")
    @classmethod
    def _known_dataset(cls, value: str) -> str:
        if value not in DATASETS:
            raise ValueError(f"Unknown dataset '{value}', expected one of {sorted(DATASETS)}")
        return value

    @property
    def n_models(self) -> int:
        return self.colony.n_ants * self.colony.n_iterations

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ProgressObserver(BaseObserver):
    """Prints one line per finished iteration."""

    def __init__(self, label: str, n_iterations: int, log: Log):
        self.label, self.n_iterations, self.log = label, n_iterations, log

    def on_global_step_end(self, ctx: IterationContext, pheromones) -> None:
        best_here = max(ant.score for ant in ctx.ants)
        self.log(
            f"[{self.label}] iteration {ctx.iteration + 1}/{self.n_iterations}: "
            f"best in iteration {best_here:.4f}, best so far {ctx.best_ant.score:.4f}"  # pyright: ignore[reportOptionalMemberAccess]
        )


SUMMARY_KEYS = (
    "experiment", "dataset", "strategy", "seed", "device", "n_models", "best_val_accuracy", "best_val_loss",
    "best_val_balanced_accuracy", "best_layers", "best_n_params", "retrain_val_accuracy", "retrain_val_loss",
    "test_accuracy", "test_loss", "test_balanced_accuracy", "train_time_s", "search_time_s",
)

Loader = Callable[["ExperimentConfig", "str | Path | None"], DataSplit]


def load_experiment_data(config: ExperimentConfig, data_dir: str | Path | None = None) -> DataSplit:
    return load_dataset(config.dataset, seed=config.data_seed, min_class_count=config.min_class_count, data_dir=data_dir)


def run_experiment(
    config: ExperimentConfig,
    out_dir: str | Path = "runs",
    data_dir: str | Path | None = None,
    log: Log = print,
    resume: bool = False,
    loader: Loader | None = None,
    tracker: MLflowTracker | None = None,
) -> pd.DataFrame:
    """Runs every seed of the experiment and returns the per-seed summary table.

    With `resume`, a seed whose folder already holds a summary.json made with the same settings is not run again.
    The data is only loaded when some seed actually needs it; `loader` lets a caller share loaded data.
    """
    load = loader or load_experiment_data
    experiment_dir = Path(out_dir) / config.name
    strategy = config.colony.strategy
    data = None
    rows = []
    for seed in config.seeds:
        run_dir = experiment_dir / f"{config.dataset}_{strategy}_s{seed}"
        done = finished_summary(run_dir, run_config(config, seed)) if resume else None
        if done is not None:
            log(f"[{config.dataset} {strategy} seed {seed}] already done, skipping")
            rows.append(done)
            continue
        if data is None:
            data = load(config, data_dir)
        rows.append(run_seed(config, data, seed, run_dir, log, tracker))
    summary = pd.DataFrame(rows)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(experiment_dir / f"summary_{config.dataset}_{strategy}.csv", index=False)
    return summary


def run_config(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """The settings of one run: the experiment config narrowed down to a single seed."""
    return config.model_copy(update={"seeds": [seed], "colony": config.colony.model_copy(update={"seed": seed})})


def finished_summary(run_dir: Path, expected: ExperimentConfig) -> dict | None:
    """The saved summary of an earlier run in `run_dir`, or None if there is none or its settings differ."""
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        details = json.loads(path.read_text(encoding="utf-8"))
        if details["config"] != expected.model_dump(mode="json"):
            return None
        return {key: details[key] for key in SUMMARY_KEYS}
    except (ValueError, KeyError):
        return None


def run_seed(
    config: ExperimentConfig,
    data: DataSplit,
    seed: int,
    run_dir: Path,
    log: Log = print,
    tracker: MLflowTracker | None = None,
) -> dict:
    """One colony run on prepared data: search on validation, then score the winner on the test set.

    Besides ants.csv and summary.json the run folder gets pheromones.npz (ACO strategies only), and with a
    `tracker` the run is logged to MLflow as well.
    """
    this_run = run_config(config, seed)
    colony_config = this_run.colony
    label = f"{config.dataset} {colony_config.strategy} seed {seed}"
    evaluator = Evaluator(data, config.evaluator)
    recorder = PheromoneRecorder()
    observers: list[BaseObserver] = [ProgressObserver(label, colony_config.n_iterations, log), recorder]
    if tracker is not None:
        observers.append(tracker)
        tags = {
            "dataset": config.dataset, "strategy": colony_config.strategy, "seed": str(seed), "device": str(evaluator.device)
        }
        tracker.start(config.name, label, flatten_params(this_run.model_dump(mode="json")), tags)

    try:
        start = time.perf_counter()
        colony = Colony(colony_config, evaluator, observers=observers)
        best = colony.run()
        search_time = time.perf_counter() - start
        if best is None:
            raise RuntimeError("The colony finished without evaluating any ant")

        # retrains the winner from scratch and scores that one model on validation and on the test set
        retrain_val, test = evaluator.final_scores(best.path)
        ants = ants_table(colony.ants)
        run_dir.mkdir(parents=True, exist_ok=True)
        ants.to_csv(run_dir / "ants.csv", index=False)
        recorder.save(run_dir / "pheromones.npz")

        summary = {
            "experiment": config.name,
            "dataset": config.dataset,
            "strategy": colony_config.strategy,
            "seed": seed,
            "device": str(evaluator.device),
            "n_models": len(colony.ants),
            "best_val_accuracy": best.result.val_accuracy,  # pyright: ignore[reportOptionalMemberAccess]
            "best_val_loss": best.result.val_loss,  # pyright: ignore[reportOptionalMemberAccess]
            "best_val_balanced_accuracy": best.result.balanced_accuracy,  # pyright: ignore[reportOptionalMemberAccess]
            "best_layers": layers_text(best.path),
            "best_n_params": best.result.n_params,  # pyright: ignore[reportOptionalMemberAccess]
            # the retrained model scored on validation: compare with best_val_accuracy to see the search's own bias
            "retrain_val_accuracy": retrain_val.val_accuracy,
            "retrain_val_loss": retrain_val.val_loss,
            "test_accuracy": test.val_accuracy,  # the metric fields of the test result hold the test numbers
            "test_loss": test.val_loss,
            "test_balanced_accuracy": test.balanced_accuracy,
            "train_time_s": float(ants["train_time"].sum()),
            "search_time_s": search_time,
        }
        details = {
            **summary, "best_path": [asdict(layer) for layer in best.path], "config": this_run.model_dump(mode="json")
        }
        (run_dir / "summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        if tracker is not None:
            tracker.finish(summary, run_dir)
    except BaseException:
        if tracker is not None:
            tracker.fail()
        raise

    log(f"[{label}] best val {summary['best_val_accuracy']:.4f}, test {summary['test_accuracy']:.4f}, {summary['best_layers']}")
    return summary


def ants_table(ants: list[Ant]) -> pd.DataFrame:
    """One row per evaluated network, in evaluation order; `best_so_far` is the convergence curve."""
    table = pd.DataFrame(
        {
            "ant": range(len(ants)),
            "iteration": [ant.iteration for ant in ants],
            "n_layers": [len(ant.path) for ant in ants],
            "layers": [layers_text(ant.path) for ant in ants],
            "val_accuracy": [ant.result.val_accuracy for ant in ants],  # pyright: ignore[reportOptionalMemberAccess]
            "val_loss": [ant.result.val_loss for ant in ants],  # pyright: ignore[reportOptionalMemberAccess]
            "val_balanced_accuracy": [ant.result.balanced_accuracy for ant in ants],  # pyright: ignore[reportOptionalMemberAccess]
            "train_time": [ant.result.train_time for ant in ants],  # pyright: ignore[reportOptionalMemberAccess]
            "n_params": [ant.result.n_params for ant in ants],  # pyright: ignore[reportOptionalMemberAccess]
        }
    )
    table["best_so_far"] = table["val_accuracy"].cummax()
    return table


def layers_text(path) -> str:
    """'64-relu-0.1 | 32-tanh-0.0' (neurons-activation-dropout per layer)."""
    return " | ".join(f"{layer.neurons}-{layer.activation}-{layer.dropout}" for layer in path) or "(no hidden layers)"
