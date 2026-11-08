"""A grid of experiments: every dataset x strategy combination, each with all the seeds.

Results land in the same folders as single experiments, plus `<out_dir>/<name>/grid_summary.csv` with one row
per run. Finished runs are skipped on a rerun (see run_experiment), so an interrupted grid can be continued.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .Data import DataSplit
from .Experiment import ExperimentConfig, Log, load_experiment_data, run_experiment
from .Tracking import MLflowTracker


def run_grid(
    config: ExperimentConfig,
    datasets: Iterable[str],
    strategies: Iterable[str],
    out_dir: str | Path = "runs",
    data_dir: str | Path | None = None,
    log: Log = print,
    resume: bool = True,
    tracker: MLflowTracker | None = None,
) -> pd.DataFrame:
    """Runs config for every dataset and strategy; each dataset is loaded at most once."""
    combos = [(dataset, strategy) for dataset in datasets for strategy in strategies]
    loaded: dict[tuple, DataSplit] = {}

    def loader(experiment: ExperimentConfig, directory) -> DataSplit:
        key = (experiment.dataset, experiment.data_seed, experiment.min_class_count)
        if key not in loaded:
            loaded[key] = load_experiment_data(experiment, directory)
        return loaded[key]

    frames = []
    for number, (dataset, strategy) in enumerate(combos, start=1):
        log(f"\n=== [{number}/{len(combos)}] {dataset} / {strategy} ===")
        experiment = config.model_copy(
            update={"dataset": dataset, "colony": config.colony.model_copy(update={"strategy": strategy})}
        )
        frames.append(run_experiment(experiment, out_dir, data_dir, log, resume=resume, loader=loader, tracker=tracker))

    summary = pd.concat(frames, ignore_index=True)
    (Path(out_dir) / config.name).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(out_dir) / config.name / "grid_summary.csv", index=False)
    return summary


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds (and std of the test accuracy) for every dataset and strategy."""
    return summary.groupby(["dataset", "strategy"], sort=False).agg(
        runs=("seed", "count"),
        val_accuracy=("best_val_accuracy", "mean"),
        test_accuracy=("test_accuracy", "mean"),
        test_std=("test_accuracy", "std"),
        train_time_s=("train_time_s", "mean"),
    ).round(4)
