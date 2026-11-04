"""Runs a grid of experiments: every dataset x strategy combination with all the seeds.

    uv run python grid.py configs/default.json --datasets all --strategies all --seeds 0 1 2 --name e2
    uv run python grid.py configs/default.json --strategies random param_dependent --dry-run

Finished runs are skipped when you start the same grid again (use --no-resume to redo them).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import settings
from ACO.Datasets import DATASETS
from ACO.Experiment import ExperimentConfig
from ACO.Grid import aggregate, run_grid
from ACO.Strategies import STRATEGIES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid of ACO neural architecture search experiments.")
    parser.add_argument("config", type=Path, help="JSON experiment config, see configs/")
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all"], help="default: the dataset of the config")
    parser.add_argument("--strategies", nargs="+", choices=[*STRATEGIES, "all"], help="default: the strategy of the config")
    parser.add_argument("--seeds", type=int, nargs="+", help="one run per seed")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], help="where to train the networks")
    parser.add_argument("--name", help="experiment name, used as the results folder")
    parser.add_argument("--out", type=Path, default=settings.RUNS_DIR, help="where to write the results")
    parser.add_argument("--data-dir", type=Path, default=settings.DATA_DIR, help="where datasets are cached")
    parser.add_argument("--no-resume", action="store_true", help="run everything again, even finished runs")
    parser.add_argument("--dry-run", action="store_true", help="only print what would run")
    return parser.parse_args(argv)


def expand(values: list[str] | None, everything, default: str) -> list[str]:
    if values is None:
        return [default]
    return list(everything) if "all" in values else list(dict.fromkeys(values))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    updates = {key: getattr(args, key) for key in ("seeds", "name") if getattr(args, key) is not None}
    if args.device:
        updates["evaluator"] = config.evaluator.model_copy(update={"device": args.device})
    config = config.model_copy(update=updates)
    datasets = expand(args.datasets, DATASETS, config.dataset)
    strategies = expand(args.strategies, STRATEGIES, config.colony.strategy)

    n_runs = len(datasets) * len(strategies) * len(config.seeds)
    print(
        f"{config.name}: datasets {datasets}, strategies {strategies}, seeds {config.seeds}, "
        f"device {config.evaluator.device}\n{n_runs} runs, {config.n_models} models each, {n_runs * config.n_models} in total"
    )
    if args.dry_run:
        return 0

    try:
        summary = run_grid(config, datasets, strategies, args.out, args.data_dir, resume=not args.no_resume)
    except RuntimeError as error:  # e.g. --device gpu without a GPU
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("\nmean over seeds:")
    print(aggregate(summary).to_string())
    print(f"\nper-run table: {args.out / config.name / 'grid_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
