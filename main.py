"""Runs one experiment from a JSON config.

    uv run python main.py configs/smoke.json
    uv run python main.py configs/default.json --dataset adult --strategy random --seeds 0 1 2 --device gpu

Command line options override the values in the config file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import settings
from ACO.Datasets import DATASETS
from ACO.Experiment import ExperimentConfig, run_experiment
from ACO.Strategies import STRATEGIES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ACO neural architecture search experiment.")
    parser.add_argument("config", type=Path, help="JSON experiment config, see configs/")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], help="where to train the networks")
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--seeds", type=int, nargs="+", help="one run per seed")
    parser.add_argument("--name", help="experiment name, used as the results folder")
    parser.add_argument("--out", type=Path, default=settings.RUNS_DIR, help="where to write the results")
    parser.add_argument("--data-dir", type=Path, default=settings.DATA_DIR, help="where datasets are cached")
    return parser.parse_args(argv)


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    top = {key: getattr(args, key) for key in ("dataset", "seeds", "name") if getattr(args, key) is not None}
    colony = config.colony.model_copy(update={"strategy": args.strategy} if args.strategy else {})
    evaluator = config.evaluator.model_copy(update={"device": args.device} if args.device else {})
    return config.model_copy(update={**top, "colony": colony, "evaluator": evaluator})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(ExperimentConfig.from_json(args.config), args)
    print(
        f"{config.name}: {config.dataset}, {config.colony.strategy}, {config.n_models} models per seed, "
        f"seeds {config.seeds}, device {config.evaluator.device}"
    )
    try:
        summary = run_experiment(config, args.out, args.data_dir)
    except RuntimeError as error:  # e.g. --device gpu without a GPU
        print(f"error: {error}", file=sys.stderr)
        return 1
    scores = summary[["best_val_accuracy", "test_accuracy", "train_time_s"]]
    print(f"\nover {len(summary)} seed(s):")
    print(scores.agg(["mean", "std"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
