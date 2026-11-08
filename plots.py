"""Draws the report figures and the summary table from finished experiments.

    uv run python plots.py e1
    uv run python plots.py e1 e2 --format pdf
    uv run python plots.py e3_evap_0.1 e3_evap_0.3 e3_evap_0.5 --out runs/e3_plots

Files go to runs/<name>/plots (png and pdf by default). With several experiments every experiment is one series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import settings
from ACO.Plots import load_runs, make_all


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the results of finished experiments.")
    parser.add_argument("experiments", nargs="+", help="experiment names, i.e. folders in the runs directory")
    parser.add_argument("--runs-dir", type=Path, default=settings.RUNS_DIR, help="where the results are")
    parser.add_argument("--out", type=Path, help="where to write the figures (default: <runs-dir>/<name>/plots)")
    parser.add_argument("--format", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out or (
        args.runs_dir / args.experiments[0] / "plots"
        if len(args.experiments) == 1
        else args.runs_dir / ("plots_" + "_".join(args.experiments))
    )
    try:
        runs = load_runs(args.experiments, args.runs_dir)
        files = make_all(runs, out, args.format)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{len(files)} files in {out}:")
    for path in files:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
