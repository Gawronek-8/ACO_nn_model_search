"""Statistical comparison of the strategies of a finished experiment (E1, E2).

For every dataset each compared strategy is tested against a reference strategy with the two-sided
Mann-Whitney U test on the test accuracy of the best network of every seed. The p-values of all
comparisons (all datasets together) are corrected with the Holm method.

    uv run python stats.py e1                                   # every ACO variant vs random
    uv run python stats.py e1 --reference independent           # E2: the dependent variants vs independent
    uv run python stats.py e1 --metric test_balanced_accuracy

Writes runs/<name>/stats/stats_<metric>_vs_<reference>.csv and .tex (booktabs, for \\input{}).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

DATASET_ORDER = ["adult", "wine", "letter"]  # same order as ACO/Plots.py
STRATEGY_ORDER = ["independent", "param_dependent", "config_dependent", "random"]


def load_summaries(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(experiment_dir.glob("*/summary.json")):
        details = json.loads(path.read_text(encoding="utf-8"))
        rows.append({key: value for key, value in details.items() if key not in ("best_path", "config")})
    if not rows:
        raise FileNotFoundError(f"No finished runs (*/summary.json) in {experiment_dir}")
    return pd.DataFrame(rows)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare(summary: pd.DataFrame, metric: str, reference: str, alpha: float) -> pd.DataFrame:
    datasets = [d for d in DATASET_ORDER if d in set(summary["dataset"])]
    datasets += sorted(set(summary["dataset"]) - set(DATASET_ORDER))
    strategies = [s for s in STRATEGY_ORDER if s in set(summary["strategy"]) and s != reference]
    rows = []
    for dataset in datasets:
        part = summary[summary["dataset"] == dataset]
        ref = part.loc[part["strategy"] == reference, metric].to_numpy()
        if len(ref) == 0:
            continue
        for strategy in strategies:
            values = part.loc[part["strategy"] == strategy, metric].to_numpy()
            if len(values) == 0:
                continue
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            rq1, rmed, rq3 = np.percentile(ref, [25, 50, 75])
            test = mannwhitneyu(values, ref, alternative="two-sided")
            rows.append({
                "dataset": dataset, "strategy": strategy, "reference": reference, "n": len(values), "n_ref": len(ref),
                "median": med, "iqr": q3 - q1, "median_ref": rmed, "iqr_ref": rq3 - rq1,
                "diff_median": med - rmed, "U": test.statistic, "p": test.pvalue,
            })
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    table["significant"] = table["p_holm"] < alpha
    return table


def _tex(text: str) -> str:
    return str(text).replace("_", r"\_")


def to_latex(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"zbiór & strategia & mediana (IQR) & mediana ref. (IQR) & różnica [pp] & $p$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    previous = None
    for row in table.itertuples():
        if previous is not None and row.dataset != previous:
            lines.append(r"\midrule")
        previous = row.dataset
        lines.append(
            f"{_tex(row.dataset)} & {_tex(row.strategy)} & {row.median:.4f} ({row.iqr:.4f}) & "
            f"{row.median_ref:.4f} ({row.iqr_ref:.4f}) & {100 * row.diff_median:+.2f} & "
            f"{row.p:.3f} & {row.p_holm:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mann-Whitney U + Holm comparison of strategies.")
    parser.add_argument("experiment", help="experiment name, i.e. a folder in the runs directory")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument("--reference", default="random", help="strategy every other strategy is compared with")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    experiment_dir = args.runs_dir / args.experiment
    try:
        summary = load_summaries(experiment_dir)
        table = compare(summary, args.metric, args.reference, args.alpha)
    except (FileNotFoundError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    out = experiment_dir / "stats"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"stats_{args.metric}_vs_{args.reference}"
    table.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.tex").write_text(to_latex(table), encoding="utf-8")

    shown = table[["dataset", "strategy", "median", "median_ref", "diff_median", "p", "p_holm", "significant"]]
    print(shown.round(4).to_string(index=False))
    print(f"\n{len(table)} comparisons, Holm-corrected at alpha={args.alpha}: "
          f"{int(table['significant'].sum())} significant")
    print(f"written: {out / (stem + '.csv')}, {out / (stem + '.tex')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
