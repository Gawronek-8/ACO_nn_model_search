"""Search efficiency of a finished experiment (E4): how fast each ACO strategy gets ahead of Random Search.

Reads the runs of one experiment (by default e1, every strategy, every seed) and measures
    budget     best validation accuracy after k models, ACO vs Random Search (Mann-Whitney U, Holm per k)
    target     first model and training time at which each run reaches Random Search's mean final result
    cost       mean depth, training time per model, run time vs Random Search, distinct architectures

    uv run python efficiency.py e1

Writes runs/e4/e4_{budget,target,cost}.csv, the report tables tab_{budget,cost}.tex (booktabs) and the figures
img/e4/{advantage,to_target}.{pdf,png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ACO.Plots import INK_MUTED, STRATEGY_STYLE, _apply_style, _panels, load_runs, save_figure
from stats import holm

ACO_STRATEGIES = ["param_dependent", "independent", "config_dependent"]
BUDGETS = [25, 50, 100, 200, 300]
TABLE_BUDGETS = [50, 100, 200, 300]


def models_to_target(best_so_far: np.ndarray, target: float) -> int | None:
    """1-based number of the first model whose best-so-far score reaches `target`, None if it never does."""
    hit = np.flatnonzero(best_so_far >= target)
    return int(hit[0]) + 1 if len(hit) else None


def budget_table(curves: dict, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for k in BUDGETS:
        part = []
        for d in datasets:
            ref = curves[d, "random"].iloc[k - 1]
            for s in ACO_STRATEGIES:
                x = curves[d, s].iloc[k - 1]
                part.append({"k": k, "dataset": d, "strategy": s, "mean": x.mean(), "random": ref.mean(),
                             "diff_pp": 100 * (x.mean() - ref.mean()),
                             "p": mannwhitneyu(x, ref, alternative="two-sided").pvalue})
        part = pd.DataFrame(part)
        part["p_holm"] = holm(part["p"].to_numpy())
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def target_table(ants: pd.DataFrame, curves: dict, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, s, seed), g in ants.sort_values("ant").groupby(["dataset", "strategy", "seed"]):
        target = curves[d, "random"].iloc[-1].mean()
        k = models_to_target(g["best_so_far"].to_numpy(), target)
        rows.append({"dataset": d, "strategy": s, "seed": seed, "target": target, "models": k,
                     "time_s": g["train_time"].to_numpy()[:k].sum() if k else np.nan})
    table = pd.DataFrame(rows)
    random_time = summary[summary["strategy"] == "random"].groupby("dataset")["train_time_s"].mean()
    table["time_share"] = table["time_s"] / table["dataset"].map(random_time)
    return table


def cost_table(ants: pd.DataFrame, summary: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    by = ants.groupby(["dataset", "strategy"])
    cost = pd.DataFrame({
        "depth": by["n_layers"].mean(),
        "s_per_model": by["train_time"].mean(),
        "unique": ants.groupby(["dataset", "strategy", "seed"])["layers"].nunique().groupby(["dataset", "strategy"]).mean(),
        "train_time_s": summary.groupby(["dataset", "strategy"])["train_time_s"].mean(),
    })
    random_time = cost["train_time_s"].xs("random", level="strategy")
    cost["time_vs_random_pct"] = 100 * (cost["train_time_s"] / random_time.reindex(cost.index.get_level_values("dataset")).to_numpy() - 1)
    reached = target.groupby(["dataset", "strategy"]).agg(
        reached=("models", "count"), models_median=("models", "median"), time_share_median=("time_share", "median"))
    return cost.join(reached)


def _tex(text: str) -> str:
    return text.replace("_", r"\_")


def budget_latex(budget: pd.DataFrame, curves: dict, datasets: list[str]) -> str:
    """Random Search's value, then each ACO strategy's difference in pp; bold where significant after Holm."""
    lines = [r"\begin{tabular}{ll" + "r" * len(TABLE_BUDGETS) + "}", r"\toprule",
             "zbiór & strategia & " + " & ".join(f"$k = {k}$" for k in TABLE_BUDGETS) + r" \\", r"\midrule"]
    for i, d in enumerate(datasets):
        if i:
            lines.append(r"\midrule")
        lines.append(f"{d} & random & " + " & ".join(f"{curves[d, 'random'].iloc[k - 1].mean():.4f}" for k in TABLE_BUDGETS) + r" \\")
        for s in ACO_STRATEGIES:
            cells = []
            for k in TABLE_BUDGETS:
                r = budget[(budget["k"] == k) & (budget["dataset"] == d) & (budget["strategy"] == s)].iloc[0]
                cells.append(rf"\textbf{{{r.diff_pp:+.2f}}}" if r.p_holm < 0.05 else f"{r.diff_pp:+.2f}")
            lines.append(f"{d} & {_tex(s)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def cost_latex(cost: pd.DataFrame, datasets: list[str]) -> str:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"zbiór & strategia & warstwy & s/model & czas vs RS & przebiegi & modele & czas \\", r"\midrule"]
    for i, d in enumerate(datasets):
        if i:
            lines.append(r"\midrule")
        for s in ACO_STRATEGIES + ["random"]:
            r = cost.loc[(d, s)]
            vs = "--" if s == "random" else f"{r.time_vs_random_pct:+.0f}\\%"
            lines.append(f"{d} & {_tex(s)} & {r.depth:.2f} & {r.s_per_model:.2f} & {vs} & {int(r.reached)}/10 & "
                         f"{r.models_median:.0f} & {100 * r.time_share_median:.0f}\\% \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def advantage_figure(runs, curves: dict, budget: pd.DataFrame):
    """Mean best-so-far of each ACO strategy minus Random Search's, from the 10th model on (the first ones are a
    random start); markers at the tested budgets, filled where the difference is significant after Holm."""
    _apply_style()
    figure, axes = _panels(runs, height=2.8)
    handles = {}
    for ax, d in zip(axes, runs.datasets):
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ref = curves[d, "random"].mean(axis=1).to_numpy()
        for s in ACO_STRATEGIES:
            color, marker = STRATEGY_STYLE[s]
            diff = 100 * (curves[d, s].mean(axis=1).to_numpy() - ref)
            x = np.arange(1, len(diff) + 1)
            (handles[s],) = ax.plot(x[9:], diff[9:], color=color, linewidth=1.6)
            for r in budget[(budget["dataset"] == d) & (budget["strategy"] == s)].itertuples():
                ax.plot(r.k, r.diff_pp, marker=marker, markersize=5, color=color,
                        markerfacecolor=color if r.p_holm < 0.05 else "white", markeredgewidth=1.2)
        ax.set_xlabel("models evaluated")
    axes[0].set_ylabel("ACO − Random Search [pp]")
    figure.legend([handles[s] for s in ACO_STRATEGIES], ACO_STRATEGIES, loc="outside lower center", ncols=3, frameon=False)
    return figure


def to_target_figure(runs, target: pd.DataFrame):
    """Models each run needs to reach Random Search's mean final result; runs that never do sit above 300."""
    _apply_style()
    figure, axes = _panels(runs, height=2.8)
    rng = np.random.default_rng(0)
    order = ACO_STRATEGIES + ["random"]
    for ax, d in zip(axes, runs.datasets):
        for i, s in enumerate(order):
            color, marker = STRATEGY_STYLE[s]
            k = target[(target["dataset"] == d) & (target["strategy"] == s)]["models"].to_numpy(dtype=float)
            missed = np.isnan(k)
            jitter = rng.uniform(-0.18, 0.18, len(k))
            ax.scatter(i + jitter[~missed], k[~missed], color=color, marker=marker, s=16, alpha=0.8, linewidths=0)
            ax.scatter(i + jitter[missed], np.full(missed.sum(), 330), facecolors="white", edgecolors=color, marker=marker, s=16)
            if (~missed).any():
                ax.hlines(np.median(k[~missed]), i - 0.3, i + 0.3, color="#0b0b0b", linewidth=1.2)
        ax.axhline(300, color=INK_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(order)), [s.replace("_", "\n") for s in order])
        ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 330], ["0", "50", "100", "150", "200", "250", "300", "never"])
        ax.set_ylim(0, 350)
    axes[0].set_ylabel("models to reach Random Search @300")
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search efficiency of ACO vs Random Search (E4).")
    parser.add_argument("experiment", nargs="?", default="e1", help="experiment with every strategy, a folder in the runs directory")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("runs/e4"))
    args = parser.parse_args(argv)

    runs = load_runs([args.experiment], args.runs_dir)
    ants, summary = runs.ants.rename(columns={"series": "strategy"}), runs.summary
    curves = {(d, s): g.pivot(index="ant", columns="seed", values="best_so_far").sort_index()
              for (d, s), g in ants.groupby(["dataset", "strategy"])}

    budget = budget_table(curves, runs.datasets)
    target = target_table(ants, curves, summary)
    cost = cost_table(ants, summary, target)

    images = args.out / "img" / "e4"
    images.mkdir(parents=True, exist_ok=True)
    budget.to_csv(args.out / "e4_budget.csv", index=False)  # unrounded, so slides round like the report tables
    target.to_csv(args.out / "e4_target.csv", index=False)
    cost.round(4).to_csv(args.out / "e4_cost.csv")
    (args.out / "tab_budget.tex").write_text(budget_latex(budget, curves, runs.datasets), encoding="utf-8")
    (args.out / "tab_cost.tex").write_text(cost_latex(cost, runs.datasets), encoding="utf-8")
    save_figure(advantage_figure(runs, curves, budget), images, "advantage", ["pdf", "png"])
    save_figure(to_target_figure(runs, target), images, "to_target", ["pdf", "png"])

    significant = budget[budget["p_holm"] < 0.05].groupby("k").size().reindex(BUDGETS, fill_value=0)
    print("significant ACO vs Random Search comparisons (of 9) per budget:", significant.to_dict())
    print(cost.round(3).to_string())
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
