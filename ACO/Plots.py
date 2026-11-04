"""Figures and tables for the report, made from the files that experiment runs leave in runs/<name>/.

One experiment folder gives one series per strategy; several experiment folders (e.g. the variants of E3)
give one series per experiment. Lines show the mean over seeds with a band of one standard deviation.

Figures (each saved in every requested format):
    convergence      best validation accuracy against the number of models evaluated
    efficiency       best validation accuracy against cumulative training time
    distribution     validation accuracy of every evaluated network
    test_accuracy    test accuracy of the winning network, one dot per seed
    pheromones_*     final pheromone tables of one ACO strategy on one dataset
    summary_table    mean results per dataset and series (.csv and a booktabs .tex)

Colors follow the entity, never its rank: every strategy keeps its color in every figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, AXIS = "#e6e5e1", "#c9c8c2"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # one hue, light to dark

# fixed per strategy; the three ACO strategies take the first three categorical slots, the baseline is neutral gray
STRATEGY_STYLE = {
    "param_dependent": (CATEGORICAL[0], "o"),
    "independent": (CATEGORICAL[1], "s"),
    "config_dependent": (CATEGORICAL[2], "^"),
    "random": ("#7d7c78", "D"),
}
STRATEGY_ORDER = list(STRATEGY_STYLE)
DATASET_ORDER = ["adult", "wine", "letter"]
PHEROMONE_PANELS = ("type", "neurons", "activation", "dropout")
TYPE_VALUES = ["layer", "END"]

RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # embed real fonts, so the text stays text in LaTeX and journals
    "ps.fonttype": 42,
}


@dataclass
class Runs:
    """Everything the figures need, loaded from one or more experiment folders."""

    summary: pd.DataFrame  # one row per run
    ants: pd.DataFrame  # one row per evaluated network, with the keys of its run
    folders: dict[tuple, Path] = field(default_factory=dict)  # (series, dataset, seed) -> run folder
    spaces: dict[tuple, dict] = field(default_factory=dict)  # same key -> search space of that run
    series_order: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)

    @property
    def styles(self) -> dict[str, tuple[str, str]]:
        """Color and marker per series; strategies keep their own, anything else takes categorical slots in order."""
        if all(name in STRATEGY_STYLE for name in self.series_order):
            return {name: STRATEGY_STYLE[name] for name in self.series_order}
        if len(self.series_order) > len(CATEGORICAL):
            raise ValueError(
                f"{len(self.series_order)} series do not fit one figure (at most {len(CATEGORICAL)}); "
                "plot fewer experiments at once"
            )
        return {name: (CATEGORICAL[i], MARKERS[i]) for i, name in enumerate(self.series_order)}


def load_runs(names: list[str], runs_dir: str | Path = "runs") -> Runs:
    """Reads every finished run (a folder with summary.json) of the named experiments."""
    rows, frames, folders, spaces = [], [], {}, {}
    multiple = len(names) > 1
    for name in names:
        summaries = sorted((Path(runs_dir) / name).glob("*/summary.json"))
        if not summaries:
            raise FileNotFoundError(f"No finished runs found in {Path(runs_dir) / name}")
        strategies = set()
        details_list = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
        strategies = {details["strategy"] for details in details_list}
        for path, details in zip(summaries, details_list):
            strategy = details["strategy"]
            series = (name if len(strategies) == 1 else f"{name}/{strategy}") if multiple else strategy
            row = {key: value for key, value in details.items() if key not in ("best_path", "config")}
            row.update(experiment=name, series=series)
            rows.append(row)
            ants = pd.read_csv(path.parent / "ants.csv")
            ants["series"], ants["dataset"], ants["seed"] = series, details["dataset"], details["seed"]
            frames.append(ants)
            key = (series, details["dataset"], details["seed"])
            folders[key] = path.parent
            spaces[key] = details["config"]["colony"]["search_space"]

    summary = pd.DataFrame(rows)
    if multiple:
        series_order = list(dict.fromkeys(summary["series"]))
    else:
        present = set(summary["series"])
        series_order = [s for s in STRATEGY_ORDER if s in present] + sorted(present - set(STRATEGY_ORDER))
    present_datasets = set(summary["dataset"])
    datasets = [d for d in DATASET_ORDER if d in present_datasets] + sorted(present_datasets - set(DATASET_ORDER))
    return Runs(summary, pd.concat(frames, ignore_index=True), folders, spaces, series_order, datasets)


# ---------------------------------------------------------------------------------------------------
# helpers


def _apply_style() -> None:
    matplotlib.rcParams.update(RC)


def _style_axes(ax: Axes, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, length=3, width=0.8, color=AXIS)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _panels(runs: Runs, width: float = 2.9, height: float = 3.3, legend_room: float = 0.5):
    figure = Figure(figsize=(width * len(runs.datasets) + 0.5, height + legend_room), layout="constrained")
    axes = figure.subplots(1, len(runs.datasets), squeeze=False)[0]
    for ax, dataset in zip(axes, runs.datasets):
        _style_axes(ax)
        ax.set_title(dataset)
    return figure, axes


def _legend(figure: Figure, runs: Runs, handles: dict) -> None:
    """A legend is always there for two or more series; a single series is named by the figure itself."""
    if len(runs.series_order) > 1:
        ordered = [name for name in runs.series_order if name in handles]
        figure.legend([handles[n] for n in ordered], ordered, loc="outside lower center", ncols=len(ordered), frameon=False)


def _cap_accuracy(ax: Axes) -> None:
    low, high = ax.get_ylim()
    ax.set_ylim(low, min(high, 1.005))


def _band(ax: Axes, x, mean, std, color) -> None:
    if std is not None and not np.all(np.isnan(std)):
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)


def best_at_times(times: np.ndarray, best: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Best score reached by each moment of `grid` (a step function); NaN before the first model finished."""
    index = np.searchsorted(times, grid, side="right") - 1
    return np.where(index >= 0, best[np.clip(index, 0, None)], np.nan)


# ---------------------------------------------------------------------------------------------------
# figures


def convergence_figure(runs: Runs) -> Figure:
    """Best validation accuracy found so far against the number of evaluated models (same budget for every series)."""
    _apply_style()
    figure, axes = _panels(runs)
    handles = {}
    for ax, dataset in zip(axes, runs.datasets):
        for series, (color, marker) in runs.styles.items():
            frame = runs.ants[(runs.ants["dataset"] == dataset) & (runs.ants["series"] == series)]
            if frame.empty:
                continue
            curves = frame.pivot(index="ant", columns="seed", values="best_so_far")
            x = curves.index.to_numpy() + 1
            mean = curves.mean(axis=1).to_numpy()
            std = curves.std(axis=1, ddof=1).to_numpy() if curves.shape[1] > 1 else None
            (line,) = ax.plot(
                x, mean, color=color, marker=marker, markersize=4.5, markevery=max(1, len(x) // 6), linewidth=1.8
            )
            _band(ax, x, mean, std, color)
            handles[series] = line
        ax.set_xlabel("models evaluated")
        _cap_accuracy(ax)
    axes[0].set_ylabel("best validation accuracy")
    _legend(figure, runs, handles)
    return figure


def efficiency_figure(runs: Runs) -> Figure:
    """Best validation accuracy against cumulative training time; a series stops where its fastest run ends."""
    _apply_style()
    figure, axes = _panels(runs)
    handles = {}
    for ax, dataset in zip(axes, runs.datasets):
        for series, (color, marker) in runs.styles.items():
            frame = runs.ants[(runs.ants["dataset"] == dataset) & (runs.ants["series"] == series)]
            if frame.empty:
                continue
            runs_of_series = [
                (np.cumsum(group["train_time"].to_numpy()), group["best_so_far"].to_numpy())
                for _, group in frame.sort_values("ant").groupby("seed")
            ]
            start = max(times[0] for times, _ in runs_of_series)
            stop = min(times[-1] for times, _ in runs_of_series)
            grid = np.linspace(start, stop, 100)
            curves = np.array([best_at_times(times, best, grid) for times, best in runs_of_series])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0, ddof=1) if len(curves) > 1 else None
            (line,) = ax.plot(grid, mean, color=color, marker=marker, markersize=4.5, markevery=20, linewidth=1.8)
            _band(ax, grid, mean, std, color)
            handles[series] = line
        ax.set_xlabel("cumulative training time [s]")
        _cap_accuracy(ax)
    axes[0].set_ylabel("best validation accuracy")
    _legend(figure, runs, handles)
    return figure


def _tick_name(series: str) -> str:
    return series.replace("_", "\n").replace("/", "/\n")


def distribution_figure(runs: Runs) -> Figure:
    """Validation accuracy of every network the series evaluated, pooled over seeds."""
    _apply_style()
    figure, axes = _panels(runs, legend_room=0.9)
    for ax, dataset in zip(axes, runs.datasets):
        names = [s for s in runs.styles if not runs.ants[(runs.ants["dataset"] == dataset) & (runs.ants["series"] == s)].empty]
        data = [runs.ants[(runs.ants["dataset"] == dataset) & (runs.ants["series"] == s)]["val_accuracy"] for s in names]
        parts = ax.boxplot(
            data,
            positions=range(len(names)),
            widths=0.55,
            patch_artist=True,
            medianprops={"color": INK, "linewidth": 1.4},
            whiskerprops={"color": AXIS, "linewidth": 1.0},
            capprops={"color": AXIS, "linewidth": 1.0},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_MUTED, "markeredgecolor": "none"},
        )
        for box, name in zip(parts["boxes"], names):
            box.set(facecolor=runs.styles[name][0], alpha=0.75, edgecolor="none")
        ax.set_xticks(range(len(names)), [_tick_name(n) for n in names])
        _cap_accuracy(ax)
    axes[0].set_ylabel("validation accuracy of evaluated networks")
    return figure


def test_accuracy_figure(runs: Runs) -> Figure:
    """Test accuracy of each run's winning network: one dot per seed, the mean with one standard deviation."""
    _apply_style()
    figure, axes = _panels(runs, legend_room=0.9)
    for ax, dataset in zip(axes, runs.datasets):
        names = [s for s in runs.styles if not runs.summary[(runs.summary["dataset"] == dataset) & (runs.summary["series"] == s)].empty]
        for position, name in enumerate(names):
            color, marker = runs.styles[name]
            values = runs.summary[(runs.summary["dataset"] == dataset) & (runs.summary["series"] == name)]["test_accuracy"].to_numpy()
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.zeros(1)
            ax.scatter(position + offsets, values, s=22, color=color, marker=marker, alpha=0.55, linewidths=0, zorder=2)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(position, values.mean(), yerr=std, color=INK, marker="_", markersize=14, linewidth=1.2, capsize=3, zorder=3)
        ax.set_xticks(range(len(names)), [_tick_name(n) for n in names])
        ax.set_xlim(-0.6, len(names) - 0.4)
        _cap_accuracy(ax)
    axes[0].set_ylabel("test accuracy of the best network")
    return figure


def pheromone_marginals(runs: Runs, series: str, dataset: str) -> dict[str, tuple[np.ndarray, list[str]]] | None:
    """Final pheromones as depth x value matrices (mean over seeds, strongest cell over contexts), per choice.

    Returns {panel: (matrix, value labels)} or None when the series keeps no pheromones (random search).
    """
    keys = [key for key in runs.folders if key[0] == series and key[1] == dataset and (runs.folders[key] / "pheromones.npz").exists()]
    if not keys:
        return None
    space = runs.spaces[keys[0]]
    shape = (len(space["neurons"]), len(space["activation"]), len(space["dropout"]))
    totals: dict[str, list[np.ndarray]] = {}
    for key in keys:
        with np.load(runs.folders[key] / "pheromones.npz") as saved:
            final = {name: saved[name][-1].max(axis=1) for name in saved.files}  # (depth, values), max over contexts
        panels = {"type": final["type"]}
        if "config" in final:
            depth = final["config"].shape[0]
            cube = final["config"].reshape(depth, *shape)
            for axis, name in enumerate(("neurons", "activation", "dropout")):
                other = tuple(i + 1 for i in range(3) if i != axis)
                panels[name] = cube.max(axis=other)
        else:
            panels.update({name: final[name] for name in ("neurons", "activation", "dropout")})
        for name, matrix in panels.items():
            totals.setdefault(name, []).append(matrix)
    labels = {"type": TYPE_VALUES, **{name: [str(v) for v in space[name]] for name in ("neurons", "activation", "dropout")}}
    return {name: (np.mean(totals[name], axis=0), labels[name]) for name in PHEROMONE_PANELS}


def pheromone_figure(runs: Runs, series: str, dataset: str) -> Figure | None:
    """Heatmaps of the final pheromone tables: layer position (rows) against the option chosen there (columns)."""
    marginals = pheromone_marginals(runs, series, dataset)
    if marginals is None:
        return None
    _apply_style()
    depth = marginals["type"][0].shape[0]
    top = max(matrix.max() for matrix, _ in marginals.values())
    figure = Figure(figsize=(7.2, 0.5 * depth + 1.9), layout="constrained")
    axes = figure.subplots(1, 4, gridspec_kw={"width_ratios": [len(labels) for _, labels in marginals.values()]})
    cmap = LinearSegmentedColormap.from_list("pheromone", SEQUENTIAL)
    image = None
    for ax, (name, (matrix, labels)) in zip(axes, marginals.items()):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=top, interpolation="nearest")
        ax.set_title(name if name != "type" else "layer or END")
        ax.set_xticks(range(len(labels)), labels, rotation=45 if name == "activation" else 0)
        ax.set_yticks(range(depth), [str(i + 1) for i in range(depth)] if ax is axes[0] else [])
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)  # thin gaps between cells
        ax.set_yticks(np.arange(-0.5, depth, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="both", colors=INK_SECONDARY, length=0)
        ax.title.set_color(INK)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("layer position")
    colorbar = figure.colorbar(image, ax=list(axes), shrink=0.8, pad=0.02)
    colorbar.set_label("pheromone", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_SECONDARY, length=0)
    figure.suptitle(f"{dataset}, {series}", color=INK, fontsize=10)
    return figure


# ---------------------------------------------------------------------------------------------------
# table


def summary_table(runs: Runs) -> pd.DataFrame:
    """Mean over seeds for every dataset and series (and the standard deviation of the test accuracy)."""
    frame = runs.summary
    table = frame.groupby(["dataset", "series"], sort=False).agg(
        runs=("seed", "count"),
        models=("n_models", "mean"),
        val_accuracy=("best_val_accuracy", "mean"),
        test_accuracy=("test_accuracy", "mean"),
        test_std=("test_accuracy", "std"),
        train_time_s=("train_time_s", "mean"),
    )
    order = [(d, s) for d in runs.datasets for s in runs.series_order if (d, s) in table.index]
    return table.loc[order].reset_index()


def _tex(text: str) -> str:
    return text.replace("_", r"\_").replace("/", "/")


def table_to_latex(table: pd.DataFrame) -> str:
    """A booktabs tabular, ready for \\input in the report (needs \\usepackage{booktabs})."""
    header = ["dataset", "strategy", "runs", "models", "val.\\ acc.", "test acc.", "test std", "train time [s]"]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    previous = None
    for row in table.itertuples(index=False):
        if previous is not None and row.dataset != previous:
            lines.append(r"\midrule")
        previous = row.dataset
        std = "--" if pd.isna(row.test_std) else f"{row.test_std:.4f}"
        cells = [
            _tex(row.dataset), _tex(row.series), str(row.runs), f"{row.models:.0f}",
            f"{row.val_accuracy:.4f}", f"{row.test_accuracy:.4f}", std, f"{row.train_time_s:.1f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------
# everything at once


def save_figure(figure: Figure, out_dir: Path, name: str, formats: list[str]) -> list[Path]:
    paths = []
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        figure.savefig(path, dpi=200, facecolor="white")
        paths.append(path)
    return paths


def make_all(runs: Runs, out_dir: str | Path, formats: list[str] = ("png", "pdf")) -> list[Path]:
    """Writes every figure and the table into `out_dir` and returns the files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = list(formats)
    written: list[Path] = []
    for name, build in (
        ("convergence", convergence_figure),
        ("efficiency", efficiency_figure),
        ("distribution", distribution_figure),
        ("test_accuracy", test_accuracy_figure),
    ):
        written += save_figure(build(runs), out, name, formats)
    for dataset in runs.datasets:
        for series in runs.series_order:
            figure = pheromone_figure(runs, series, dataset)
            if figure is not None:
                safe = series.replace("/", "_")
                written += save_figure(figure, out, f"pheromones_{dataset}_{safe}", formats)
    table = summary_table(runs)
    table.round(4).to_csv(out / "summary_table.csv", index=False)
    (out / "summary_table.tex").write_text(table_to_latex(table), encoding="utf-8")
    return written + [out / "summary_table.csv", out / "summary_table.tex"]
