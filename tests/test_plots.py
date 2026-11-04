import numpy as np
import pandas as pd
import pytest
import torch

import plots
from ACO import DataSplit, Experiment
from ACO.Experiment import ExperimentConfig
from ACO.Grid import run_grid
from ACO.Plots import (
    STRATEGY_STYLE,
    best_at_times,
    load_runs,
    make_all,
    pheromone_marginals,
    summary_table,
    table_to_latex,
)


def make_data(n=200, n_features=5, n_classes=3, seed=0) -> DataSplit:
    gen = torch.Generator().manual_seed(seed)

    def split(size):
        y = torch.randint(0, n_classes, (size,), generator=gen)
        return torch.randn(size, n_features, generator=gen) + 3.0 * y[:, None], y

    return DataSplit(*split(n), *split(n // 2), *split(n // 2))


def config(name="p", seeds=(0, 1)) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "dataset": "wine",
            "seeds": list(seeds),
            "colony": {"n_ants": 2, "n_iterations": 2, "search_space": {"max_depth": 2}},
            "evaluator": {"device": "cpu", "epochs": 3, "learning_rate": 1e-2},
        }
    )


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("runs")
    original = Experiment.load_dataset
    Experiment.load_dataset = lambda name, seed, min_class_count, data_dir: make_data()
    try:
        strategies = ["random", "independent", "config_dependent", "param_dependent"]
        run_grid(config(), ["wine", "letter"], strategies, path, log=lambda _: None)
        run_grid(config("p2"), ["wine"], ["independent"], path, log=lambda _: None)
    finally:
        Experiment.load_dataset = original
    return path


def test_best_at_times_is_a_step_function():
    times, best = np.array([1.0, 2.0, 4.0]), np.array([0.5, 0.7, 0.9])
    result = best_at_times(times, best, np.array([0.5, 1.0, 3.0, 4.0, 10.0]))
    assert np.isnan(result[0])
    assert result[1:].tolist() == [0.5, 0.7, 0.9, 0.9]


def test_load_runs_names_series_after_strategies(runs_dir):
    runs = load_runs(["p"], runs_dir)
    assert runs.series_order == ["param_dependent", "independent", "config_dependent", "random"]
    assert runs.datasets == ["wine", "letter"]
    assert len(runs.summary) == 2 * 4 * 2
    assert len(runs.ants) == 2 * 4 * 2 * 4
    assert runs.styles["independent"] == STRATEGY_STYLE["independent"]


def test_several_experiments_become_series_of_their_own(runs_dir):
    runs = load_runs(["p", "p2"], runs_dir)
    assert "p/random" in runs.series_order and "p2" in runs.series_order
    assert len(set(runs.styles.values())) == len(runs.series_order)


def test_missing_experiment_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No finished runs"):
        load_runs(["nope"], tmp_path)


def test_pheromones_exist_only_for_aco(runs_dir):
    runs = load_runs(["p"], runs_dir)
    assert pheromone_marginals(runs, "random", "wine") is None
    for strategy in ("independent", "param_dependent", "config_dependent"):
        panels = pheromone_marginals(runs, strategy, "wine")
        assert set(panels) == {"type", "neurons", "activation", "dropout"}
        assert panels["type"][0].shape == (2, 2)
        assert panels["neurons"][0].shape[1] == len(panels["neurons"][1])


def test_summary_table_and_latex(runs_dir):
    runs = load_runs(["p"], runs_dir)
    table = summary_table(runs)
    assert len(table) == 2 * 4
    assert (table["runs"] == 2).all() and (table["models"] == 4).all()
    tex = table_to_latex(table)
    assert tex.startswith(r"\begin{tabular}") and r"\toprule" in tex and r"config\_dependent" in tex


def test_make_all_writes_png_and_pdf(runs_dir, tmp_path):
    files = make_all(load_runs(["p"], runs_dir), tmp_path)
    names = {path.name for path in files}
    for base in ("convergence", "efficiency", "distribution", "test_accuracy", "pheromones_wine_independent"):
        assert f"{base}.png" in names and f"{base}.pdf" in names
    assert not any("random" in name and name.startswith("pheromones") for name in names)
    assert all(path.exists() and path.stat().st_size > 0 for path in files)
    assert (tmp_path / "convergence.pdf").read_bytes().startswith(b"%PDF")
    assert pd.read_csv(tmp_path / "summary_table.csv").shape[0] == 8


def test_cli_defaults_to_plots_folder_of_the_experiment(runs_dir, capsys):
    assert plots.main(["p", "--runs-dir", str(runs_dir), "--format", "png"]) == 0
    assert (runs_dir / "p" / "plots" / "convergence.png").exists()
    assert not (runs_dir / "p" / "plots" / "convergence.pdf").exists()


def test_cli_reports_missing_runs(tmp_path, capsys):
    assert plots.main(["nope", "--runs-dir", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err
