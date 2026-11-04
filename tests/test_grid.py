import pandas as pd
import pytest
import torch

import grid
import main
from ACO import DataSplit, Experiment
from ACO.Experiment import ExperimentConfig, run_experiment
from ACO.Grid import aggregate, run_grid


def make_data(n=240, n_features=5, n_classes=3, seed=0) -> DataSplit:
    gen = torch.Generator().manual_seed(seed)

    def split(size):
        y = torch.randint(0, n_classes, (size,), generator=gen)
        return torch.randn(size, n_features, generator=gen) + 3.0 * y[:, None], y

    return DataSplit(*split(n), *split(n // 2), *split(n // 2))


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def fake_load(name, seed, min_class_count, data_dir):
        calls.append(name)
        return make_data()

    monkeypatch.setattr(Experiment, "load_dataset", fake_load)
    return calls


def small_config(**overrides) -> ExperimentConfig:
    config = {
        "name": "g",
        "dataset": "wine",
        "seeds": [0, 1],
        "colony": {"n_ants": 2, "n_iterations": 2, "search_space": {"max_depth": 2}},
        "evaluator": {"device": "cpu", "epochs": 4, "learning_rate": 1e-2},
    }
    config.update(overrides)
    return ExperimentConfig.model_validate(config)


def quiet(_):
    pass


def test_grid_runs_every_combination_and_loads_each_dataset_once(tmp_path, loads):
    summary = run_grid(small_config(), ["wine", "letter"], ["random", "independent"], tmp_path, log=quiet)
    assert len(summary) == 2 * 2 * 2
    assert set(zip(summary["dataset"], summary["strategy"])) == {
        ("wine", "random"), ("wine", "independent"), ("letter", "random"), ("letter", "independent")
    }
    assert sorted(loads) == ["letter", "wine"]
    assert pd.read_csv(tmp_path / "g" / "grid_summary.csv").shape[0] == 8
    assert (tmp_path / "g" / "letter_independent_s1" / "ants.csv").exists()


def test_aggregate_averages_over_seeds(tmp_path, loads):
    summary = run_grid(small_config(), ["wine"], ["random", "independent"], tmp_path, log=quiet)
    table = aggregate(summary)
    assert list(table.index) == [("wine", "random"), ("wine", "independent")]
    assert (table["runs"] == 2).all()
    assert {"val_accuracy", "test_accuracy", "test_std", "train_time_s"} <= set(table.columns)


def test_resume_skips_finished_runs_and_does_not_even_load_data(tmp_path, loads, monkeypatch):
    first = run_grid(small_config(), ["wine"], ["random"], tmp_path, log=quiet)
    loads.clear()

    def must_not_run(*args, **kwargs):
        raise AssertionError("a finished run was executed again")

    monkeypatch.setattr(Experiment, "run_seed", must_not_run)
    again = run_grid(small_config(), ["wine"], ["random"], tmp_path, log=quiet)
    assert loads == []
    assert again["test_accuracy"].tolist() == first["test_accuracy"].tolist()


def test_resume_picks_up_where_an_interrupted_run_stopped(tmp_path, loads):
    run_experiment(small_config(seeds=[0]), tmp_path, log=quiet)
    lines = []
    summary = run_experiment(small_config(seeds=[0, 1]), tmp_path, log=lines.append, resume=True)
    assert summary["seed"].tolist() == [0, 1]
    assert sum("already done" in line for line in lines) == 1


def test_changed_settings_are_rerun(tmp_path, loads, monkeypatch):
    run_grid(small_config(), ["wine"], ["random"], tmp_path, log=quiet)
    ran = []
    original = Experiment.run_seed
    monkeypatch.setattr(Experiment, "run_seed", lambda *args, **kwargs: ran.append(1) or original(*args, **kwargs))
    changed = small_config(evaluator={"device": "cpu", "epochs": 5, "learning_rate": 1e-2})
    run_grid(changed, ["wine"], ["random"], tmp_path, log=quiet)
    assert len(ran) == 2  # both seeds, because epochs differ from the saved runs


def test_no_resume_reruns_everything(tmp_path, loads, monkeypatch):
    run_grid(small_config(), ["wine"], ["random"], tmp_path, log=quiet)
    ran = []
    original = Experiment.run_seed
    monkeypatch.setattr(Experiment, "run_seed", lambda *args, **kwargs: ran.append(1) or original(*args, **kwargs))
    run_grid(small_config(), ["wine"], ["random"], tmp_path, log=quiet, resume=False)
    assert len(ran) == 2


def test_grid_cli_dry_run_only_prints_the_plan(tmp_path, loads, capsys):
    code = grid.main(
        ["configs/smoke.json", "--datasets", "all", "--strategies", "all", "--seeds", "0", "1", "2",
         "--out", str(tmp_path), "--dry-run"]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "36 runs, 4 models each, 144 in total" in printed  # 3 datasets x 4 strategies x 3 seeds
    assert list(tmp_path.iterdir()) == [] and loads == []


def test_grid_cli_runs_and_prints_the_table(tmp_path, loads, capsys):
    code = grid.main(
        ["configs/smoke.json", "--strategies", "random", "independent", "--seeds", "0", "--device", "cpu",
         "--name", "cli", "--out", str(tmp_path), "--data-dir", str(tmp_path)]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "mean over seeds" in printed and "independent" in printed
    assert (tmp_path / "cli" / "grid_summary.csv").exists()


@pytest.mark.skipif(torch.cuda.is_available() or torch.backends.mps.is_available(), reason="this machine has a GPU")
@pytest.mark.parametrize("entry", [grid, main])
def test_gpu_request_without_gpu_is_a_clean_error(tmp_path, loads, capsys, entry):
    code = entry.main(["configs/smoke.json", "--device", "gpu", "--out", str(tmp_path), "--data-dir", str(tmp_path)])
    assert code == 1
    assert "error:" in capsys.readouterr().err
