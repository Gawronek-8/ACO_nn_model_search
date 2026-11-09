import json

import pandas as pd
import pytest
import torch

import main
from ACO import DataSplit, Experiment
from ACO.Experiment import ExperimentConfig, ants_table, layers_text, run_experiment
from ACO.utils import LayerConfig


def make_data(n=240, n_features=5, n_classes=3, seed=0) -> DataSplit:
    gen = torch.Generator().manual_seed(seed)

    def split(size):
        y = torch.randint(0, n_classes, (size,), generator=gen)
        return torch.randn(size, n_features, generator=gen) + 3.0 * y[:, None], y

    return DataSplit(*split(n), *split(n // 2), *split(n // 2))


@pytest.fixture(autouse=True)
def synthetic_dataset(monkeypatch):
    calls = []

    def fake_load(name, seed, min_class_count, data_dir):
        calls.append((name, seed, min_class_count, data_dir))
        return make_data()

    monkeypatch.setattr(Experiment, "load_dataset", fake_load)
    return calls


def small_config(**overrides) -> ExperimentConfig:
    config = {
        "name": "t",
        "dataset": "wine",
        "seeds": [0, 1],
        "colony": {"n_ants": 3, "n_iterations": 2, "search_space": {"max_depth": 2}},
        "evaluator": {"device": "cpu", "epochs": 5, "learning_rate": 1e-2},
    }
    config.update(overrides)
    return ExperimentConfig.model_validate(config)


def test_config_defaults_and_budget():
    config = ExperimentConfig()
    assert config.n_models == 60
    assert small_config().n_models == 6


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="mnist"):
        ExperimentConfig(dataset="mnist")


def test_shipped_configs_are_valid():
    for name in ("default", "smoke", "pilot", "pilot_reuse_on", "pilot_reuse_off", "pilot_reuse_on_ep50", "pilot_reuse_off_ep50"):
        ExperimentConfig.from_json(f"configs/{name}.json")


def test_diagnostic_configs_differ_from_pilot_only_where_intended():
    pilot = ExperimentConfig.from_json("configs/pilot.json").model_dump()
    for reuse in ("on", "off"):
        for suffix, epochs in (("", 15), ("_ep50", 50)):
            diag = ExperimentConfig.from_json(f"configs/pilot_reuse_{reuse}{suffix}.json").model_dump()
            assert diag["name"] == f"diag_reuse_{reuse}{suffix}"
            assert diag["colony"].pop("reuse_weights") == (reuse == "on")
            assert diag["evaluator"].pop("epochs") == epochs
            expected = {**pilot, "name": diag["name"]}
            expected["colony"] = {k: v for k, v in pilot["colony"].items() if k != "reuse_weights"}
            expected["evaluator"] = {k: v for k, v in pilot["evaluator"].items() if k != "epochs"}
            assert diag == expected


def test_run_experiment_writes_one_folder_per_seed(tmp_path, synthetic_dataset):
    summary = run_experiment(small_config(), tmp_path, data_dir="cache", log=lambda _: None)
    assert synthetic_dataset == [("wine", 42, 0, "cache")]  # data is loaded once for all seeds
    assert list(summary["seed"]) == [0, 1]
    assert (summary["n_models"] == 6).all()
    assert (summary["test_accuracy"] > 0.5).all()
    for column in ("retrain_val_accuracy", "test_balanced_accuracy", "best_val_balanced_accuracy"):
        assert summary[column].between(0, 1).all()

    for seed in (0, 1):
        run_dir = tmp_path / "t" / f"wine_param_dependent_s{seed}"
        ants = pd.read_csv(run_dir / "ants.csv")
        assert len(ants) == 6
        assert ants["val_balanced_accuracy"].between(0, 1).all()
        assert ants["best_so_far"].is_monotonic_increasing
        details = json.loads((run_dir / "summary.json").read_text())
        assert details["config"]["colony"]["n_ants"] == 3
        assert details["best_path"] and details["seed"] == seed
    assert pd.read_csv(tmp_path / "t" / "summary_wine_param_dependent.csv").shape[0] == 2


def test_random_search_gets_the_same_budget(tmp_path):
    config = small_config()
    config = config.model_copy(update={"colony": config.colony.model_copy(update={"strategy": "random"})})
    summary = run_experiment(config, tmp_path, log=lambda _: None)
    assert (summary["n_models"] == config.n_models).all()
    assert (tmp_path / "t" / "wine_random_s0" / "ants.csv").exists()


def test_same_seed_gives_the_same_run(tmp_path):
    config = small_config(seeds=[3])
    first = run_experiment(config, tmp_path / "a", log=lambda _: None)
    second = run_experiment(config, tmp_path / "b", log=lambda _: None)
    assert first["best_layers"][0] == second["best_layers"][0]
    assert first["best_val_accuracy"][0] == second["best_val_accuracy"][0]


def test_progress_is_logged_per_iteration(tmp_path):
    lines = []
    run_experiment(small_config(seeds=[0]), tmp_path, log=lines.append)
    assert sum("iteration" in line for line in lines) == 2
    assert "test" in lines[-1]


def test_layers_text_and_ants_table_helpers():
    assert layers_text([]) == "(no hidden layers)"
    assert layers_text([LayerConfig(64, "relu", 0.1), LayerConfig(32, "tanh", 0.0)]) == "64-relu-0.1 | 32-tanh-0.0"
    assert len(ants_table([])) == 0


def test_cli_overrides_the_config(tmp_path, capsys):
    out = tmp_path / "out"
    code = main.main(
        ["configs/smoke.json", "--strategy", "independent", "--seeds", "5", "6", "--device", "cpu",
         "--name", "cli", "--out", str(out), "--data-dir", str(tmp_path)]
    )
    assert code == 0
    assert (out / "cli" / "wine_independent_s5" / "summary.json").exists()
    assert (out / "cli" / "wine_independent_s6" / "ants.csv").exists()
    printed = capsys.readouterr().out
    assert "independent" in printed and "over 2 seed(s)" in printed


def test_cli_rejects_unknown_choices(tmp_path):
    with pytest.raises(SystemExit):
        main.main(["configs/smoke.json", "--strategy", "magic"])
