import mlflow
import numpy as np
import pytest
import torch
from mlflow import MlflowClient

import main
from ACO import Colony, ColonyConfig, DataSplit, Evaluator, EvaluatorConfig, Experiment, SearchSpace
from ACO.Experiment import ExperimentConfig, run_experiment
from ACO.Observers import PheromoneRecorder
from ACO.Tracking import MLflowTracker, flatten_params


def make_data(n=240, n_features=5, n_classes=3, seed=0) -> DataSplit:
    gen = torch.Generator().manual_seed(seed)

    def split(size):
        y = torch.randint(0, n_classes, (size,), generator=gen)
        return torch.randn(size, n_features, generator=gen) + 3.0 * y[:, None], y

    return DataSplit(*split(n), *split(n // 2), *split(n // 2))


@pytest.fixture
def synthetic_dataset(monkeypatch):
    monkeypatch.setattr(Experiment, "load_dataset", lambda name, seed, min_class_count, data_dir: make_data())


def small_config(strategy="param_dependent", **overrides) -> ExperimentConfig:
    config = {
        "name": "obs",
        "dataset": "wine",
        "seeds": [0],
        "colony": {"strategy": strategy, "n_ants": 3, "n_iterations": 3, "search_space": {"max_depth": 3}},
        "evaluator": {"device": "cpu", "epochs": 4, "learning_rate": 1e-2},
    }
    config.update(overrides)
    return ExperimentConfig.model_validate(config)


def quiet(_):
    pass


def run_colony(strategy: str, recorder: PheromoneRecorder):
    config = ColonyConfig(
        strategy=strategy,
        n_ants=3,
        n_iterations=3,
        search_space=SearchSpace(neurons=[8, 16], activation=["relu", "tanh"], dropout=[0.0, 0.1], max_depth=3),
    )
    evaluator = Evaluator(make_data(), EvaluatorConfig(device="cpu", epochs=4, learning_rate=1e-2))
    return Colony(config, evaluator, observers=recorder).run()


@pytest.mark.parametrize("strategy", ["independent", "param_dependent", "config_dependent"])
def test_recorder_keeps_one_snapshot_per_iteration_plus_the_start(strategy):
    recorder = PheromoneRecorder()
    run_colony(strategy, recorder)
    tables = recorder.tables()
    assert "type" in tables
    for table in tables.values():
        assert table.shape[0] == 3 + 1
        assert np.all(table[0] == 0.1)  # tau0 before anything happened
        assert not np.array_equal(table[-1], table[0])  # the colony changed something by the end


def test_snapshots_are_independent_copies():
    recorder = PheromoneRecorder()
    run_colony("independent", recorder)
    first = recorder.snapshots[0]["type"]
    assert np.all(first == 0.1)  # later updates did not leak into the stored start state


def test_random_search_records_nothing_and_saves_nothing(tmp_path):
    recorder = PheromoneRecorder()
    run_colony("random", recorder)
    assert recorder.tables() == {}
    assert recorder.save(tmp_path / "p.npz") is False
    assert not (tmp_path / "p.npz").exists()


def test_recorder_save_roundtrip(tmp_path):
    recorder = PheromoneRecorder()
    run_colony("param_dependent", recorder)
    assert recorder.save(tmp_path / "p.npz") is True
    with np.load(tmp_path / "p.npz") as saved:
        assert set(saved.files) == {"type", "neurons", "activation", "dropout"}
        assert np.array_equal(saved["neurons"], recorder.tables()["neurons"])


def test_experiment_runs_save_pheromones_for_aco_but_not_for_random(tmp_path, synthetic_dataset):
    run_experiment(small_config("param_dependent"), tmp_path, log=quiet)
    run_experiment(small_config("random"), tmp_path, log=quiet)
    assert (tmp_path / "obs" / "wine_param_dependent_s0" / "pheromones.npz").exists()
    assert (tmp_path / "obs" / "wine_random_s0" / "ants.csv").exists()
    assert not (tmp_path / "obs" / "wine_random_s0" / "pheromones.npz").exists()


def test_flatten_params():
    assert flatten_params({"a": 1, "b": {"c": [1, 2], "d": {"e": 0.5}}}) == {"a": "1", "b.c": "[1, 2]", "b.d.e": "0.5"}


def mlflow_runs(store, experiment):
    client = MlflowClient(f"sqlite:///{(store / 'mlflow.db').resolve().as_posix()}")
    found = client.get_experiment_by_name(experiment)
    assert found is not None
    return client, client.search_runs([found.experiment_id])


def test_mlflow_tracker_logs_params_metrics_and_artifacts(tmp_path, synthetic_dataset):
    store = tmp_path / "store"
    tracker = MLflowTracker(store)
    run_experiment(small_config(seeds=[0, 1]), tmp_path / "runs", log=quiet, tracker=tracker)
    client, runs = mlflow_runs(store, "obs")
    assert len(runs) == 2 and mlflow.active_run() is None
    run = next(r for r in runs if r.data.tags["seed"] == "0")
    assert run.info.status == "FINISHED"
    assert run.data.params["colony.n_ants"] == "3" and run.data.params["colony.strategy"] == "param_dependent"
    assert run.data.tags["dataset"] == "wine" and run.data.tags["best_layers"]
    assert 0 <= run.data.metrics["test_accuracy"] <= 1 and run.data.metrics["train_time_s"] > 0
    assert len(client.get_metric_history(run.info.run_id, "iteration_best")) == 3
    assert len(client.get_metric_history(run.info.run_id, "ant_val_accuracy")) == 9
    artifacts = {item.path for item in client.list_artifacts(run.info.run_id)}
    assert {"ants.csv", "summary.json", "pheromones.npz"} <= artifacts


def test_mlflow_run_is_marked_failed_when_the_run_crashes(tmp_path, synthetic_dataset, monkeypatch):
    def boom(self, path):
        raise RuntimeError("training exploded")

    monkeypatch.setattr(Evaluator, "final_test", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        run_experiment(small_config(), tmp_path / "runs", log=quiet, tracker=MLflowTracker(tmp_path / "store"))
    _, runs = mlflow_runs(tmp_path / "store", "obs")
    assert [r.info.status for r in runs] == ["FAILED"] and mlflow.active_run() is None


def test_cli_mlflow_flag(tmp_path, synthetic_dataset, capsys):
    code = main.main(
        ["configs/smoke.json", "--mlflow", "--mlflow-dir", str(tmp_path / "store"), "--device", "cpu",
         "--name", "cli_mlflow", "--out", str(tmp_path / "runs"), "--data-dir", str(tmp_path)]
    )
    assert code == 0
    _, runs = mlflow_runs(tmp_path / "store", "cli_mlflow")
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
