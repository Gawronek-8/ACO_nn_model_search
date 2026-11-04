"""Optional MLflow logging of experiment runs.

Every run (one seed of one experiment) becomes an MLflow run inside an MLflow experiment named like the
experiment config. The store is a SQLite file, so nothing has to be started to log; browse it with

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

from pathlib import Path

from .Observers import BaseObserver
from .Strategies.base import AntContext, IterationContext

FINAL_METRICS = (
    "best_val_accuracy", "best_val_loss", "best_n_params", "test_accuracy", "test_loss", "train_time_s", "search_time_s",
)


def flatten_params(values: dict, prefix: str = "") -> dict[str, str]:
    """{'colony': {'n_ants': 10}} -> {'colony.n_ants': '10'}, the shape MLflow params need."""
    flat = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{name}."))
        else:
            flat[name] = str(value)
    return flat


class MLflowTracker(BaseObserver):
    """Logs parameters, per-iteration scores and the final result of a run; also an observer of the colony.

    Call `start` before the search, then `finish` (or `fail`) after it.
    """

    def __init__(self, store_dir: str | Path):
        import mlflow  # imported here so MLflow is only needed when tracking is switched on

        self.mlflow = mlflow
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(self.store_dir / 'mlflow.db').resolve().as_posix()}")
        self._ants = 0

    def start(self, experiment: str, run_name: str, params: dict[str, str], tags: dict[str, str]) -> None:
        mlflow = self.mlflow
        if mlflow.get_experiment_by_name(experiment) is None:
            artifacts = (self.store_dir / "artifacts" / experiment).resolve()
            mlflow.create_experiment(experiment, artifact_location=artifacts.as_uri())
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        self._ants = 0

    def on_local_step_end(self, ctx: AntContext, pheromones) -> None:
        self.mlflow.log_metric("ant_val_accuracy", ctx.ant.score, step=self._ants)
        self._ants += 1

    def on_global_step_end(self, ctx: IterationContext, pheromones) -> None:
        scores = [ant.score for ant in ctx.ants]
        self.mlflow.log_metrics(
            {
                "iteration_best": max(scores),
                "iteration_mean": sum(scores) / len(scores),
                "best_so_far": ctx.best_ant.score,  # pyright: ignore[reportOptionalMemberAccess]
            },
            step=ctx.iteration,
        )

    def finish(self, summary: dict, run_dir: str | Path) -> None:
        """Logs the final numbers and the run folder (ants.csv, pheromones.npz, summary.json) as artifacts."""
        mlflow = self.mlflow
        mlflow.log_metrics({key: float(summary[key]) for key in FINAL_METRICS})
        mlflow.set_tag("best_layers", summary["best_layers"])
        mlflow.log_artifacts(str(run_dir))
        mlflow.end_run()

    def fail(self) -> None:
        self.mlflow.end_run("FAILED")
