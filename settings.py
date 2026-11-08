from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"  # datasets are downloaded to DATA_DIR / "openml"
RUNS_DIR = ROOT_DIR / "runs"
MLFLOW_DIR = ROOT_DIR / "mlruns"  # SQLite store and artifacts of MLflow (--mlflow)
