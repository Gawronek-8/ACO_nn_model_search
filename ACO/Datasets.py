"""Loading the benchmark datasets and turning them into a DataSplit.

Pipeline for every dataset: download from OpenML (cached by scikit-learn in `<data_dir>/openml`),
stratified 60/20/20 split, then preprocessing fitted on the training part only so nothing leaks from
validation or test. Numeric columns are median-imputed and standardized, categorical columns get a
"missing" category and one-hot encoding (categories unseen in training become all zeros).

Run `python -m ACO.Datasets` to download everything and print a summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .Data import DataSplit


@dataclass(frozen=True)
class DatasetSpec:
    openml_id: int
    drop: tuple[str, ...] = ()  # columns that are not features


DATASETS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(1590, drop=("fnlwgt",)),  # census sampling weight, not a property of the person
    "wine": DatasetSpec(287),  # red and white Vinho Verde together, target = quality score
    "letter": DatasetSpec(6),
}


def load_dataset(
    name: str,
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
    min_class_count: int = 0,
    data_dir: str | Path | None = None,
) -> DataSplit:
    """Downloads (or reads from cache) one dataset and returns the preprocessed split.

    `seed` only controls the split, so the same seed gives every method the same data.
    `min_class_count` drops classes with fewer samples (0 keeps everything); Wine Quality has classes with
    a handful of samples.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}', expected one of {sorted(DATASETS)}")
    spec = DATASETS[name]
    data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
    X, y = fetch_openml(data_id=spec.openml_id, as_frame=True, return_X_y=True, data_home=str(data_dir / "openml"))
    X = X.drop(columns=list(spec.drop))
    return make_split(X, y, seed, val_size, test_size, min_class_count)


def make_split(
    X: pd.DataFrame,
    y,
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
    min_class_count: int = 0,
) -> DataSplit:
    """Stratified split + leak-free preprocessing of a raw feature frame and its labels."""
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    if pd.isna(y).any():
        raise ValueError("Labels contain missing values")

    if min_class_count > 0:
        values, counts = np.unique(y, return_counts=True)
        keep = np.isin(y, values[counts >= min_class_count])
        X, y = X[keep].reset_index(drop=True), y[keep]

    classes, y_ids = np.unique(y, return_inverse=True)  # class ids 0 .. n_classes - 1, sorted by label
    train, val, test = stratified_indices(y_ids, val_size, test_size, seed)

    preprocessor = fit_preprocessor(X.iloc[train])
    features = [preprocessor.transform(_clean(X.iloc[idx])) for idx in (train, val, test)]
    labels = [y_ids[idx] for idx in (train, val, test)]
    return DataSplit.from_numpy(features[0], labels[0], features[1], labels[1], features[2], labels[2])


def stratified_indices(
    y_ids: np.ndarray, val_size: float, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices of the train / validation / test parts, each keeping the class proportions."""
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    rows = np.arange(len(y_ids))
    held_out = val_size + test_size
    try:
        train, rest = train_test_split(rows, test_size=held_out, stratify=y_ids, random_state=seed)
        val, test = train_test_split(rest, test_size=test_size / held_out, stratify=y_ids[rest], random_state=seed)
    except ValueError as error:
        rarest = int(np.bincount(y_ids).min())
        raise ValueError(
            f"Stratified split failed ({error}). The rarest class has {rarest} samples; "
            "use min_class_count to drop rare classes."
        ) from error
    return train, val, test


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Median-impute + standardize numeric columns, 'missing' + one-hot for the rest. Fitted on X_train only."""
    X_train = _clean(X_train)
    numeric = list(X_train.select_dtypes("number").columns)
    categorical = [column for column in X_train.columns if column not in numeric]
    preprocessor = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    return preprocessor.fit(X_train)


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    """Makes non-numeric columns plain objects with np.nan for gaps, which the imputers understand."""
    X = X.copy()
    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            values = X[column].astype(object)
            X[column] = values.where(values.notna(), np.nan)
    return X


def _default_data_dir() -> Path:
    from settings import DATA_DIR  # repo-root module, importable when running from the repo root

    return DATA_DIR


def describe(name: str, split: DataSplit) -> str:
    sizes = {part: len(getattr(split, f"y_{part}")) for part in ("train", "val", "test")}
    counts = np.bincount(split.y_train.numpy(), minlength=split.n_classes)
    return (
        f"{name}: {split.n_features} features, {split.n_classes} classes, "
        f"rows {sizes}, train class counts min/max {counts.min()}/{counts.max()}"
    )


if __name__ == "__main__":
    for dataset in DATASETS:
        print(describe(dataset, load_dataset(dataset)))
