import numpy as np
import pandas as pd
import pytest

from ACO import DataSplit, Datasets
from ACO.Datasets import fit_preprocessor, load_dataset, make_split, stratified_indices


def frame(n=500, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.choice(["a", "b", "c"], size=n, p=[0.6, 0.3, 0.1])
    X = pd.DataFrame(
        {
            "num1": rng.normal(10, 5, n),
            "num2": rng.uniform(0, 100, n),
            "color": rng.choice(["red", "green", "blue"], n),
        }
    )
    return X, y


def test_stratified_indices_partition_the_rows_and_keep_proportions():
    y = np.repeat([0, 1, 2], [600, 300, 100])
    train, val, test = stratified_indices(y, 0.2, 0.2, seed=1)
    assert sorted(np.concatenate([train, val, test])) == list(range(1000))
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    for part, size in ((train, 600), (val, 200), (test, 200)):
        assert np.bincount(y[part]).tolist() == [int(size * 0.6), int(size * 0.3), int(size * 0.1)]


def test_split_is_deterministic_per_seed():
    y = np.random.default_rng(0).integers(0, 3, 300)
    first, again, other = (stratified_indices(y, 0.2, 0.2, seed) for seed in (7, 7, 8))
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not np.array_equal(first[0], other[0])


def test_invalid_fractions_are_rejected():
    for val, test in ((0, 0.2), (0.2, 0), (0.5, 0.5)):
        with pytest.raises(ValueError, match="sum to less than 1"):
            stratified_indices(np.zeros(10, dtype=int), val, test, 0)


def test_make_split_shapes_dtypes_and_standardization():
    X, y = frame()
    split = make_split(X, y)
    assert isinstance(split, DataSplit)
    assert split.n_features == 2 + 3 and split.n_classes == 3  # two numeric + three one-hot colours
    assert (len(split.y_train), len(split.y_val), len(split.y_test)) == (300, 100, 100)
    assert split.X_train.dtype == split.X_val.dtype == split.X_test.dtype == __import__("torch").float32
    train_numeric = split.X_train[:, :2].numpy()
    assert np.allclose(train_numeric.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(train_numeric.std(axis=0), 1, atol=1e-5)
    # validation and test use the training statistics, so they are close to but not exactly standardized
    assert not np.allclose(split.X_val[:, :2].numpy().mean(axis=0), 0, atol=1e-6)


def test_preprocessing_uses_training_statistics_only():
    train = pd.DataFrame({"x": [0.0, 10.0], "c": ["a", "b"]})
    other = pd.DataFrame({"x": [1000.0], "c": ["z"]})
    out = fit_preprocessor(train).transform(other)
    assert out[0, 0] == pytest.approx((1000.0 - 5.0) / 5.0)  # mean 5, std 5 from the training rows
    assert out[0, 1:].tolist() == [0.0, 0.0]  # unseen category is all zeros


def test_missing_values_are_imputed_from_training_data():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "c": ["a", "b", None, "a"]})
    pre = fit_preprocessor(train)
    out = pre.transform(pd.DataFrame({"x": [np.nan], "c": [None]}))
    assert not np.isnan(out).any()
    assert out[0, 0] == pytest.approx(0.0)  # NaN -> training median -> standardized to the training median's z-score
    assert pre.named_transformers_["categorical"][-1].categories_[0].tolist() == ["a", "b", "missing"]


def test_labels_become_contiguous_ids_even_for_numeric_scores():
    X, _ = frame(400)
    scores = np.random.default_rng(1).choice([3.0, 5.0, 6.0, 9.0], size=400)
    split = make_split(X, scores)
    assert split.n_classes == 4
    assert split.y_train.dtype == __import__("torch").int64
    assert set(split.y_train.tolist()) == {0, 1, 2, 3}


def test_min_class_count_drops_rare_classes():
    X, y = frame(400)
    y = y.astype(object)
    y[:2] = "rare"
    split = make_split(X, y, min_class_count=5)
    assert split.n_classes == 3
    assert len(split.y_train) + len(split.y_val) + len(split.y_test) == 398


def test_rare_class_gives_a_helpful_error():
    X, y = frame(400)
    y = y.astype(object)
    y[0] = "rare"
    with pytest.raises(ValueError, match="min_class_count"):
        make_split(X, y)


def test_missing_labels_are_rejected():
    X, y = frame(50)
    y = y.astype(object)
    y[0] = None
    with pytest.raises(ValueError, match="missing"):
        make_split(X, y)


def fake_openml(monkeypatch, X, y):
    calls = []

    def fetch(data_id, as_frame, return_X_y, data_home):
        calls.append((data_id, as_frame, return_X_y, data_home))
        return X.copy(), pd.Series(y)

    monkeypatch.setattr(Datasets, "fetch_openml", fetch)
    return calls


def test_load_dataset_adult_drops_fnlwgt_and_handles_missing(monkeypatch, tmp_path):
    X, y = frame(500)
    X["fnlwgt"] = np.random.default_rng(2).uniform(1e4, 1e6, 500)
    X.loc[::10, "color"] = None
    calls = fake_openml(monkeypatch, X, np.where(y == "a", "<=50K", ">50K"))
    split = load_dataset("adult", data_dir=tmp_path)
    assert calls == [(1590, True, True, str(tmp_path / "openml"))]
    assert split.n_features == 2 + 4  # fnlwgt gone; red, green, blue and the 'missing' category
    assert split.n_classes == 2
    assert not np.isnan(split.X_train.numpy()).any()


@pytest.mark.parametrize("name, openml_id", [("wine", 287), ("letter", 6)])
def test_load_dataset_other_datasets(monkeypatch, tmp_path, name, openml_id):
    X, y = frame(300)
    calls = fake_openml(monkeypatch, X.drop(columns="color"), y)
    split = load_dataset(name, seed=3, data_dir=tmp_path)
    assert calls[0][0] == openml_id
    assert split.n_features == 2


def test_unknown_dataset_name():
    with pytest.raises(ValueError, match="adult"):
        load_dataset("mnist")
