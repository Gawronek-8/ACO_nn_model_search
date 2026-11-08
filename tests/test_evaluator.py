import pytest
import torch
import torch.nn as nn

from ACO import Colony, ColonyConfig, DataSplit, Evaluator, EvaluatorConfig, LayerConfig, SearchSpace
from ACO.Evaluator import resolve_device

A = LayerConfig(16, "relu", 0.0)
B = LayerConfig(8, "tanh", 0.1)
C = LayerConfig(8, "elu", 0.0)


@pytest.fixture(autouse=True)
def seed_torch():
    torch.manual_seed(0)  # model init, batch shuffling and dropout all draw from torch's global RNG


def make_data(n=300, n_features=6, n_classes=3, seed=0) -> DataSplit:
    """Separable blobs: class k is centred on k * 3 along every feature."""
    gen = torch.Generator().manual_seed(seed)

    def split(size):
        y = torch.randint(0, n_classes, (size,), generator=gen)
        X = torch.randn(size, n_features, generator=gen) + 3.0 * y[:, None]
        return X, y

    return DataSplit(*split(n), *split(n // 2), *split(n // 2))


def make_evaluator(**config) -> Evaluator:
    settings = {"device": "cpu", "epochs": 40, "learning_rate": 1e-2, **config}
    return Evaluator(make_data(), EvaluatorConfig(**settings))

def test_build_model_layout():
    model = make_evaluator().build_model([A, B])
    kinds = [type(m) for m in model]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.Tanh, nn.Dropout, nn.Linear]
    assert (model[0].in_features, model[0].out_features) == (6, 16)
    assert (model[2].in_features, model[2].out_features) == (16, 8)
    assert (model[5].in_features, model[5].out_features) == (8, 3)


def test_build_model_empty_path_is_linear_classifier():
    assert [type(m) for m in make_evaluator().build_model([])] == [nn.Linear]


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="swish"):
        make_evaluator().build_model([LayerConfig(8, "swish", 0.0)])


def test_evaluate_path_learns_and_reports_everything():
    result = make_evaluator().evaluate_path([A, B], reuse_weights=False)
    assert result.val_accuracy > 0.8
    assert result.val_loss > 0 and result.train_time > 0
    assert result.n_params == (6 * 16 + 16) + (16 * 8 + 8) + (8 * 3 + 3)


def test_early_stopping_keeps_best_epoch_and_can_be_disabled():
    # patience=1 stops at the first epoch without improvement, so it may train very little; both must still learn
    for patience, minimum in ((1, 0.7), (None, 0.8)):
        result = make_evaluator(patience=patience).evaluate_path([A], reuse_weights=False)
        assert result.val_accuracy > minimum  # chance level is 1/3


def test_reuse_off_stores_nothing():
    evaluator = make_evaluator()
    evaluator.evaluate_path([A, B], reuse_weights=False)
    assert evaluator.weights == {}


def test_reuse_warm_starts_longest_matching_prefix():
    evaluator = make_evaluator()
    evaluator.evaluate_path([A, B], reuse_weights=True)
    assert set(evaluator.weights) == {(A,), (A, B)}

    model = evaluator.build_model([A, B, C])
    assert evaluator.load_reusable_weights([A, B, C], model) == 2
    stored = evaluator.weights[(A, B)]
    assert torch.equal(model[0].weight.cpu(), stored.hidden[0]["weight"])
    assert torch.equal(model[2].weight.cpu(), stored.hidden[1]["weight"])
    assert not torch.equal(model[-1].weight.cpu(), stored.head["weight"])  # output layer only for identical paths

    other = evaluator.build_model([C])
    assert evaluator.load_reusable_weights([C], other) == 0


def test_identical_path_also_restores_output_layer():
    evaluator = make_evaluator()
    evaluator.evaluate_path([A], reuse_weights=True)
    model = evaluator.build_model([A])
    assert evaluator.load_reusable_weights([A], model) == 1
    assert torch.equal(model[-1].weight.cpu(), evaluator.weights[(A,)].head["weight"])


def test_store_keeps_only_the_better_model_per_prefix():
    evaluator = make_evaluator()
    model = evaluator.build_model([A])
    worse, better = (evaluator.evaluate(model) for _ in range(2))
    worse.val_accuracy, better.val_accuracy = 0.5, 0.9
    evaluator.store_weights([A], model, better)
    with torch.no_grad():
        model[0].weight.add_(1.0)
    evaluator.store_weights([A], model, worse)
    assert evaluator.weights[(A,)].val_accuracy == 0.9
    assert not torch.equal(model[0].weight.cpu(), evaluator.weights[(A,)].hidden[0]["weight"])


def test_final_test_scores_the_test_split():
    result = make_evaluator().final_test([A, B])
    assert result.val_accuracy > 0.8


def test_device_selection():
    assert resolve_device("cpu").type == "cpu"
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        assert resolve_device("auto").type == "cpu"
        with pytest.raises(RuntimeError, match="gpu"):
            resolve_device("gpu")


def test_data_split_validates_shapes_and_labels():
    d = make_data()
    with pytest.raises(ValueError, match="features"):
        DataSplit(d.X_train, d.y_train, d.X_val[:, :3], d.y_val, d.X_test, d.y_test)
    with pytest.raises(ValueError, match="int64"):
        DataSplit(d.X_train, d.y_train.float(), d.X_val, d.y_val, d.X_test, d.y_test)
    assert d.n_features == 6 and d.n_classes == 3


def test_colony_requires_an_evaluator():
    with pytest.raises(ValueError, match="Evaluator"):
        Colony(ColonyConfig())


@pytest.mark.parametrize("strategy", ["independent", "param_dependent", "config_dependent", "random"])
def test_colony_runs_end_to_end_with_every_strategy(strategy):
    config = ColonyConfig(
        strategy=strategy,
        n_ants=3,
        n_iterations=3,
        search_space=SearchSpace(neurons=[8, 16], activation=["relu", "tanh"], dropout=[0.0, 0.1], max_depth=3),
    )
    best = Colony(config, make_evaluator()).run()
    assert best is not None and best.score > 0.5
