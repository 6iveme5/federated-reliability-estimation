import numpy as np

from fedrel_journal.data import ClientDataset
from fedrel_journal.surrogate.training import build_hash_teacher_surrogate_clients
from fedrel_journal.task import (
    FederatedTaskConfig,
    TaskClientSplit,
    predict_binary_classifier,
    train_federated_binary_classifier,
)


def _clients():
    rng = np.random.default_rng(7)
    clients = []
    for client_id in range(2):
        x = rng.normal(size=(24, 4)).astype(np.float32)
        y = (x[:, 0] + 0.4 * x[:, 1] > 0).astype(int)
        clients.append(ClientDataset(client_id=client_id, name=f"client_{client_id}", x=x, y=y))
    return clients


def test_federated_binary_classifier_predicts_probabilities():
    protocol = build_hash_teacher_surrogate_clients(
        _clients(),
        include_federated_confidence=True,
        federated_task_config=FederatedTaskConfig(rounds=1, local_epochs=1, seed=7),
        seed=7,
    )
    surrogate_clients = protocol.eval_clients

    assert "pmax_fl" in surrogate_clients[0].baselines
    assert "entropy_fl" in surrogate_clients[0].baselines
    assert "margin_fl" in surrogate_clients[0].baselines
    assert "centroid_fl" in surrogate_clients[0].baselines
    assert "pmax_fl" in surrogate_clients[0].baseline_errors
    assert "centroid_fl" in surrogate_clients[0].baseline_errors
    assert np.array_equal(surrogate_clients[0].errors, surrogate_clients[0].baseline_errors["pmax_fl"])
    assert surrogate_clients[0].task_y_true is not None
    assert surrogate_clients[0].task_pred is not None
    assert surrogate_clients[0].task_proba is not None
    assert np.array_equal(
        surrogate_clients[0].errors,
        (surrogate_clients[0].task_pred != surrogate_clients[0].task_y_true).astype(int),
    )
    assert np.array_equal(
        surrogate_clients[0].errors,
        surrogate_clients[0].baseline_errors["centroid_fl"],
    )


def test_train_federated_binary_classifier_directly():
    splits = []
    for client in _clients():
        splits.append(
            TaskClientSplit(
                client_id=client.client_id,
                name=client.name,
                x_train=client.x[:16],
                y_train=client.y[:16],
                x_val=client.x[16:],
                y_val=client.y[16:],
            )
        )
    model = train_federated_binary_classifier(
        splits,
        FederatedTaskConfig(rounds=1, local_epochs=1, seed=7),
    )
    pred, proba = predict_binary_classifier(model, splits[0].x_val)

    assert pred.shape == splits[0].y_val.shape
    assert proba.shape == (len(splits[0].y_val), 2)
