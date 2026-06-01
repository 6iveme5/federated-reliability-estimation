import numpy as np

from fedrel_journal.surrogate.training import (
    FederatedSurrogateConfig,
    SurrogateClientDataset,
    train_federated_surrogate,
)


def _clients():
    rng = np.random.default_rng(42)
    clients = []
    for client_id in range(2):
        x = rng.normal(size=(12, 3)).astype(np.float32)
        y = (0.2 * x[:, 0] - 0.1 * x[:, 1] + 0.5).astype(np.float32)
        clients.append(
            SurrogateClientDataset(
                client_id=client_id,
                name=f"client_{client_id}",
                x=x,
                teacher_reliability=y,
            )
        )
    return clients


def test_train_federated_surrogate_supports_fedprox():
    config = FederatedSurrogateConfig(
        optimizer="fedprox",
        rounds=1,
        local_epochs=1,
        batch_size=4,
        seed=42,
    )

    result = train_federated_surrogate(_clients(), config=config)

    assert result.optimizer == "fedprox"
    assert len(result.history) == 1


def test_train_federated_surrogate_supports_fedadam():
    config = FederatedSurrogateConfig(
        optimizer="fedadam",
        rounds=1,
        local_epochs=1,
        batch_size=4,
        seed=42,
    )

    result = train_federated_surrogate(_clients(), config=config)

    assert result.optimizer == "fedadam"
    assert result.n_clients == 2
