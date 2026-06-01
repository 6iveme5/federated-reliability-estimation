from pathlib import Path

import numpy as np

from fedrel_journal.data import (
    CHUC_FEATURE_COLUMNS,
    CHUC_TARGET_COLUMN,
    load_chuc_txt,
    make_synthetic_client_indices,
)


def test_load_chuc_txt_names_columns(tmp_path: Path):
    path = tmp_path / "chuc.txt"
    np.savetxt(path, np.array([[1, 2, 3, 1, 5, 6, 7, 8, 9, 0, 1]], dtype=float))

    df = load_chuc_txt(path)

    assert list(df.columns) == CHUC_FEATURE_COLUMNS + [CHUC_TARGET_COLUMN]
    assert df[CHUC_TARGET_COLUMN].iloc[0] == 1


def test_synthetic_iid_partition_covers_all_rows():
    y = np.array([0, 1] * 12)
    import pandas as pd

    df = pd.DataFrame({"target": y})
    indices = make_synthetic_client_indices(
        df,
        n_clients=4,
        strategy="iid",
        target_column="target",
        seed=42,
    )

    assigned = np.concatenate(indices)
    assert sorted(assigned.tolist()) == list(range(len(y)))
    assert all(len(idx) > 0 for idx in indices)


def test_feature_skew_partition_sorts_by_feature():
    import pandas as pd

    df = pd.DataFrame({"feature": np.arange(10), "target": [0, 1] * 5})
    indices = make_synthetic_client_indices(
        df,
        n_clients=2,
        strategy="feature_skew",
        target_column="target",
        feature_column="feature",
    )

    assert df.iloc[indices[0]]["feature"].max() < df.iloc[indices[1]]["feature"].min()
