from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "patientGender",
    "patientAge",
    "glasgowScale",
    "hematocrit",
    "hemoglobin",
    "leucocitos",
    "lymphocytes",
    "urea",
    "creatinine",
    "platelets",
    "diuresis",
    "SBP",
    "DBP",
    "glasgowScale_missing",
]

TARGET_COLUMN = "outcomeType"

CHUC_FEATURE_COLUMNS = [
    "age",
    "systolic_bp",
    "heart_rate",
    "killip_class",
    "glomerular_filtration_ratio",
    "albumin",
    "pcr",
    "max_troponin",
    "max_creatinin",
    "st_segment_elevation",
]

CHUC_TARGET_COLUMN = "death_6_months"

SUPPORT2_FEATURE_COLUMNS = [
    "age",
    "sex",
    "dzgroup",
    "dzclass",
    "num.co",
    "diabetes",
    "dementia",
    "ca",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
]

SUPPORT2_CATEGORICAL_COLUMNS = ["sex", "dzgroup", "dzclass", "ca"]
SUPPORT2_TARGET_COLUMN = "hospdead"


@dataclass
class ClientDataset:
    client_id: int
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class FederatedDataset:
    clients: list[ClientDataset]
    x_central: np.ndarray | None = None
    y_central: np.ndarray | None = None


PartitionStrategy = Literal["iid", "label_skew", "feature_skew"]


def read_xy(
    csv_path: Path,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    feature_columns = feature_columns or FEATURE_COLUMNS
    missing = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int).to_numpy()
    return x, y


def load_federated_csv_dir(
    data_dir: Path,
    fit_global_scaler: bool = True,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> FederatedDataset:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No client CSV files found in {data_dir}")

    raw_clients = []
    for client_id, path in enumerate(files):
        x_df, y = read_xy(path, feature_columns=feature_columns, target_column=target_column)
        raw_clients.append((client_id, path.stem, x_df, y))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if fit_global_scaler:
        x_all = pd.concat([item[2] for item in raw_clients], ignore_index=True)
        imputer.fit(x_all)
        scaler.fit(imputer.transform(x_all))

    clients: list[ClientDataset] = []
    for client_id, name, x_df, y in raw_clients:
        if not fit_global_scaler:
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(imputer.fit_transform(x_df)).astype(np.float32)
        else:
            x_scaled = scaler.transform(imputer.transform(x_df)).astype(np.float32)
        clients.append(ClientDataset(client_id=client_id, name=name, x=x_scaled, y=y))

    x_central = np.vstack([client.x for client in clients])
    y_central = np.concatenate([client.y for client in clients])
    return FederatedDataset(clients=clients, x_central=x_central, y_central=y_central)


def load_chuc_txt(path: Path) -> pd.DataFrame:
    """Read the CHUC 6-month mortality text matrix into a named dataframe."""
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    expected_cols = len(CHUC_FEATURE_COLUMNS) + 1
    if data.ndim != 2 or data.shape[1] != expected_cols:
        raise ValueError(f"Expected CHUC matrix with {expected_cols} columns, got {data.shape}")
    columns = CHUC_FEATURE_COLUMNS + [CHUC_TARGET_COLUMN]
    df = pd.DataFrame(data, columns=columns)
    df[CHUC_TARGET_COLUMN] = df[CHUC_TARGET_COLUMN].astype(int)
    df["st_segment_elevation"] = df["st_segment_elevation"].astype(int)
    return df


def dataframe_to_federated_dataset(
    df: pd.DataFrame,
    client_indices: list[np.ndarray],
    feature_columns: list[str],
    target_column: str,
    fit_global_scaler: bool = True,
    client_prefix: str = "client",
) -> FederatedDataset:
    raw_clients = []
    for client_id, indices in enumerate(client_indices):
        client_df = df.iloc[np.asarray(indices)]
        x_df = client_df[feature_columns].apply(pd.to_numeric, errors="coerce")
        y = (
            pd.to_numeric(client_df[target_column], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        raw_clients.append((client_id, f"{client_prefix}_{client_id}", x_df, y))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    if fit_global_scaler:
        x_all = pd.concat([item[2] for item in raw_clients], ignore_index=True)
        imputer.fit(x_all)
        scaler.fit(imputer.transform(x_all))

    clients: list[ClientDataset] = []
    for client_id, name, x_df, y in raw_clients:
        if not fit_global_scaler:
            local_imputer = SimpleImputer(strategy="median")
            local_scaler = StandardScaler()
            x_scaled = local_scaler.fit_transform(local_imputer.fit_transform(x_df)).astype(
                np.float32
            )
        else:
            x_scaled = scaler.transform(imputer.transform(x_df)).astype(np.float32)
        clients.append(ClientDataset(client_id=client_id, name=name, x=x_scaled, y=y))

    x_central = np.vstack([client.x for client in clients])
    y_central = np.concatenate([client.y for client in clients])
    return FederatedDataset(clients=clients, x_central=x_central, y_central=y_central)


def encoded_dataframe_to_federated_dataset(
    x_df: pd.DataFrame,
    y: np.ndarray,
    client_indices: list[np.ndarray],
    fit_global_scaler: bool = True,
    client_prefix: str = "client",
) -> FederatedDataset:
    raw_clients = []
    for client_id, indices in enumerate(client_indices):
        idx = np.asarray(indices)
        raw_clients.append((client_id, f"{client_prefix}_{client_id}", x_df.iloc[idx], y[idx]))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    if fit_global_scaler:
        x_all = pd.concat([item[2] for item in raw_clients], ignore_index=True)
        imputer.fit(x_all)
        scaler.fit(imputer.transform(x_all))

    clients: list[ClientDataset] = []
    for client_id, name, x_local, y_local in raw_clients:
        if fit_global_scaler:
            x_scaled = scaler.transform(imputer.transform(x_local)).astype(np.float32)
        else:
            local_imputer = SimpleImputer(strategy="median")
            local_scaler = StandardScaler()
            x_scaled = local_scaler.fit_transform(local_imputer.fit_transform(x_local)).astype(
                np.float32
            )
        clients.append(ClientDataset(client_id=client_id, name=name, x=x_scaled, y=y_local))

    x_central = np.vstack([client.x for client in clients])
    y_central = np.concatenate([client.y for client in clients])
    return FederatedDataset(clients=clients, x_central=x_central, y_central=y_central)


def make_synthetic_client_indices(
    df: pd.DataFrame,
    n_clients: int,
    strategy: PartitionStrategy,
    target_column: str,
    feature_column: str | None = None,
    seed: int = 42,
) -> list[np.ndarray]:
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if strategy == "iid":
        return _iid_indices(df[target_column].to_numpy(), n_clients=n_clients, seed=seed)
    if strategy == "label_skew":
        return _label_skew_indices(df[target_column].to_numpy(), n_clients=n_clients, seed=seed)
    if strategy == "feature_skew":
        if feature_column is None:
            raise ValueError("feature_column is required for feature_skew partitioning")
        return _feature_skew_indices(df[feature_column].to_numpy(), n_clients=n_clients)
    raise ValueError(f"Unknown partition strategy: {strategy}")


def load_chuc_federated(
    path: Path,
    n_clients: int = 4,
    strategy: PartitionStrategy = "iid",
    feature_column: str | None = None,
    seed: int = 42,
    fit_global_scaler: bool = True,
) -> FederatedDataset:
    df = load_chuc_txt(path)
    indices = make_synthetic_client_indices(
        df,
        n_clients=n_clients,
        strategy=strategy,
        target_column=CHUC_TARGET_COLUMN,
        feature_column=feature_column,
        seed=seed,
    )
    return dataframe_to_federated_dataset(
        df,
        indices,
        feature_columns=CHUC_FEATURE_COLUMNS,
        target_column=CHUC_TARGET_COLUMN,
        fit_global_scaler=fit_global_scaler,
        client_prefix=f"chuc_{strategy}",
    )


def load_support2_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [
        column
        for column in SUPPORT2_FEATURE_COLUMNS + [SUPPORT2_TARGET_COLUMN]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing SUPPORT2 columns: {missing}")
    df = df.copy()
    df[SUPPORT2_TARGET_COLUMN] = (
        pd.to_numeric(df[SUPPORT2_TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    )
    return df


def load_support2_federated(
    path: Path,
    n_clients: int = 4,
    strategy: PartitionStrategy = "iid",
    feature_column: str | None = None,
    seed: int = 42,
    fit_global_scaler: bool = True,
) -> FederatedDataset:
    df = load_support2_csv(path)
    partition_column = feature_column
    if strategy == "feature_skew" and partition_column is None:
        partition_column = "dzclass"
    indices = make_synthetic_client_indices(
        df,
        n_clients=n_clients,
        strategy=strategy,
        target_column=SUPPORT2_TARGET_COLUMN,
        feature_column=partition_column,
        seed=seed,
    )
    x_df = encode_support2_features(df)
    y = df[SUPPORT2_TARGET_COLUMN].to_numpy(dtype=int)
    return encoded_dataframe_to_federated_dataset(
        x_df,
        y,
        indices,
        fit_global_scaler=fit_global_scaler,
        client_prefix=f"support2_{strategy}",
    )


def encode_support2_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SUPPORT2_FEATURE_COLUMNS].copy()
    categorical = [column for column in SUPPORT2_CATEGORICAL_COLUMNS if column in x.columns]
    for column in x.columns:
        if column not in categorical:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    x = pd.get_dummies(x, columns=categorical, dummy_na=True, dtype=float)
    return x


def write_client_csvs(
    df: pd.DataFrame,
    client_indices: list[np.ndarray],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for client_id, indices in enumerate(client_indices):
        df.iloc[np.asarray(indices)].to_csv(output_dir / f"client_{client_id}.csv", index=False)


def _iid_indices(y: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    splits = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(y))
    return [test_idx for _, test_idx in splits.split(dummy_x, y)]


def _label_skew_indices(y: np.ndarray, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]
    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        rng.shuffle(label_indices)
        proportions = rng.dirichlet(np.full(n_clients, 0.4))
        boundaries = np.cumsum(proportions)[:-1]
        chunks = np.split(label_indices, (boundaries * len(label_indices)).astype(int))
        for client_id, chunk in enumerate(chunks):
            client_indices[client_id].extend(chunk.tolist())

    result = []
    for indices in client_indices:
        arr = np.asarray(indices, dtype=int)
        rng.shuffle(arr)
        result.append(arr)
    return _rebalance_empty_clients(result, len(y), seed)


def _feature_skew_indices(feature: np.ndarray, n_clients: int) -> list[np.ndarray]:
    series = pd.Series(feature)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isnan(values), np.inf, values)
    else:
        codes = series.astype("category").cat.codes.to_numpy(dtype=float)
        values = np.where(codes < 0, np.inf, codes)
    order = np.argsort(values, kind="mergesort")
    return [chunk.astype(int) for chunk in np.array_split(order, n_clients)]


def _rebalance_empty_clients(
    client_indices: list[np.ndarray],
    n_samples: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    empty = [idx for idx, indices in enumerate(client_indices) if len(indices) == 0]
    if not empty:
        return client_indices

    sizes = np.asarray([len(indices) for indices in client_indices])
    for empty_client in empty:
        donor = int(np.argmax(sizes))
        moved_pos = int(rng.integers(0, sizes[donor]))
        moved = client_indices[donor][moved_pos]
        client_indices[donor] = np.delete(client_indices[donor], moved_pos)
        client_indices[empty_client] = np.asarray([moved], dtype=int)
        sizes[donor] -= 1
        sizes[empty_client] = 1

    assigned = np.concatenate(client_indices)
    if len(np.unique(assigned)) != n_samples:
        raise ValueError("Synthetic partitioning produced duplicate or missing samples")
    return client_indices
