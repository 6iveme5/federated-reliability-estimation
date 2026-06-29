# Federated Reliability Estimation

Code for the paper:

**Transferable and Deployable Reliability Estimation for Trustworthy Federated Healthcare AI**

This repository implements a pipeline for reliability estimation in federated
learning settings for clinical risk prediction:

```text
reliability transfer analysis
        -> transfer-stable teacher selection
        -> federated surrogate learning
        -> low-cost trustworthy deployment
```

## Research Questions

1. Which reliability estimators preserve centralized-learning behavior after
   adaptation to federated learning?
2. Do transferable reliability scores improve misclassification detection and
   high-risk prediction identification?
3. Can a federated surrogate model approximate a transfer-stable reliability
   teacher?
4. How do FedAvg, FedProx, and FedAdam affect surrogate reliability learning
   under native and synthetic non-IID partitions?

## Methods

- `GMM`: density-based reliability with centralized and federated mixture
  scoring.
- `RHH-kNN`: random-hyperplane hashing neighborhood reliability.
- `Embedding`: representation-space kNN reliability.
- `Class-aware hash teacher`: p-stable hash reliability conditioned on predicted
  class.
- `SurrogateMLP`: federated reliability surrogate trained with FedAvg, FedProx,
  or FedAdam.

## Baseline Taxonomy

| Group | Methods | FL role | Main limitation |
| --- | --- | --- | --- |
| FL-native confidence baselines | `pmax(FL)`, `Entropy(FL)`, `Margin(FL)` | Directly computed from the federated classifier output on each client. | Only measure model confidence; can fail under overconfidence or distribution shift. |
| Federated prototype baseline | `Centroid(FL)` | Clients upload per-class feature sums; server aggregates global centroids; clients score by distance to global class centroid. | A single centroid per class is too coarse for heterogeneous clinical data. |
| Local/direct reliability baselines | centroid distance, feature-space kNN, LOF | Computed locally from each client's reference data. | Local reference set may not represent the global FL distribution. |
| Proposed teacher-surrogate framework | class-aware hash teacher, `SurrogateMLP` | Hash teacher provides a strong neighborhood-style target; surrogate is trained with FedAvg/FedProx/FedAdam to approximate this target. | Small one-time FL training cost; enables very cheap local inference. |

## Data

Local data are intentionally ignored by git.

Expected layout:

```text
data/
  federated/            Native multi-hospital client CSV files
  raw/                  Central train/test CSV files
  chuc/
    CHUC.txt            External CHUC 6-month mortality dataset
  support2/
    support2.csv        Public SUPPORT2 in-hospital mortality dataset
```

The CHUC text matrix uses 10 input features and one binary target:

1. age
2. systolic BP
3. heart rate
4. Killip class
5. glomerular filtration ratio
6. albumin
7. PCR
8. maximum troponin
9. maximum creatinin
10. ST segment elevation
11. target: death after 6 months

CHUC is not a native federated dataset. It is evaluated with controlled
synthetic partitions: `iid`, `label_skew`, `feature_skew` by Killip class, and
`feature_skew` by ST elevation.

SUPPORT2 is a public critical-care prognosis dataset with 9,105 patients and 47
raw variables. We use `hospdead` as the binary in-hospital mortality target and
select 22 clinically relevant variables as predictors: age, sex, disease group,
disease class, number of comorbidities, diabetes, dementia, cancer status,
mean arterial pressure, white blood cell count, heart rate, respiratory rate,
temperature, PaO2/FiO2, albumin, bilirubin, creatinine, sodium, pH, glucose,
BUN, and urine output. Categorical variables are one-hot encoded, resulting in
39 features. SUPPORT2 is evaluated with controlled synthetic FL partitions:
`iid`, `label_skew`, `feature_skew` by disease class, and `feature_skew` by
cancer status.

## Layout

```text
fedrel_journal/          Core reusable package
  data.py                Dataset loaders and synthetic client partitions
  metrics.py             Approximation, detection, and risk metrics
  methods/               GMM, RHH, embedding, calibration, confidence baselines
  federated/             Federated aggregation utilities
  surrogate/             Teacher construction and federated surrogate learning
  overhead.py            Timing and communication-size utilities
experiments/             Reproducible experiment and plotting entry points
configs/                 YAML experiment configuration
outputs/                 Generated outputs, ignored by git
tests/                   Unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e ".[dev]"
```

Run checks:

```bash
python -m pytest -q
python -m ruff check .
```

## Single-Seed Experiments

Native federated dataset:

```bash
python experiments/run_transfer_analysis.py --dataset federated_csv --output-dir outputs/full/transfer_native
python experiments/run_surrogate_learning.py --dataset federated_csv --output-dir outputs/full/surrogate_native
```

CHUC examples:

```bash
python experiments/run_transfer_analysis.py --dataset chuc --partition iid --n-clients 4 --output-dir outputs/full/transfer_chuc_iid
python experiments/run_transfer_analysis.py --dataset chuc --partition label_skew --n-clients 4 --output-dir outputs/full/transfer_chuc_label_skew
python experiments/run_transfer_analysis.py --dataset chuc --partition feature_skew --feature-column killip_class --n-clients 4 --output-dir outputs/full/transfer_chuc_killip_skew
python experiments/run_transfer_analysis.py --dataset chuc --partition feature_skew --feature-column st_segment_elevation --n-clients 4 --output-dir outputs/full/transfer_chuc_st_skew
```

Surrogate learning (all three optimizers by default):

```bash
python experiments/run_surrogate_learning.py --dataset chuc --partition iid --n-clients 4 --output-dir outputs/full/surrogate_chuc_iid
```

## Multi-Seed Experiments

Five-seed run:

```bash
python experiments/run_multiseed.py --seeds 42,43,44,45,46 --output-dir outputs/multiseed_5_baselines
```

With controlled parallelism:

```bash
python experiments/run_multiseed.py --seeds 42,43,44,45,46 --jobs 4 --threads-per-job 2 --skip-existing --output-dir outputs/multiseed_5_baselines
```

Main outputs:

```text
outputs/multiseed_5_baselines/
  all_transfer_metrics.csv
  all_surrogate_metrics.csv
  summary_transfer_mean_std.csv
  summary_surrogate_mean_std.csv
  summary_surrogate_best_by_r2_mean.csv
  MULTISEED_SUMMARY.md
```

## Ablations

Hash-dimension and surrogate-size ablations:

```bash
python experiments/run_surrogate_ablation.py --seeds 42,43,44,45,46 --output-dir outputs/surrogate_ablation
python experiments/plot_ablation.py --input-dir outputs/surrogate_ablation --output-dir outputs/figures_ablation
```

## Overhead Analysis

Compute and communication overhead:

```bash
python experiments/run_overhead_analysis.py --dataset federated_csv --optimizer fedavg --seed 42 --output-dir outputs/overhead
python experiments/run_overhead_analysis.py --dataset chuc --partition iid --optimizer fedavg --seed 42 --output-dir outputs/overhead
python experiments/summarize_overhead.py --input-dir outputs/overhead --output-dir outputs/overhead
python experiments/run_communication_analysis.py --output-dir outputs/communication --seed 42
```

Five-seed timing (run serially to avoid CPU contention):

```bash
python experiments/run_multiseed_overhead.py --seeds 42,43,44,45,46 --output-dir outputs/overhead_multiseed
```

## Figures

```bash
python experiments/plot_results.py --multiseed-dir outputs/multiseed_5_baselines --full-dir outputs/full --output-dir outputs/figures_baselines
```

## Reproducibility Notes

- `outputs/`, `data/`, and model checkpoints are ignored by git.
- Multi-seed surrogate experiments can take several minutes on CPU.
- Run timing experiments on an otherwise idle machine to reduce measurement noise.
- `Risk@1%`, `Risk@5%`, and `Risk@10%` are reported in the surrogate tables.
