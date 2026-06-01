# FedRel Journal Reliability

This repository contains the journal-paper codebase for a unified study on
transferable and deployable reliability estimation in federated healthcare AI.

Working title:

**Transferable and Deployable Reliability Estimation for Trustworthy Federated Healthcare AI**

The project combines two earlier research threads:

- DSN reliability transfer: centralized-to-federated behavior of point-wise
  reliability estimators.
- ISSRE surrogate reliability: federated surrogate learning for low-cost
  deployment of reliability scores.

The journal contribution is not a direct concatenation of those papers. The
main new story is **transfer-guided deployability**: first identify reliability
signals that remain valid under federated deployment, then use the
transfer-stable signal as a teacher for low-cost federated surrogate learning.

## Relation to Prior Conference Work

| Component | DSN transfer paper | ISSRE surrogate paper | This journal study |
| --- | --- | --- | --- |
| CL-to-FL reliability transfer | yes | no | yes, used to guide teacher selection |
| GMM/RHH/embedding comparison | yes | no | yes, extended to CHUC partitions |
| Embedding negative transfer | yes | no | yes, used as a design argument |
| Class-aware hash teacher | no | yes | yes, motivated by transfer stability |
| Federated surrogate learning | no | yes | yes, with FedAvg/FedProx/FedAdam |
| External CHUC validation | no | no | yes, controlled synthetic FL partitions |
| Multi-seed mean/std | limited | limited | yes, five seeds |
| Compute/communication overhead | limited | partial | yes, integrated into the framework |

In short, the DSN work establishes **which reliability signals transfer**, the
ISSRE work establishes **that hash reliability can be distilled**, and this
repository connects them into a single pipeline:

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

## Current Methods

- `GMM`: density-based reliability with centralized and federated mixture
  scoring.
- `RHH-kNN`: random-hyperplane hashing neighborhood reliability.
- `Embedding`: representation-space kNN reliability used as a negative-transfer
  baseline.
- `Class-aware hash teacher`: p-stable hash reliability conditioned on predicted
  class.
- `SurrogateMLP`: federated reliability surrogate trained with FedAvg, FedProx,
  or FedAdam.

## Baseline Taxonomy

The baseline comparison is organized by FL compatibility rather than treating
all methods as newly federated algorithms.

| Group | Methods | FL role | Main limitation |
| --- | --- | --- | --- |
| FL-native confidence baselines | `pmax(FL)`, entropy(FL), margin(FL) | Directly computed from the federated classifier output on each client. No extra FL adaptation is needed. | Cheap and deployable, but they only measure model confidence and can fail under overconfidence or distribution shift. |
| Federated prototype baseline | `Centroid(FL)` | Clients upload per-class feature sums and counts; the server aggregates global class centroids; clients score samples by distance to the predicted global class centroid. | Truly federated and cheap, but a single centroid per class is too coarse for heterogeneous clinical data. |
| Local confidence baselines | `pmax`, entropy, margin | Computed from local task models. These are kept for comparison with the earlier non-FL confidence setting. | Not the main FL baseline because the task model is not federated. |
| Local/direct reliability baselines | centroid distance, feature-space kNN, LOF | Computed locally from each client's reference data and predictions. | Easy to run locally, but the local reference set may not represent the global FL data distribution. Global versions would require sharing features, prototypes, or repeated cross-client queries. |
| Proposed teacher-surrogate framework | class-aware hash teacher, `SurrogateMLP` | The hash teacher provides a strong local neighborhood-style target; the surrogate is trained with FedAvg, FedProx, or FedAdam to approximate this target. | The surrogate adds a small one-time FL training communication cost, but enables very cheap local inference. |

Thus, `pmax(FL)`, entropy(FL), and margin(FL) are included because they are the
simplest and most practical FL baselines, not because they require special FL
adaptation. The paper's main claim is that federated surrogate learning can
preserve much of a stronger neighborhood/hash reliability signal while remaining
deployable.

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
39 features in the current preprocessing. SUPPORT2 is evaluated with controlled
synthetic FL partitions: `iid`, `label_skew`, `feature_skew` by disease class,
and `feature_skew` by cancer status.

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
docs/                    Paper plan and notes
outputs/                 Generated outputs, ignored by git
tests/                   Unit tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run checks:

```powershell
python -m pytest -q
python -m ruff check .
```

Current status:

```text
pytest: 7 passed
ruff: All checks passed
```

## Single-Seed Experiments

Native federated dataset:

```powershell
python experiments\run_transfer_analysis.py --dataset federated_csv --output-dir outputs\full\transfer_native
python experiments\run_surrogate_learning.py --dataset federated_csv --output-dir outputs\full\surrogate_native
```

CHUC examples:

```powershell
python experiments\run_transfer_analysis.py --dataset chuc --partition iid --n-clients 4 --output-dir outputs\full\transfer_chuc_iid
python experiments\run_transfer_analysis.py --dataset chuc --partition label_skew --n-clients 4 --output-dir outputs\full\transfer_chuc_label_skew
python experiments\run_transfer_analysis.py --dataset chuc --partition feature_skew --feature-column killip_class --n-clients 4 --output-dir outputs\full\transfer_chuc_killip_skew
python experiments\run_transfer_analysis.py --dataset chuc --partition feature_skew --feature-column st_segment_elevation --n-clients 4 --output-dir outputs\full\transfer_chuc_st_skew
```

Surrogate learning uses all three optimizers by default:

```powershell
python experiments\run_surrogate_learning.py --dataset chuc --partition iid --n-clients 4 --output-dir outputs\full\surrogate_chuc_iid
```

## Multi-Seed Experiments

Run the five-seed experiment used for the current summary tables:

```powershell
python experiments\run_multiseed.py --seeds 42,43,44,45,46 --output-dir outputs\multiseed_5_baselines
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

The surrogate experiment now adds FL-native confidence baselines by default:
`pmax_fl`, `entropy_fl`, `margin_fl`, and `centroid_fl`. The confidence scores
are produced by a FedAvg-trained task classifier and should be labeled in the
paper as `pmax(FL)`, `Entropy(FL)`, and `Margin(FL)`. `centroid_fl` is the
federated global-prototype baseline and should be labeled `Centroid(FL)`. The
earlier `pmax`, entropy, margin, and centroid columns remain in the CSVs as
local baselines for continuity.

The current five-seed FL-confidence run is stored in:

```text
outputs/multiseed_5_fl_confidence/
  all_surrogate_metrics.csv
  summary_surrogate_mean_std.csv
  summary_surrogate_best_by_r2_mean.csv
  MULTISEED_SUMMARY.md
```

The current five-seed run with `Centroid(FL)` is stored in:

```text
outputs/multiseed_5_centroid_fl/
  all_surrogate_metrics.csv
  summary_surrogate_mean_std.csv
  summary_surrogate_best_by_r2_mean.csv
  MULTISEED_SUMMARY.md
```

The SUPPORT2 five-seed run is stored in:

```text
outputs/support2_multiseed_5/
  all_transfer_metrics.csv
  all_surrogate_metrics.csv
  summary_transfer_mean_std.csv
  summary_surrogate_mean_std.csv
  summary_surrogate_best_by_r2_mean.csv
  MULTISEED_SUMMARY.md
```

## Ablations

Run hash-dimension and surrogate-size ablations:

```powershell
python experiments\run_surrogate_ablation.py --seeds 42,43,44,45,46 --output-dir outputs\surrogate_ablation
python experiments\plot_ablation.py --input-dir outputs\surrogate_ablation --output-dir outputs\figures_ablation
```

Main outputs:

```text
outputs/surrogate_ablation/
  surrogate_ablation_metrics.csv
  surrogate_ablation_summary.csv
  SURROGATE_ABLATION_SUMMARY.md
outputs/figures_ablation/
  hash_dim_r2.png/pdf
  hash_dim_auroc.png/pdf
  model_size_r2.png/pdf
  model_size_auroc.png/pdf
```

## Overhead Analysis

Run representative compute and communication overhead experiments:

```powershell
python experiments\run_overhead_analysis.py --dataset federated_csv --optimizer fedavg --seed 42 --output-dir outputs\overhead
python experiments\run_overhead_analysis.py --dataset chuc --partition iid --optimizer fedavg --seed 42 --output-dir outputs\overhead
python experiments\run_overhead_analysis.py --dataset chuc --partition label_skew --optimizer fedprox --seed 42 --output-dir outputs\overhead
python experiments\run_overhead_analysis.py --dataset chuc --partition feature_skew --feature-column killip_class --optimizer fedprox --seed 42 --output-dir outputs\overhead
python experiments\run_overhead_analysis.py --dataset chuc --partition feature_skew --feature-column st_segment_elevation --optimizer fedprox --seed 42 --output-dir outputs\overhead
python experiments\summarize_overhead.py --input-dir outputs\overhead --output-dir outputs\overhead
python experiments\run_communication_analysis.py --output-dir outputs\communication --seed 42
```

Main outputs:

```text
outputs/overhead/
  overhead_summary.csv
  OVERHEAD_SUMMARY.md
  overhead_compute_time.png/pdf
  overhead_communication.png/pdf
```

Communication-specific outputs:

```text
outputs/communication/
  communication_overhead.csv
  COMMUNICATION_SUMMARY.md
  surrogate_training_communication.png/pdf
  native_method_communication.png/pdf
```

## Figures

Generate paper-ready PNG/PDF figures:

```powershell
python experiments\plot_results.py --multiseed-dir outputs\multiseed_5_baselines --full-dir outputs\full --output-dir outputs\figures_baselines
python experiments\plot_results.py --multiseed-dir outputs\multiseed_5_fl_confidence --full-dir outputs\full --output-dir outputs\figures_fl_confidence
python experiments\plot_results.py --multiseed-dir outputs\multiseed_5_centroid_fl --full-dir outputs\full --output-dir outputs\figures_centroid_fl
```

Generated figures:

```text
outputs/figures/
  transfer_spearman_heatmap.png/pdf
  transfer_scatter_examples.png/pdf
  surrogate_optimizer_r2.png/pdf
  error_detection_auroc.png/pdf
  surrogate_convergence_r2.png/pdf
  native_transfer_scatter.png/pdf
  native_surrogate_optimizer_r2.png/pdf
  native_error_detection_auroc.png/pdf
  baseline_auroc_heatmap.png/pdf
```

## Current Key Results

Five-seed summary with FL confidence and `Centroid(FL)` baselines:

| Scenario | Best optimizer | R2 | Centroid(FL) AUROC | pmax(FL) AUROC | Surrogate AUROC | Teacher AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Multi-hospital | FedAvg | 0.770 +/- 0.017 | 0.588 +/- 0.010 | 0.717 +/- 0.022 | 0.829 +/- 0.005 | 0.838 |
| CHUC-IID | FedProx | 0.757 +/- 0.049 | 0.574 +/- 0.048 | 0.670 +/- 0.011 | 0.709 +/- 0.012 | 0.739 |
| CHUC label-skew | FedProx | 0.377 +/- 0.088 | 0.621 +/- 0.081 | 0.688 +/- 0.049 | 0.756 +/- 0.037 | 0.893 |
| CHUC Killip-skew | FedProx | 0.543 +/- 0.052 | 0.565 +/- 0.026 | 0.690 +/- 0.023 | 0.682 +/- 0.029 | 0.751 |
| CHUC ST-skew | FedProx | 0.554 +/- 0.063 | 0.565 +/- 0.036 | 0.702 +/- 0.025 | 0.686 +/- 0.027 | 0.775 |

Ablation summary:

- Hash dimension is not a sensitive hyperparameter in this range. Across 64,
  128, and 256 hash dimensions, surrogate AUROC changes only slightly in all
  scenarios.
- The current MLP `(64, 32)` is consistently stronger than the small MLP
  `(32, 16)`, especially for teacher approximation R2 on CHUC.
- These ablations support the argument that the main result is not a one-off
  artifact of a single hash dimension or under-sized/over-sized surrogate.

SUPPORT2 summary:

| Scenario | Best optimizer | R2 | Centroid(FL) AUROC | pmax(FL) AUROC | Feature-kNN AUROC | Surrogate AUROC | Teacher AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SUPPORT2-IID | FedAvg | 0.909 | 0.482 | 0.703 | 0.787 | 0.786 | 0.791 |
| SUPPORT2 label-skew | FedProx | 0.548 | 0.520 | 0.692 | 0.889 | 0.778 | 0.879 |
| SUPPORT2 disease-skew | FedProx | 0.651 | 0.521 | 0.691 | 0.759 | 0.715 | 0.764 |
| SUPPORT2 cancer-skew | FedAvg | 0.854 | 0.516 | 0.669 | 0.784 | 0.781 | 0.791 |

SUPPORT2 reinforces the deployment story: the surrogate remains close to the
teacher on IID and cancer-skew partitions and still beats simple FL confidence
baselines across all SUPPORT2 settings. However, SUPPORT2 disease-class and
cancer-status feature skew also show that transfer stability can degrade under
strong semantic client heterogeneity. This should be presented as an important
boundary condition rather than hidden.

Transfer analysis shows a consistent pattern:

- RHH-kNN transfers strongly from centralized to federated settings.
- GMM preserves ranking but can suffer scale mismatch, motivating calibration.
- Embedding reliability is a stable negative-transfer baseline with near-zero
  Pearson/Spearman correlation.

Representative overhead results, seed 42:

| Scenario | Teacher generation | Surrogate training | Surrogate inference | Communication |
| --- | ---: | ---: | ---: | ---: |
| Multi-hospital | 11.60 s | 7.36 s | 0.00028 s | 3.75 MB |
| CHUC-IID | 5.58 s | 5.32 s | 0.00037 s | 1.72 MB |
| CHUC-label | 5.52 s | 5.30 s | 0.00031 s | 1.72 MB |
| CHUC-Killip | 5.79 s | 6.20 s | 0.00035 s | 1.72 MB |
| CHUC-ST | 5.73 s | 6.24 s | 0.00033 s | 1.72 MB |

The overhead results support the deployment argument: teacher generation is
orders of magnitude slower than surrogate inference, while the surrogate model
is small enough that total FL communication remains low.

Baseline comparison shows that feature-space kNN is a strong distance-based
baseline, often close to the hash teacher. The journal argument should therefore
emphasize transfer-stable teacher selection and low-cost surrogate deployment,
not claim that hash reliability dominates every direct kNN reliability variant.

## Reproducibility Notes

- `outputs/`, `data/`, `artifacts/`, and model checkpoints are ignored by git.
- Multi-seed surrogate experiments can take several minutes on CPU.
- The current overhead table is a representative seed-42 measurement. It can be
  extended to multi-seed timing if needed.
- `Risk@5%` is reported in the current surrogate tables. For the paper, the next
  planned extension is to report Risk@1%, Risk@5%, and Risk@10%.
