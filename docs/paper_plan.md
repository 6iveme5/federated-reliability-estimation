# Journal Paper Plan

## Proposed Story

Reliability estimation in federated healthcare learning needs two properties:

1. **Transferability**: reliability behavior should remain consistent when the
   estimator moves from centralized learning to federated learning.
2. **Deployability**: reliable signals should be available at low cost during
   deployment, without repeated expensive neighbor or density computation.

## Proposed Research Questions

- RQ1: Which reliability estimators transfer from centralized to federated
  healthcare learning?
- RQ2: Why do distance/hash-based estimators transfer more reliably than
  representation-based estimators under client heterogeneity?
- RQ3: Can class-aware hash reliability serve as a teacher signal for
  federated surrogate learning?
- RQ4: Does federated surrogate reliability improve error detection and
  high-risk prediction identification?
- RQ5: How robust are reliability transfer and surrogate learning under FedAvg,
  FedProx, and adaptive server optimizers?
- RQ6: What deployability tradeoff does the surrogate offer compared with
  FL-native confidence scores and local/direct neighborhood or density
  baselines?

## Baseline Positioning

The experimental comparison should separate reliability methods by their role
in FL:

| Group | Methods | How to describe them |
| --- | --- | --- |
| FL-native confidence baselines | `pmax(FL)`, entropy(FL), margin(FL) | These require no special FL adaptation once the classifier is federated. They are the cheapest practical baselines, but only reflect prediction confidence. |
| Federated prototype baseline | `Centroid(FL)` | Clients upload per-class feature sums and counts; the server aggregates global class centroids; clients score samples by distance to the predicted global class centroid. |
| Local confidence baselines | `pmax`, entropy, margin | These are computed from local non-federated task models and should be kept separate from the FL-native confidence results. |
| Local/direct reliability baselines | centroid distance, feature-space kNN, LOF | These can be computed locally on each client. They are useful comparators, but local reference data may not capture the global FL distribution. |
| Teacher-surrogate framework | class-aware hash teacher, federated surrogate | The teacher supplies the stronger neighborhood/hash reliability signal; the surrogate is the federated model that approximates it for low-cost deployment. |

This framing avoids the claim that all baselines are newly federated. The
stronger claim is that confidence scores are easy but limited, direct
neighborhood/density methods are strong but harder to deploy globally in FL, and
the surrogate compresses a strong teacher signal into a compact federated model.

## Added Ablations

- `Centroid(FL)`: included to test whether a simple federated global-prototype
  reliability method is sufficient. Current results show it is consistently
  weaker than `pmax(FL)` and the surrogate.
- Hash dimension: `64`, `128`, and `256`. Current results show only small
  differences, so the method is not highly sensitive to this parameter.
- Surrogate size: small MLP `(32, 16)` versus current MLP `(64, 32)`. The
  current MLP is more stable, especially for teacher approximation on CHUC.

## SUPPORT2 Extension

SUPPORT2 has been added as a public critical-care mortality dataset. We use
`hospdead` as the target and 22 clinically relevant variables as predictors.
The current preprocessing yields 39 encoded features. The dataset is evaluated
with IID, label-skew, disease-class feature-skew, and cancer-status feature-skew
partitions.

Current SUPPORT2 results strengthen the public-data validation story. The
surrogate beats `pmax(FL)` in all SUPPORT2 partitions and is close to the
teacher under IID and cancer-status skew. Disease-class and cancer-status
feature-skew transfer results also reveal a boundary condition: when clients are
partitioned by semantic disease groups, RHH-kNN transfer can degrade
substantially even though the teacher remains useful for surrogate learning.

## Planned Sections

1. Introduction
2. Background and Problem Formulation
3. Reliability Transfer Analysis
4. Class-aware Hash Teacher Reliability
5. Federated Surrogate Reliability Learning
6. Experimental Setup
7. Results and Ablations
8. Discussion
9. Conclusion

## Migration Notes

- DSN code should be migrated into reusable estimator modules first.
- ISSRE code should be refactored into teacher generation, surrogate model,
  federated optimizer, and evaluation modules.
- Existing generated outputs should not be copied into the main repository until
  the final experiment layout is stable.
