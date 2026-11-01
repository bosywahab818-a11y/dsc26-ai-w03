
# Tawseel – Baseline Results

## Experimental setup

- Dataset: 4,000 synthetic COD parcels per seed.
- Training: chronological first 80%.
- Testing: latest 20%.
- Model: Logistic Regression with preprocessing inside the pipeline.
- Baseline: DummyClassifier(strategy="prior").
- Evaluation metrics: ROC-AUC, PR-AUC, and Brier score.
- Customer history features use only strictly earlier parcels.

## Results

### Seed 0 – Detailed comparison

| Metric | Logistic Regression | Dummy |
|---|---:|---:|
| ROC-AUC | 0.6456 | 0.5000 |
| PR-AUC | 0.3806 | 0.2200 |
| Brier score | 0.1633 | 0.1717 |
| Base rate | 0.2200 | 0.2200 |
| Test samples | 800 | 800 |

PR-AUC lift: +0.1606

### Additional seed experiments

| Seed | Model PR-AUC | Dummy PR-AUC | PR-AUC Lift | Model Brier | Dummy Brier |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3132 | 0.2037 | 0.1095 | 0.1589 | 0.1624 |
| 7 | 0.3750 | 0.2050 | 0.1700 | 0.1561 | 0.1633 |
| 11 | 0.3308 | 0.2250 | 0.1058 | 0.1701 | 0.1744 |
| 42 | 0.3205 | 0.2137 | 0.1067 | 0.1646 | 0.1681 |

All five tested seeds achieved a PR-AUC lift greater than 0.05 and a Brier score lower than the dummy baseline.

ROC-AUC ranged from 0.6303 to 0.6597 across the additional seed experiments.

## Notes to Sherif

The Logistic Regression baseline consistently outperformed the prior-probability dummy model across five tested random seeds, with PR-AUC improvements ranging from 0.1058 to 0.1700.

We used a chronological train/test split, point-in-time customer history, and preprocessing fitted only on the training data to reduce the risk of data leakage.

This is a modest first baseline rather than a production-ready system, and its performance should be monitored and validated on real delivery data before operational use.
