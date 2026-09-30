# Sample portfolio project README

An illustrative template for a data science portfolio project. It uses a CFPB-complaints example. Every bracketed value and number is a placeholder: replace it with your own real work, and delete anything you cannot support.

## Suggested repo layout

```
cfpb-relief-model/
├── README.md
├── environment.yml          # or requirements.txt, pinned versions
├── Makefile                 # make data, make train, make report
├── data/
│   └── README.md            # where the data comes from, how to download it
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_shap_analysis.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_features.py     # e.g., no leakage columns, date splits are ordered
├── reports/
│   └── figures/             # calibration curve, PR curve, SHAP plots
└── docs/
    └── decisions.md         # the "why" behind key choices
```

## README template

---

# Predicting CFPB complaints that end in monetary relief

## Problem

Which consumer complaints are likely to result in monetary relief?
Why it matters: prioritizing review queues. Positive class: ~[X]%.

## Data

CFPB Consumer Complaint Database. [N] rows, [date range].
Download steps in `data/README.md`. Known issues: missing narratives,
consent flags, changing product categories over time.

## Approach and why

- Baseline: logistic regression on structured fields (PR-AUC [A]).
- Final: CatBoost with native categorical handling (PR-AUC [B]).
- Time-based split (train on earlier years, test on later) to mimic deployment.

## What failed

- Oversampling with SMOTE before the split leaked near-duplicates. Fixed by
  resampling inside training folds only, then dropped it for class weights.
- Free-text features added little over structured fields.

## Results vs. baseline

| Model | PR-AUC | Recall @ 20% precision |
|---|---|---|
| Logistic regression | [A] | [A2] |
| CatBoost | [B] | [B2] |

## Explainability

TreeSHAP shows that [top feature] and [second feature] drive predictions.
One feature looked too strong; I traced it to [leakage source] and removed it.
See `notebooks/03_shap_analysis.ipynb`.

## Calibration

Reliability curve before and after isotonic calibration. Brier [C] to [D].

## AI usage note

I used an AI assistant for boilerplate and plotting code. I reviewed and
tested all of it, and wrote the feature logic and evaluation myself.

## Limitations and next steps

Complaint outcomes depend on company behavior that may change over time.
Next: monitor drift by year, try a text embedding feature.

## How to run

```
make data && make train && make report
```

---

## What makes a repo like this strong

- **Reasoning is visible:** problem, why each choice, and a "What failed" section.
- **Rigor:** baseline, time-based split, leakage check, PR-AUC instead of accuracy, calibration.
- **Explainability:** SHAP used to sanity-check the model, not just decorate it.
- **Honest AI disclosure:** says what AI helped with and how you verified it.
- **Reproducible:** pinned environment, one command to rerun, a tests folder.
- **Real commit history:** visible iteration instead of one giant commit.
- **Defensible:** you can explain every file.
