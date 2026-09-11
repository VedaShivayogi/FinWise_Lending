# FinWise Lending — Baseline Report
## Checkpoint 2 (Week 3): Data Pipeline + Baseline Model

**Status:** Complete
**Model:** Logistic Regression (class-balanced)
**Purpose:** Establish the data pipeline and the performance floor that any
later model (e.g. the XGBoost champion model) must clearly exceed.

---

## 1. Data Pipeline Summary

- **Source file:** `finwise_loan_applications.csv`
- **Rows ingested:** 5,000
- **Rows after validation:** 5,000
  (0.00% dropped)
- **Duplicate application IDs removed:** 0
- **Rows dropped for missing required fields:** 0
- **Rows dropped for out-of-range values:** 0

**Pipeline stages:** ingest -> validate schema -> clean rows -> engineer
features -> stratified train/val/test split -> fit preprocessor on train
only -> transform all splits.

**Split sizes:**

| Split | Rows | Default Rate |
|---|---|---|
| Train | 3,500 | 15.74% |
| Validation | 500 | 15.80% |
| Test | 1,000 | 15.80% |

**Features used (23):** age, income_annual_inr, employment_years, existing_loans, existing_emi_inr, credit_score, loan_amount_inr, ltv_ratio, dti_ratio, emi_to_income_ratio, loan_to_income_ratio, total_debt, debt_to_income, employment_stability, high_dti, high_ltv, low_credit, multiple_loans, risk_score, employment_type_encoded, loan_purpose_encoded, credit_score_band_encoded, age_group_encoded

---

## 2. Baseline Model

- **Algorithm:** Logistic Regression, `class_weight="balanced"`, `max_iter=1000`
- **Why this baseline:** fast, interpretable via coefficients, and the
  standard industry starting point for credit scoring before reaching for
  gradient boosting. It gives a defensible floor: any added model
  complexity later must be justified by a clear lift over these numbers.

---

## 3. Evaluation Benchmark

This is the fixed benchmark all future models are compared against.

| Split | AUC | Gini | KS Statistic | Brier Score |
|---|---|---|---|---|
| Dummy (majority class), Val | 0.5043 | 0.0086 | 0.1273 | 0.2720 |
| Baseline, Train | 0.5910 | 0.1819 | 0.1363 | 0.2437 |
| Baseline, Validation | 0.6243 | 0.2485 | 0.2118 | 0.2453 |
| Baseline, Test | 0.5478 | 0.0956 | 0.0838 | 0.2496 |

**Project target (champion model):** Gini >= 0.45 on test.
**Baseline Gini on test:** 0.0956
**Gap remaining for the champion model to close:** 0.3544

The baseline is not expected to meet the champion-model target — its role
is to prove the pipeline works end-to-end and to give a numeric floor
("beat this or the added complexity isn't worth it").

---

## 4. Top Risk Directions (Baseline Coefficients)

Top absolute-value coefficients from the baseline model, on standardized
features (positive = increases default risk):

| feature              |   coefficient |
|:---------------------|--------------:|
| dti_ratio            |     0.24015   |
| emi_to_income_ratio  |    -0.152225  |
| credit_score         |    -0.137688  |
| income_annual_inr    |    -0.130717  |
| existing_emi_inr     |     0.119785  |
| high_ltv             |     0.0864556 |
| employment_years     |    -0.0724396 |
| loan_to_income_ratio |     0.0675933 |
| loan_amount_inr      |    -0.0595551 |
| multiple_loans       |    -0.0532917 |

---

## 5. Known Limitations of the Baseline

1. Logistic Regression assumes linear/additive relationships — it will
   miss interaction effects (e.g. high DTI *combined with* low tenure)
   that a tree-based model can capture.
2. No hyperparameter tuning was performed — this is intentionally a
   floor, not an optimized model.
3. Bias/fairness audit is deferred to the champion-model checkpoint.
4. Calibration was not explicitly tuned; `class_weight="balanced"` trades
   calibration for better separation on the minority (default) class.

## 6. Next Steps (Checkpoint 3)

- Train the champion model (XGBoost) using the **same** `data_pipeline.py`
  and `preprocessor.pkl` produced here, so the comparison is apples-to-apples.
- Compare champion vs. baseline on this exact benchmark table.
- Run the full SHAP explainability and bias-audit workstreams on the
  champion model.
