# FinWise Lending — Credit Risk Project
## Checkpoint 2 (Week 3): Data Pipeline + Baseline Model

This folder contains everything needed to run the data ingestion pipeline,
train the baseline model, and reproduce the baseline report.

---

## 1. Files in this delivery

| File | Purpose |
|---|---|
| `data_pipeline.py` | Reusable ingestion + preprocessing pipeline. Import this in any notebook/script — do not duplicate its logic. |
| `Checkpoint2_Baseline_Model.ipynb` | Runs the pipeline, trains the Logistic Regression baseline, computes the evaluation benchmark, and auto-writes the report. |
| `baseline_report.md` | The generated write-up (data summary, benchmark table, limitations, next steps). Regenerated every time the notebook runs. |
| `README.md` | This file. |

Running the notebook also creates an `artifacts/` folder (not included here,
since it's generated fresh from your data) containing:

- `preprocessor.pkl` — the fitted encoder/scaler, reused unchanged by later models
- `baseline_model.pkl` — the trained Logistic Regression model
- `evaluation_benchmark.json` — machine-readable version of the benchmark table
- `ingestion_report.json` — row counts, drops, and reasons
- `baseline_coefficients.csv` — every feature's coefficient, sorted by impact
- `baseline_report.md` — same as above, freshly written

---

## 2. Requirements

- Python 3.9+
- Packages:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter jupytext tabulate
  ```

---

## 3. Setup

1. Create a project folder and place these three files in it.
2. Add your real data file, named exactly:
   ```
   finwise_loan_applications.csv
   ```
   in the **same folder** as `data_pipeline.py`.

3. Your CSV must contain these columns (case-sensitive):

   ```
   application_id, age, income_annual_inr, employment_type, employment_years,
   existing_loans, existing_emi_inr, credit_score, loan_amount_inr,
   loan_purpose, ltv_ratio, dti_ratio, default_flag
   ```

   If your real schema uses different column names or value ranges, edit
   `REQUIRED_COLUMNS` and `VALID_RANGES` near the top of `data_pipeline.py`
   to match before running.

---

## 4. How to run

### Option A — Jupyter (recommended, matches the deliverable)
```bash
cd your-project-folder
jupyter notebook
```
Open `Checkpoint2_Baseline_Model.ipynb`, then **Kernel > Restart & Run All**.

### Option B — Run just the pipeline from the command line
```bash
python data_pipeline.py
```
This ingests the data, engineers features, splits it, fits the
preprocessor, and prints the resulting shapes. Useful for a quick sanity
check before opening the notebook.

---

## 5. What "done" looks like

After a successful run you should have:

- No errors in any notebook cell
- An `artifacts/` folder with all 6 files listed in section 1
- `baseline_report.md` showing your **real** default rate, split sizes,
  and benchmark numbers (AUC, Gini, KS statistic, Brier score)
- Console output ending with `CHECKPOINT 2 COMPLETE`

---

## 6. Notes for Checkpoint 3 (champion model)

- Reuse `data_pipeline.py` and the exact `artifacts/preprocessor.pkl`
  produced here — do not refit a new preprocessor on different data, or
  the baseline vs. champion comparison will not be apples-to-apples.
- Compare the champion model against `artifacts/evaluation_benchmark.json`
  using the same metrics (AUC, Gini, KS, Brier) and the same test split.
- The project's target for the champion model is **Gini >= 0.45 on test**.
  The baseline is not expected to meet this — its job is to set the floor.

---

## 7. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: Could not find data file` | `finwise_loan_applications.csv` isn't in the same folder as `data_pipeline.py`, or the filename doesn't match exactly. |
| `ValueError: Missing required columns` | Your CSV's column names don't match `REQUIRED_COLUMNS` in `data_pipeline.py`. Rename the columns or edit that list. |
| Notebook cell `from data_pipeline import run_pipeline` fails | The notebook and `data_pipeline.py` must be in the same folder, and you must launch Jupyter from that folder. |
| Gini/AUC look very low | Check `artifacts/ingestion_report.json` — if most rows were dropped by validation, your data may not match the expected schema or ranges. |
