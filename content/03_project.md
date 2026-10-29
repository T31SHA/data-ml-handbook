# Worked Project: California Housing, End to End

tagline: A complete, executed project from a messy CSV to a validated, tuned, served model. Every number on these pages was produced by running the code.

## The brief

**Goal:** predict the median house value of a California census block group (20,640 groups, 1990 census) from income, age, rooms, population, occupancy and location, so that an analyst can flag under- or over-priced areas. **Metric:** RMSE in units of 100k USD (penalises large misses), with MAE reported for interpretability. **Baseline:** predict the training mean.

To make the cleaning realistic, the raw export below has defects injected on purpose: 3% missing income, 1% missing house age, messy region labels (`"norcal "`) and 150 duplicate rows. The `region` column is derived from latitude/longitude.

## Step 1 · Build the raw CSV

@@code code/proj01_make_csv.py@@
@@out code/out/proj01_make_csv.txt@@

## Step 2 · EDA

@@code code/proj02_eda.py@@
@@out code/out/proj02_eda.txt@@

![](figures/proj_eda.png)

**What EDA told us.**

- **Duplicates** (150) must go before splitting, otherwise copies land in both train and test.
- **Region labels** need normalising (trailing spaces, case).
- **Income** is the dominant linear driver (r = 0.69), but location matters non-linearly: the coast is expensive, so a tree model should beat a linear one.
- **The target is capped at 5.0** (4.8% of rows): these are *censored* values. Errors there are partly irreducible; a residual plot will show it.
- **`AveOccup` and `AveRooms` have extreme outliers** (max 1,243 occupants per household). Tree models are robust to this; linear models need capping or logs.

## Step 3 · Pipeline and model comparison (5-fold CV on training data)

@@code code/proj03_model.py@@
@@out code/out/proj03_model.txt@@

!!! intuition "Reading the table"
    Ridge cuts the baseline error by 40% but plateaus at R² ≈ 0.64 because price depends on location non-linearly. Tree ensembles capture interactions (income × coast) and reach R² ≈ 0.85. LightGBM wins narrowly, and its fold-to-fold standard deviation is small enough that the ranking is stable.

## Step 4 · Tune, evaluate once on test, explain

@@code code/proj04_tune_eval.py@@
@@out code/out/proj04_tune_eval.txt@@

![](figures/proj_eval.png)

![](figures/proj_optuna.svg)

**Interpretation.**

- Test RMSE is close to the CV estimate, so the CV protocol was honest (no leakage, no overfitting to folds).
- Residuals fan out and hit a hard wall at the 5.0 cap: the model cannot predict above the censoring point. Fixes: treat capped rows as censored (survival-style loss), or drop them from training *and* clearly scope the model to values below 500k USD.
- Latitude and longitude dominate permutation importance. Location is the price; feature ideas that follow from this are distance to the coast, distance to San Francisco or Los Angeles, and geo-clusters.

## Step 5 · Serve the model behind a typed API

@@code code/proj05_serve.py@@
@@out code/out/proj05_serve.txt@@

!!! pitfall "Pitfall met in this project: pickling custom functions"
    The first version defined `add_feats` inside the training script; the saved pipeline then failed to load in the API process (`AttributeError: can't get attribute 'add_feats'`). Pickle stores a reference to the function's module, so custom transformers must live in an importable module (`housing_feats.py`) that ships with the model.

## Project retrospective

| Phase | Decision | Why |
|---|---|---|
| Ingest | dedupe before split; normalise labels | prevent leakage and phantom categories |
| Split | random 80/20, KFold(5) on train | rows are independent block groups (spatial CV would be stricter) |
| Features | ratios (rooms per person, bedroom ratio) | domain ratios generalise better than raw counts |
| Model | LightGBM in a pipeline | best CV, handles skew/outliers/interactions |
| Tuning | 25 Optuna TPE trials, 3-fold | good gains for low compute |
| Evaluation | single test pass, residual analysis | honest estimate; discovered censoring |
| Serving | FastAPI + pydantic validation | reject bad inputs (422) before they reach the model |
| Next | spatial CV, geo features, censored loss, monitoring of income drift | known weaknesses |

!!! interview "Interview question: your CV score is great but production is worse. What happened?"
    Check, in order: (1) leakage (features unavailable at prediction time, preprocessing outside the pipeline, duplicates across splits); (2) a split that doesn't match deployment (random split on time or grouped data); (3) training–serving skew (features computed differently online); (4) distribution shift since training (data drift, concept drift); (5) selection effects (the model changes the data it later sees). Fix the protocol, then monitor.
