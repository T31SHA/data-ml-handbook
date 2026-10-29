# From CSV to Model: The Workflow

tagline: Loading, EDA, cleaning, feature engineering, splitting, leakage and pipelines, done in the order that keeps your estimates honest.

## 1 · Loading data correctly

@@code code/s02_load.py@@
@@out code/out/s02_load.txt@@

| Problem | Symptom | Fix |
|---|---|---|
| Encoding | `UnicodeDecodeError` | `encoding="utf-8-sig"` (BOM), `"latin-1"`, `"cp1252"`; detect with `charset-normalizer` |
| Separator / quoting | one giant column, shifted fields | `sep=None, engine="python"` (sniff), `quotechar`, `escapechar`, `on_bad_lines="warn"` |
| Numbers as text | `object` dtype, `"1,234"`, `"KES 50"` | `thousands=","`, `decimal=","`, regex strip then `pd.to_numeric(errors="coerce")` |
| Sentinel missing values | `-999`, `"?"`, `"N/A"` treated as data | `na_values=[...]`, `keep_default_na=True` |
| Dates | strings, mixed formats, time zones | `parse_dates=`, `format=` (fast, strict), `utc=True`, then `tz_convert` |
| Leading zeros lost | IDs/phone numbers as ints | `dtype={"id": "string"}` |
| Memory | OOM on read | `usecols`, explicit `dtype`, `chunksize`, Polars/DuckDB, convert once to Parquet |

!!! rule "Rule of thumb"
    Read the raw CSV **once**, validate it, and persist a typed Parquet copy. All downstream work reads Parquet: it is 3–10× smaller, typed, columnar and supports predicate pushdown.

## 2 · Exploratory data analysis (EDA) protocol

1. **Grain:** what is one row (a customer? a transaction? a day per store)? Is there a natural key, and is it unique?
2. **Target:** distribution, class balance, censoring/caps (e.g. house values capped at 500k), label quality and delay.
3. **Time and groups:** is there a timestamp or an entity (user, patient, device) with repeated rows? This dictates the split.
4. **Univariate:** missing ratio, cardinality, ranges, impossible values, skew, outliers, constant columns.
5. **Bivariate:** feature–target relationships (binned means, correlations, mutual information), feature–feature collinearity.
6. **Availability at prediction time:** for every column, ask "would I know this value at the moment I predict?" If not, it leaks.
7. **Write down hypotheses** to test with features, not conclusions.

```python
df.info(memory_usage="deep"); df.describe(percentiles=[.01, .5, .99]).T
df.isna().mean().sort_values(ascending=False).head(20)
df.nunique().sort_values()                                  # constants & IDs
df[target].value_counts(normalize=True)                     # imbalance
df.groupby(pd.qcut(df.x, 10))[target].mean()                # shape of the relationship
from sklearn.feature_selection import mutual_info_regression   # non-linear dependence
```

!!! pitfall "Pitfall: EDA on the full dataset"
    Choosing features, bins or outlier thresholds after looking at the test rows quietly overfits your decisions to the test set. Split off the test set first and do EDA on the training portion.

## 3 · Cleaning

| Issue | Detect | Repair options | Watch out |
|---|---|---|---|
| Missing values | `isna().mean()`, missingness heatmap, missingness vs target | drop rows/cols; median/mode; `KNNImputer`; `IterativeImputer`; model-native (GBMs) | add `is_missing` indicator; fit imputer on train only |
| Duplicates | `duplicated(subset=keys)` | drop exact dups; resolve conflicting dups by recency | dups across train/test inflate scores |
| Outliers | IQR fences, robust z (MAD), Isolation Forest | fix if error; cap (winsorise); log transform; robust loss (Huber) | never delete rare-but-real extremes blindly (fraud!) |
| Types | `dtypes`, failed casts | `to_numeric(errors="coerce")`, `to_datetime`, `astype("category")` | count coerced NaNs |
| Inconsistent categories | `value_counts()` | `str.strip().str.lower()`, mapping dict, fuzzy matching | rare levels → "other" |
| Units / scales | domain ranges | convert to one unit; document | mixed currencies, cm vs m |
| Censoring / caps | spike at max | flag; Tobit/survival models; exclude from evaluation | treat cap as "≥ value" not exact |

**Missingness mechanisms.** MCAR (missing completely at random): dropping rows is unbiased but loses power. MAR (missing at random given observed variables): model-based imputation using those variables is valid. MNAR (missing not at random, e.g. high earners hide income): no imputation fixes it; the missingness itself is a signal, so keep an indicator and use domain knowledge.

!!! example "Worked example: IQR fences"
    Values 3, 4, 5, 5, 6, 7, 40. $Q_1 = 4$, $Q_3 = 7$, IQR $= 3$. Fences: $4 - 1.5 \cdot 3 = -0.5$ and $7 + 4.5 = 11.5$. So 40 is flagged. The median (5) is unaffected by it; the mean jumps from 5 to 10.

## 4 · Feature engineering

| Data type | Technique | Formula / call | Notes |
|---|---|---|---|
| Nominal | one-hot | `OneHotEncoder(handle_unknown="ignore", min_frequency=20)` | linear models, NNs; sparse |
| Ordinal | ordered integers | `OrdinalEncoder(categories=[[…]])` | trees handle fine |
| High cardinality | target encoding | $\tilde{x}_c = \frac{n_c \bar{y}_c + m \bar{y}}{n_c + m}$ | smoothing $m$; **cross-fitted** (`TargetEncoder`) |
| High cardinality | hashing / frequency / embeddings | `FeatureHasher`, `map(value_counts)` | NN entity embeddings |
| Numeric scale | standardise | $z = (x - \mu)/\sigma$ | linear, SVM, kNN, NN, PCA |
| Heavy tails | log / Yeo-Johnson | `np.log1p`, `PowerTransformer()` | counts, money |
| Non-linearity | splines, binning | `SplineTransformer`, `KBinsDiscretizer` | linear models gain flexibility |
| Interactions | products, ratios | `PolynomialFeatures(2, interaction_only=True)` | domain ratios beat blind products |
| Datetime | parts, cyclical, elapsed | $\sin(2\pi h/24), \cos(2\pi h/24)$ | hour 23 is close to hour 0 |
| Group aggregates | entity statistics | `groupby(user).amount.transform("mean")` | compute on train / past only |
| Time series | lags, rolling, EWM | `shift(k)`, `rolling(w).mean()` | lags ≥ forecast horizon |
| Text | TF-IDF, embeddings | `TfidfVectorizer`, sentence embeddings | |
| Geo | distances, clusters | haversine, geohash, k-means on lat/lon | |

!!! example "Worked example: smoothed target encoding"
    Category "Kisumu" has $n_c = 4$ rows with mean target $0.75$; global mean $\bar{y} = 0.20$; smoothing $m = 10$. Encoded value $= (4 \cdot 0.75 + 10 \cdot 0.20)/(4 + 10) = 5/14 \approx 0.36$. Small categories shrink toward the global mean, which is exactly what prevents overfitting to noise.

**Feature selection.** Filter (variance, correlation, mutual information, χ²) is fast but univariate; wrapper (RFE, sequential) is accurate but expensive; embedded (L1, tree importance) is a good default; permutation importance on validation data is the most honest. Always run selection *inside* cross-validation.

## 5 · Splitting and leakage

| Data structure | Split | sklearn |
|---|---|---|
| i.i.d. rows | random, stratified for classification | `train_test_split(stratify=y)`, `StratifiedKFold` |
| Repeated entities | by group | `GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit` |
| Time-ordered | past → future, with a gap | `TimeSeriesSplit(gap=…)`, manual cut-off dates |
| Spatial | by region / blocks | spatial block CV |
| Tiny data | repeated CV / nested CV | `RepeatedStratifiedKFold` |

**Leakage** is any information in training that will not be available at prediction time. The following executed demo uses *pure noise* features: honest accuracy must be 50%.

@@code code/s02_leakage.py@@
@@out code/out/s02_leakage.txt@@

!!! pitfall "Pitfall: the leakage checklist"
    - Preprocessing (scaler, imputer, encoder, feature selection, SMOTE) fitted before the split.
    - Target leakage: features created after the outcome (refund issued → fraud; discharge code → readmission).
    - Temporal leakage: random splits on time series; rolling windows that include the current or future row.
    - Group leakage: the same customer/patient/image source in train and test; near-duplicates.
    - Test-set reuse: tuning many times on the "test" set turns it into a validation set.
    - Red flag: a single feature with enormous importance, or a score far above the literature.

## 6 · Pipelines: the leakage-proof pattern

```python
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
pre = ColumnTransformer([("num", num, sel(dtype_include="number")),
                         ("cat", cat, low_card_cols),
                         ("te", TargetEncoder(), high_card_cols)])      # cross-fitted internally
model = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier())])
# cross_val_score(model, X_train, y_train, cv=cv) fits `pre` inside every fold -> no leakage
# model.set_output(transform="pandas") keeps column names for debugging
```

!!! tip "Pro tip: custom steps"
    Wrap row-wise feature functions in `FunctionTransformer` (stateless) and anything that learns statistics in a class with `fit`/`transform` (stateful). Put such functions in an importable module, not a notebook cell: a pickled pipeline stores a *reference* to the function and fails to load elsewhere otherwise.

## 7 · Model, evaluate, tune, ship

1. **Baseline:** `DummyClassifier`/`DummyRegressor`, a heuristic rule, or last-value for time series.
2. **Strong default:** regularised linear model + gradient boosting, both in pipelines, compared with the same CV splitter.
3. **Tune:** random search or Optuna on the training folds; early stopping for boosting and NNs.
4. **Error analysis:** residuals by segment, confusion pairs, worst predictions, calibration.
5. **Final evaluation:** refit on train+validation, evaluate **once** on test, report a bootstrap CI.
6. **Ship:** serialise the *whole* pipeline (`joblib`, ONNX, MLflow), record data version + code commit + metrics, expose a typed API, monitor.

Section 03 runs all of this end to end on a real dataset.

<div class="card" markdown="1">

## Cheat card: CSV → model

<div class="cols" markdown="1">

#### Order of operations
1. Load → validate → Parquet
2. Split off test (time/group aware)
3. EDA on train only
4. Pipeline: impute → encode → scale → model
5. CV on train, tune, error analysis
6. Refit, test once, ship whole pipeline

#### Cleaning defaults
- numeric NaN → median + indicator
- categorical NaN → "missing" level
- outliers → cap at 1st/99th pct or log
- rare levels (< 1%) → "other"

#### Encoding defaults
- ≤ 15 levels → one-hot
- many levels → cross-fitted target enc.
- trees: ordinal / native categorical OK
- dates → parts + cyclical + elapsed

#### Leakage smell test
- AUC > 0.98 on a hard problem
- one feature dominates importance
- CV ≫ production performance
- preprocessing outside the pipeline

#### Splits
- iid: stratified K-fold (k = 5)
- entities: GroupKFold
- time: TimeSeriesSplit + gap
- test set: touched exactly once

#### Ship checklist
- pipeline object, not loose steps
- pinned versions + input schema
- metrics + data/code version logged
- monitoring and rollback plan

</div>
</div>
