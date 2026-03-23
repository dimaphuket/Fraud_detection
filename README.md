# Credit Card Fraud Detection

End-to-end machine learning project for detecting fraudulent credit card transactions using the IBM synthetic dataset (~24 million transactions, 1991–2019).

**Task:** binary classification — predict `Is Fraud?`  
**Perspective:** bank-issuer (full access to transaction, card, and customer data)  
**Class imbalance:** ~0.12% fraud rate

---

## Results

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost (tuned) — Champion** | **0.883** | **0.9997** | **0.822** | **0.798** | **0.809** |
| XGBoost (basic) | 0.860 | 0.9995 | 0.767 | 0.795 | 0.781 |
| CatBoost (tuned) | 0.838 | 0.9989 | 0.779 | 0.760 | 0.770 |
| CatBoost (no tuning) | 0.828 | 0.9994 | 0.719 | 0.778 | 0.747 |
| Random Forest (manual opt.) | 0.786 | 0.9993 | 0.758 | 0.715 | 0.736 |
| Stacking (LogReg + RF) | 0.720 | 0.9992 | 0.171 | 0.973 | 0.291 |
| Logistic Regression (tuned) | 0.581 | 0.9958 | 0.578 | 0.525 | 0.551 |
| Rule-based risk score | 0.211 | 0.935 | 0.459 | 0.225 | 0.302 |
| Always predict "no fraud" | 0.001 | 0.500 | — | — | 0.000 |

> **Metric choice:** PR-AUC (area under Precision-Recall curve) is the primary metric because ROC-AUC is inflated at extreme class imbalance and does not reflect real detection quality. F1 scores computed at the best threshold found on the validation set.

---

## Project structure

```
.
├── 1. Preliminary.ipynb          # Data loading, merging, cleaning, memory optimisation
├── 2. EDA.ipynb                  # Exploratory data analysis, significance testing
├── 3. Feature_Engineering.ipynb  # Feature creation, drift analysis, dataset preparation
├── 4. Baseline_and_linear.ipynb  # Non-ML baselines + Logistic Regression
├── 5. Tree_models.ipynb          # Decision Tree, Random Forest
├── 6. Stacking.ipynb             # Model stacking experiment
├── 7. Boosting.ipynb             # XGBoost, CatBoost (champion model)
├── 8. Bonus experiment.ipynb     # Clustering as a feature engineering approach
├── Fragments.ipynb               # Prototype / utility code fragments
├── data/                         # Source CSV files (not included in repo)
│   ├── credit_card_transactions-ibm_v2.csv
│   ├── sd254_cards.csv
│   └── sd254_users.csv
├── catboost_info/                # CatBoost training logs
├── results.json                  # Baseline and linear model metrics
├── results2.json                 # All model metrics (tree, boosting, stacking)
├── logreg_top_features.json      # Top-20 logistic regression feature importances
├── fraud_after_EDA_schema.json   # Column schema after EDA
├── data_after_FE_schema.json     # Column schema after Feature Engineering
├── data_linear_schema.json       # Schema for linear model dataset
├── data_tree_schema.json         # Schema for tree model dataset
└── requirements.txt
```

> Intermediate `.parquet` files and `.pkl` model files are not tracked by git (see `.gitignore`).

---

## Pipeline walkthrough

### 1. Preliminary (`1. Preliminary.ipynb`)
- Merges three source tables on `User` / `CARD INDEX` keys (~24M rows).
- Applies dataset errata fix: replaces erroneous `Merchant_State = "Italy"` values by sampling the real state distribution.
- Renames columns to snake_case, downcasts dtypes — **memory reduced from 8.2 GB to 3.5 GB (~57%)**.
- Merges `Year`, `Month`, `Day`, `Time` into a single `Timestamp` column.
- Saves `fraud.parquet`.

### 2. EDA (`2. EDA.ipynb`)
- Fills missing values: `Merchant_State` for online transactions → `"ONLINE"`, `Has_Error` binary flag, `Is_Apartment` binary flag.
- Creates `Is_Online`, converts `Use_Chip` to binary chip/no-chip.
- Identifies and removes future-leak risks: `PIN_Last_Changed_Year`, highly correlated columns.
- Performs statistical significance tests (Mann-Whitney U for numeric, χ² for categorical).
- **Key finding:** fraud is a *merchant-level* phenomenon; cardholder demographics and residential address are not statistically significant predictors.
- Drops 8 uninformative/leaking columns; saves `fraud_after_EDA.parquet`.

### 3. Feature Engineering (`3. Feature_Engineering.ipynb`)
Features are grouped into two kinds:

**Row-level (no leak risk):**
- `Amount_to_Income` = Amount / (Annual Income / 12)
- `user_age` at the time of each transaction
- Error dummies: `Error_Bad_CVV`, `Error_Bad_PIN`, `Error_Bad_Expiration`, `Error_Insufficient_Balance`, etc.
- Time features: `is_night`, `is_business_hours`, `is_weekend`, cyclical encodings (`hour_sin/cos`, `dow_sin/cos`)
- `amount_log`, `amount_round_10` (multiples of $10 — common fraud pattern)

**Time-sorted window / lag features (computed strictly on past data):**
- Velocity: `txn_count_5m_card`, `txn_count_1h_card`, `txn_count_5m_user`, `txn_count_1h_user`
- `time_since_prev_txn_card_min`, `time_since_prev_txn_user_min`
- Merchant velocity: `merchant_txn_count_1h/24h`, `merchant_amount_sum_1h/24h`
- Historical fraud rates: `merchant_fraud_rate`, `state_fraud_rate`
- `first_user_payment_to_this_merchant`, `state_changed_1d`, `card_burst_5m`, `errors_prev_1h`

**Drift analysis (PSI / TVD across years):**  
Data before 2015 exhibits significant feature drift → training restricted to **2015–2019** (2020 excluded due to anomalously low fraud rate). Two final datasets saved:
- `data_linear.parquet` — for logistic regression (reduced feature set, StandardScaler-ready)
- `data_tree.parquet` — for tree models (OrdinalEncoder-ready, full feature set)

### 4. Baselines & Logistic Regression (`4. Baseline_and_linear.ipynb`)
Five non-ML baselines establish the minimum acceptable bar:

| Baseline | F1 | Notes |
|---|---|---|
| Always predict 0 | 0.000 | Trivial |
| Random | 0.002 | At fraud base rate |
| MCC + Amount heuristic | 0.006 | Amount > 3× MCC average |
| Visual heuristic | 0.036 | ≥4 of top-10 risky features |
| **Rule-based risk score** | **0.302** | Circa 1990s banking rules |

The rule-based system mirrors real late-1990s anti-fraud systems and serves as the business baseline.

Logistic Regression models are trained with `TimeSeriesSplit` cross-validation; best result: **PR-AUC = 0.581, F1 = 0.551**. Polynomial features and PCA do not help. Top-20 features saved to `logreg_top_features.json`.

### 5. Tree Models (`5. Tree_models.ipynb`)
- **Decision Tree (GridSearchCV):** PR-AUC = 0.498, F1 = 0.536
- **Random Forest (manual optimisation):** `n_estimators=500`, `max_depth=25`, `min_samples_leaf=5`, `max_features=0.5` → **PR-AUC = 0.786, F1 = 0.736**
- Optuna (Bayesian search) formally wins on CV but overfits; manual depth restriction generalises better.
- RF on logistic regression's top-20 features: PR-AUC = 0.639 — confirms LogReg has reached its ceiling.

### 6. Stacking (`6. Stacking.ipynb`)
- OOF stacking: LogReg + RF as base learners, LogReg as meta-model.
- **Result: failed** — the quality gap between base models is too large. The meta-model almost entirely ignores LogReg's contribution, producing high recall (0.973) at the cost of very low precision (0.171), which is operationally unacceptable.

### 7. Boosting (`7. Boosting.ipynb`)
- **XGBoost** (GPU `tree_method='hist'`, `scale_pos_weight`, `eval_metric='aucpr'`, early stopping): **PR-AUC = 0.883, F1 = 0.809**
- **CatBoost** (GPU, `use_best_model=True`, early stopping ~190 iterations): PR-AUC = 0.838, F1 = 0.747
- Hyperparameters tuned with `RandomizedSearchCV(TimeSeriesSplit)`.
- Best model saved as `best_xgb.pkl`.

**Business interpretation:** ~80% of all fraud correctly blocked; ~82% of sent alerts are true positives — suitable for automated card blocking with manageable false-positive complaint volume.

### 8. Bonus Experiment (`8. Bonus experiment.ipynb`)
Tests the hypothesis: *fraud = atypical transaction pattern → fraud transactions lie far from cluster centroids*.

Six independent clustering spaces (MiniBatchKMeans) are trained on 300K samples:

| Space | Features | Clusters |
|---|---|---|
| `transaction_profile` | Amount, MCC, chip, time, geography | 35 |
| `customer_profile` | Debt, FICO, credit limit, income ratio, age | 20 |
| `behavior_profile` | Velocity, time-since-prev, errors | 30 |
| `merchant_profile` | Merchant velocity, amounts | 30 |
| `error_profile` | Error flags and counts | 10 |
| `risk_context_profile` | Historical fraud rates by merchant and state | 20 |

Cluster IDs + distances to all centroids (~145 features total) are fed into LogReg and XGBoost.

**Conclusion:** clustering successfully identifies behavioral archetypes and centroid distances carry predictive fraud signal. However, performance (PR-AUC ~0.65–0.70) does not approach the full-feature model. Cluster features could be useful as additional inputs in an extended model ensemble.

---

## Key findings

1. **Primary drivers of fraud:** `Is_Online`, `Use_Chip`, `Has_Error`, `MCC`, merchant geography (state/city), transaction time, velocity features, historical fraud rates.
2. **Not predictive:** cardholder demographics (gender, residential address, city/zip), card brand/type.
3. **Data drift:** only 2015–2019 is stable enough for training; earlier years degrade model performance.
4. **Feature engineering is the single largest lever** — velocity, historical fraud rates, and burst detection account for the majority of the gap between LogReg (PR-AUC 0.58) and XGBoost (PR-AUC 0.88).
5. **Stacking fails** when base model quality gap is too large.
6. **Clustering as FE** preserves useful signal but cannot replace handcrafted features.

---

## Reproducing the results

> The source CSV files are not included in the repository due to size. Download from the [IBM Credit Card Transactions dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place in `data/`.

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run notebooks in order:**
```
1. Preliminary.ipynb
2. EDA.ipynb
3. Feature_Engineering.ipynb
4. Baseline_and_linear.ipynb
5. Tree_models.ipynb
6. Stacking.ipynb
7. Boosting.ipynb
```
Notebooks 8 and `Fragments.ipynb` are optional experiments.

**Hardware:** XGBoost and CatBoost notebooks use CUDA GPU (`device="cuda"`). To run on CPU, change `device="cuda"` → `device="cpu"` and `task_type="GPU"` → `task_type="CPU"` in the respective cells.

---

## Requirements

See [requirements.txt](requirements.txt).

| Package | Version |
|---|---|
| pandas | 2.3.0 |
| numpy | 2.3.4 |
| scikit-learn | 1.7.2 |
| xgboost | 3.1.2 |
| catboost | 1.2.8 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scipy | 1.15.3 |
| pyarrow | 23.0.1 |
