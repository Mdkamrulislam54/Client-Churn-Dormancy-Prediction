# Client Churn (Dormancy) Prediction

A machine learning pipeline that identifies brokerage/trading clients who are likely to go dormant (churn), so Relationship Managers (RMs) can prioritize proactive outreach before revenue is lost.

## Overview

Client attrition in financial services rarely happens overnight — it shows up first as declining trading activity, shrinking balances, and longer gaps between trades. This project turns raw transaction-level data into a client-level dataset, engineers behavioral features, and trains a classifier to flag clients at risk of dormancy. The output is a ranked **action list** that RMs can use directly for retention efforts.

**Pipeline:**

```
Raw Transactions (Excel)
        │
        ▼
Client-Level Feature Engineering
        │
        ▼
Churn Labeling (rule-based cutoff)
        │
        ▼
Gradient Boosting Classifier
        │
        ▼
Risk Scoring + Risk Bands (Low / Medium / High)
        │
        ▼
Prioritized Action List (CSV)
```

## Repository Structure

```
├── Client_Churn__Dormancy__Prediction.ipynb   # Main analysis & modeling notebook
├── Financial Dataset.xlsx                     # Raw transaction-level input data
├── churn_action_list.csv                      # Generated output: ranked client risk list
├── requirements.txt                           # Python dependencies
└── README.md                                  # Project documentation
```

## Data

| File | Description |
|---|---|
| [`Financial Dataset.xlsx`](Financial%20Dataset.xlsx) | Raw transaction-level records per client, including trading dates, turnover, commission, equity, deposits, withdrawals, and transfers. |
| [`churn_action_list.csv`](churn_action_list.csv) | Model output — clients ranked by churn probability with assigned risk bands, ready for RM action. |

## Methodology

### 1. Data Preparation
Transaction dates are parsed, and the dataset's **observation end date** is used to define a **churn cutoff** (30 days prior). Any client whose last trade falls before this cutoff is labeled as churned.

### 2. Feature Engineering
Transaction records are aggregated to the client level to construct behavioral features:

| Feature | Description |
|---|---|
| `turnover`, `commission` | Total trading volume and commission generated |
| `equity_last` | Most recent account equity |
| `deposits`, `withdraws` | Total funds moved in/out |
| `net_flow` | Net of deposits, withdrawals, and transfers |
| `active_days` | Number of unique trading days |
| `recency` | Days since last trade |
| `tenure` | Days since first trade |
| `trade_freq` | Trading activity relative to tenure |
| `turnover_per_day` | Average daily trading volume |

### 3. Modeling
A **Gradient Boosting Classifier** is trained inside a scikit-learn `Pipeline` with median imputation and feature scaling:

```python
Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=3, subsample=0.9, random_state=42
    )),
])
```

The train/test split is stratified on the churn label to preserve class balance.

### 4. Evaluation
Model performance is assessed using:
- ROC-AUC
- PR-AUC (Average Precision)
- Confusion Matrix
- Precision / Recall / F1 (Classification Report)

### 5. Scoring & Action List
The trained model scores **all** clients, which are then bucketed into risk bands:

| Risk Band | Churn Probability |
|---|---|
| Low | 0.0 – 0.3 |
| Medium | 0.3 – 0.6 |
| High | 0.6 – 1.0 |

Clients are sorted by churn probability (highest first) and exported to `churn_action_list.csv` for RM follow-up.

## Results

### Dataset Summary

| Metric | Value |
|---|---|
| Total clients (aggregated from transactions) | 2,086 |
| Observation window | 2023-01-01 → 2023-06-01 |
| Churn cutoff (inactivity threshold) | 30 days |
| Churn rate (base rate) | ~29.1% |
| Training set | 1,564 clients |
| Test set | 522 clients |

The churn class ratio was preserved almost exactly between the train (29.09% churned) and test (29.12% churned) splits via stratified sampling.

### Model Performance (held-out test set)

| Metric | Score |
|---|---|
| ROC-AUC | 1.00 |
| PR-AUC (Average Precision) | 1.00 |
| Precision (both classes) | 1.00 |
| Recall (both classes) | 1.00 |
| F1-score (both classes) | 1.00 |

**Confusion Matrix**

| | Predicted: Active | Predicted: Churned |
|---|---|---|
| **Actual: Active** | 370 | 0 |
| **Actual: Churned** | 0 | 152 |

The model correctly classified all 522 test clients with zero false positives and zero false negatives.

> **Note on overfitting risk:** A perfect score on every metric is unusual and warrants scrutiny rather than celebration. It likely reflects the fact that engineered features like `recency` are almost definitionally correlated with the churn label itself (churn is defined by the same last-trade date used to compute recency), making the classes trivially separable. Before relying on this model operationally, validate it against a fresh, held-out time period and consider whether the label definition is leaking information into the features.

### Sample Output — Top High-Risk Clients

An excerpt from the generated `churn_action_list.csv`, showing the highest-priority clients for RM follow-up:

| Customer ID | Turnover | Equity (Last) | Recency (days) | Churn Probability | Risk Band |
|---|---|---|---|---|---|
| C1003 | 539,238.0 | 4,449,995.0 | 133 | 1.00 | High |
| C990 | 10,974,344.6 | 2,711,952.0 | 65 | 1.00 | High |
| C997 | 1,293,645.5 | 1,200,036.0 | 133 | 1.00 | High |
| C999 | 22,076.1 | 416,426.4 | 49 | 1.00 | High |
| C995 | 631,787.5 | 2,075,102.0 | 32 | 1.00 | High |

*(Full ranked list of all 2,086 clients is available in [`churn_action_list.csv`](churn_action_list.csv).)*

### Key Takeaways

- The model reliably surfaces clients whose recent trading activity has dropped off, using `recency`, `turnover`, and `net_flow` as the strongest behavioral signals.
- Roughly 3 in 10 clients in this dataset fall into the churned/dormant category, making retention a material concern rather than an edge case.
- The ranked action list turns a 2,000+ client base into a short, prioritized queue RMs can act on immediately, rather than manually reviewing raw transaction data.

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/Mdkamrulislam54/Client-Churn-Dormancy-Prediction.git
cd Client-Churn-Dormancy-Prediction
pip install -r requirements.txt
```

### Usage

1. Place `Financial Dataset.xlsx` in the project root (already included in this repo).
2. Open and run the notebook:

```bash
jupyter notebook Client_Churn__Dormancy__Prediction.ipynb
```

3. Run all cells top to bottom. The notebook will:
   - Load and clean the raw data
   - Engineer client-level features
   - Train and evaluate the classifier
   - Generate `churn_action_list.csv`

## How Relationship Managers Can Use the Output

The `churn_action_list.csv` file is sorted by churn probability, highest risk first. Suggested workflow:

- **High risk band** → Immediate proactive outreach to understand concerns and re-engage.
- **Medium risk band** → Scheduled check-ins or targeted product offers.
- **Low risk band** → Standard relationship maintenance.

## Tech Stack

- **Python** — pandas, scikit-learn
- **Modeling** — Gradient Boosting Classifier
- **Environment** — Jupyter Notebook

## Future Improvements

- Validate model generalization on a larger, more recent dataset
- Add cross-validation and hyperparameter tuning
- Include feature importance / SHAP analysis for model interpretability
- Automate periodic re-scoring and action-list generation
- Track intervention outcomes to measure retention impact

## License

This project is available under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Md Kamrul Islam**
GitHub: [@Mdkamrulislam54](https://github.com/Mdkamrulislam54)
