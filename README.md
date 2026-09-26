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

The model achieved perfect discrimination on the held-out test set (ROC-AUC = 1.0, PR-AUC = 1.0, no misclassifications). This level of separability likely reflects a small dataset with clearly distinct churn behavior rather than a guarantee of real-world generalization.

> **Note on overfitting risk:** Near-perfect scores on this kind of dataset warrant caution. Before deploying this model in production, validate on a larger, more recent sample and monitor performance over time (e.g., via a rolling backtest or A/B holdout).

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
