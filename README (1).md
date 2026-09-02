# 📉 Telco Customer Churn Prediction

Predicting which telecom customers are likely to churn (cancel their service) using
classical machine learning, so a retention team can prioritize outreach where it
matters most.

> **Business framing:** *"You're the DS team at a regional telecom. The CFO wants to
> know: which customers will churn next quarter, why, and what's the ROI of a
> targeted retention campaign vs. blanket discounts?"*

---

## 📌 Project Overview

| | |
|---|---|
| **Problem type** | Binary classification (`Churn`: Yes / No) |
| **Domain** | Telecom customer retention |
| **Dataset** | Merged Telco Customer Churn dataset (customer, demographic, location, services & status data) |
| **Rows / Columns (raw merge)** | 7,043 customers × 75 columns (before cleaning) |
| **Class balance** | ~26.5% churn — imbalanced |
| **Models compared** | Dummy, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Naive Bayes |

The dataset was assembled from **7 source spreadsheets** (customer churn, Telco churn,
demographics, location, population, services, and status) merged on `Customer ID` /
`Zip Code`, then cleaned, deduplicated, and encoded for modeling.

---



## 🔄 Pipeline (what the notebook does)

1. **Load & merge** — 7 raw Excel exports (customer churn, demographics, location,
   population, services, status) are merged on `Customer ID`, and `Population` is
   joined in on `Zip Code`.
2. **Drop leakage / identifier columns** — IDs, lat/long, churn score/label/reason
   (post-hoc churn fields that would leak the target), and other non-predictive
   columns are dropped.
3. **Handle missing values** — `Offer` and `Internet Type` are imputed with the mode;
   `Total Charges` blank strings are coerced to numeric and imputed with the median.
4. **Deduplicate merged columns** — `_x` / `_y` suffix pairs created by the merges are
   compared and identical duplicates are dropped.
5. **Exploratory Data Analysis** — churn distribution, tenure & monthly charge
   distributions, churn by contract type, correlation heatmap, and pairplots by churn
   status.
6. **Encode categoricals** — binary columns via `LabelEncoder`, multi-category columns
   via one-hot encoding (`pd.get_dummies`).
7. **Model comparison** — 8 classifiers are benchmarked on the same stratified 80/20
   split (see [`docs/Churn_ML_Models_Overview.pdf`](docs/Churn_ML_Models_Overview.pdf)
   for the full write-up):

   | # | Model | Type | Needs Scaling? |
   |---|-------|------|-----------------|
   | 1 | Dummy Classifier | Baseline | No |
   | 2 | Logistic Regression | Linear | Yes |
   | 3 | Decision Tree | Tree-based | No |
   | 4 | Random Forest | Ensemble (bagging) | No |
   | 5 | Gradient Boosting | Ensemble (boosting) | No |
   | 6 | K-Nearest Neighbors | Distance-based | Yes |
   | 7 | Support Vector Machine | Kernel-based | Yes |
   | 8 | Naive Bayes | Probabilistic | No |

   Because of the ~26.5% churn class imbalance, models are compared on
   **accuracy, precision, recall, F1, and ROC-AUC** — not accuracy alone, since a
   model that always predicts "no churn" would still score ~73% accuracy while
   catching zero churners.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/telco-customer-churn.git
cd telco-customer-churn
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/advance_data.ipynb
```

The notebook currently reads the raw source Excel files from a local path — update the
file paths at the top of the notebook to point at `data/` (or your own copy of the raw
source files) before running end-to-end.

---

## 📊 Key Findings (from EDA)

- **Month-to-month contracts** show the highest churn rate compared to one- and
  two-year contracts.
- **Tenure** is strongly associated with churn — newer customers churn far more than
  long-tenured ones.
- **Electronic check** as a payment method correlates with higher churn.
- The dataset is **imbalanced (~26.5% churn)**, so recall/F1/ROC-AUC on the churn
  class are prioritized over raw accuracy when comparing models.

---


---

## 🛠️ Tech Stack

- **Python 3.13**
- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling & preprocessing

---

