# Telco Customer Churn — Hypothesis Testing & ML Project

A college project notebook covering statistical hypothesis testing and machine learning
model training on Telco customer churn data.

## Files in this folder

| File | What it is |
|---|---|
| `Telco Customer Churn Prediction Model.ipynb` | The main notebook — EDA, 10 hypothesis tests, and 7 trained ML models |
| `Telco_Customer_Churn_Merged.xlsx` | The dataset the notebook loads (synthetic — see note below) |
| `make_data.py` | Script that generates `Telco_Customer_Churn_Merged.xlsx` |
| `telco_churn_best_model.pkl` | The final trained model, saved for reuse without retraining |
| `README.md` | This file |

> **Note on the data:** the original merged Excel file used for this project wasn't
> available when the notebook was completed, so `make_data.py` generates a synthetic
> dataset with the same column names and realistic, correlated churn patterns. If you
> have the real dataset, replace `Telco_Customer_Churn_Merged.xlsx` with it (same column
> names) and re-run the notebook — no code changes needed.

---

## 1. One-time setup

**Install Python** (if not already installed): download from
[python.org](https://www.python.org/downloads/) and make sure to check
**"Add python.exe to PATH"** during installation.

**Install the required packages.** Open Command Prompt / Terminal and run:

```
pip install notebook pandas numpy matplotlib seaborn scipy scikit-learn openpyxl xgboost joblib
```

If `pip` isn't recognized, use `python -m pip install ...` instead.

---

## 2. Put all files in one folder

Make sure `Telco Customer Churn Prediction Model.ipynb`, `Telco_Customer_Churn_Merged.xlsx`,
`make_data.py`, and `telco_churn_best_model.pkl` are all in the **same folder**. The
notebook loads the Excel file by name from wherever it's run.

---

## 3. Run the notebook

1. Open Command Prompt / Terminal, navigate into the project folder:
   ```
   cd path\to\your\Telco_Project
   ```
2. Launch Jupyter:
   ```
   python -m notebook
   ```
3. In the browser tab that opens, click `Telco Customer Churn Prediction Model.ipynb`.
4. Run everything top to bottom: **Cell → Run All** (or **Run → Run All Cells**).
5. Wait for it to finish — model training near the end takes a minute or two. No red
   error boxes means it worked.

### What you'll see after running
- Churn distribution charts (bar + pie)
- A 4-panel EDA plot (contract type, tenure, monthly charges, payment method vs. churn)
- 10 hypothesis tests (chi-square / t-test / Mann-Whitney / ANOVA) with p-values and
  effect sizes
- A summary table with **Bonferroni-corrected** significance verdicts
- A correlation heatmap
- 7 trained models (Logistic Regression, Decision Tree, KNN, Gradient Boosting, Random
  Forest, SVM, XGBoost) with confusion matrices and metrics
- ROC curve comparison across all models
- Feature importance charts
- 5-fold cross-validation results
- Hyperparameter tuning (grid search) on the top models
- The final tuned model saved to `telco_churn_best_model.pkl`
- A written conclusions section at the bottom

---

## 4. Regenerating the dataset (optional)

Only needed if you want a fresh synthetic dataset (e.g. different random seed). From
the project folder:

```
python make_data.py
```

This overwrites `Telco_Customer_Churn_Merged.xlsx`.

---

## 5. Testing whether a new customer will churn

Add a new cell at the bottom of the notebook (click the last cell, then press `b` to
add one below it), paste this in, edit the `new_customer` values to describe the
customer you want to check, and run it:

```python
import joblib
import pandas as pd

# 1. Load the saved model bundle
bundle = joblib.load("telco_churn_best_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

# 2. Describe your ONE new customer here — same fields as the raw dataset
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0,
    "Age": 34,
    "SatisfactionScore": 2,
    "CLTV": 3200,
    "NumReferrals": 0,
}])

# 3. Apply the SAME encoding used in training
for col in new_customer.select_dtypes(include="object").columns:
    if new_customer[col].nunique() == 1:
        new_customer[col] = new_customer[col].map(
            {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        ).fillna(new_customer[col])

new_customer_encoded = pd.get_dummies(new_customer)

# 4. Align columns to exactly what the model expects
new_customer_final = new_customer_encoded.reindex(columns=feature_columns, fill_value=0)

# 5. Predict
prediction = model.predict(new_customer_final)[0]
probability = model.predict_proba(new_customer_final)[0][1]

print("Prediction:", "WILL CHURN" if prediction == 1 else "WILL STAY")
print(f"Churn probability: {probability:.1%}")
```

Just change the field values (tenure, contract, monthly charges, etc.) to test different
customers. The output gives both a yes/no prediction and a churn probability percentage.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `'jupyter' is not recognized` | Use `python -m notebook` instead of `jupyter notebook` |
| `ModuleNotFoundError: No module named 'xgboost'` (or any other package) | Run `pip install <package name>` in a new cell: `!pip install xgboost`, then re-run |
| Notebook can't find the `.xlsx` file | Make sure it's in the same folder as the notebook, and that you launched Jupyter from that folder |
| `python` / `pip` not recognized at all | Reinstall Python and make sure "Add python.exe to PATH" was checked, then restart the terminal |

---

## 7. Important disclaimer for submission

This project currently runs on a **synthetic dataset** generated to mimic real Telco
churn patterns, since the original merged file wasn't available. If your coursework
requires the real dataset, swap `Telco_Customer_Churn_Merged.xlsx` for the genuine file
(same column names) before submitting, and mention this substitution if asked during
review or a viva.
