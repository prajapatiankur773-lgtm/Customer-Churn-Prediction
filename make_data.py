import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 7043

def pick(cats, p, size=n):
    return rng.choice(cats, size=size, p=p)

# ---- core demographics ----
gender = pick(["Male", "Female"], [0.50, 0.50])
senior = pick([0, 1], [0.84, 0.16])
partner = pick(["Yes", "No"], [0.48, 0.52])
# dependents more likely if has partner
dependents = np.where(
    partner == "Yes",
    pick(["Yes", "No"], [0.42, 0.58]),
    pick(["Yes", "No"], [0.12, 0.88]),
)

age = np.where(senior == 1,
               rng.integers(65, 91, size=n),
               rng.integers(18, 65, size=n))

# ---- tenure (right-skewed, lots of new + lots of long-term) ----
tenure = np.clip(rng.gamma(shape=1.6, scale=18, size=n), 0, 72).round().astype(int)

# ---- phone / lines ----
phone_service = pick(["Yes", "No"], [0.90, 0.10])
multiple_lines = np.array([
    "No phone service" if ps == "No" else rng.choice(["Yes", "No"], p=[0.42, 0.58])
    for ps in phone_service
])

# ---- internet service ----
internet_service = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])

def internet_dependent_col(p_yes_given_internet):
    out = []
    for isv in internet_service:
        if isv == "No":
            out.append("No internet service")
        else:
            out.append(rng.choice(["Yes", "No"], p=[p_yes_given_internet, 1 - p_yes_given_internet]))
    return np.array(out)

online_security = internet_dependent_col(0.29)
online_backup = internet_dependent_col(0.34)
device_protection = internet_dependent_col(0.34)
tech_support = internet_dependent_col(0.29)
streaming_tv = internet_dependent_col(0.38)
streaming_movies = internet_dependent_col(0.39)

# ---- contract / billing ----
contract = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
paperless_billing = pick(["Yes", "No"], [0.59, 0.41])
payment_method = pick(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    [0.34, 0.23, 0.22, 0.21],
)

# ---- charges ----
base_charge = np.select(
    [internet_service == "No", internet_service == "DSL", internet_service == "Fiber optic"],
    [20.0, 55.0, 80.0],
)
addon_count = (
    (online_security == "Yes").astype(int) + (online_backup == "Yes").astype(int) +
    (device_protection == "Yes").astype(int) + (tech_support == "Yes").astype(int) +
    (streaming_tv == "Yes").astype(int) + (streaming_movies == "Yes").astype(int)
)
phone_addon = (phone_service == "Yes").astype(int) * rng.normal(5, 1, n)
monthly_charges = np.clip(
    base_charge + addon_count * rng.normal(5.5, 1.0, n) + phone_addon + rng.normal(0, 3, n),
    18.25, 118.75
)
total_charges = np.clip(monthly_charges * tenure + rng.normal(0, 30, n), 0, None)
# brand new customers -> blank string (mirrors real dataset quirk)
total_charges_str = np.where(tenure == 0, "", np.round(total_charges, 2).astype(str))

# ---- satisfaction score (drives churn, like the real IBM extended dataset) ----
sat_latent = (
    2.6
    + 0.020 * tenure
    - 0.012 * monthly_charges
    + np.where(contract == "Two year", 1.1, np.where(contract == "One year", 0.5, 0))
    + np.where(tech_support == "Yes", 0.4, 0)
    + np.where(online_security == "Yes", 0.25, 0)
    - np.where(internet_service == "Fiber optic", 0.35, 0)
    - np.where(payment_method == "Electronic check", 0.3, 0)
    + rng.normal(0, 0.6, n)
)
satisfaction_score = np.clip(np.round(sat_latent), 1, 5).astype(int)

# ---- churn probability (logistic combination) ----
logit = (
    -0.60
    - 0.045 * tenure
    + 0.016 * monthly_charges
    - 0.55 * satisfaction_score
    + np.where(contract == "Month-to-month", 1.0, np.where(contract == "One year", 0.1, -0.9))
    + np.where(internet_service == "Fiber optic", 0.35, 0)
    + np.where(payment_method == "Electronic check", 0.45, 0)
    - np.where(tech_support == "Yes", 0.35, 0)
    - np.where(online_security == "Yes", 0.3, 0)
    + np.where(senior == 1, 0.25, 0)
)
churn_prob = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

# ---- CLTV & referrals (extended IBM fields), correlated with satisfaction/tenure/churn ----
cltv = np.clip(
    3000 + 25 * tenure + 120 * satisfaction_score + rng.normal(0, 400, n)
    - np.where(churn == "Yes", 250, 0),
    2000, 6500
).round().astype(int)

referral_lambda = np.clip(0.15 * satisfaction_score + 0.01 * tenure - (churn == "Yes") * 0.4, 0.05, None)
num_referrals = rng.poisson(referral_lambda)

customer_id = np.array([f"{rng.integers(1000,9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)])

raw = pd.DataFrame({
    "Customer ID": customer_id,
    "Gender": gender,
    "Senior Citizen_x": senior,
    "Partner": partner,
    "Dependents_x": dependents,
    "Tenure": tenure,
    "Phone Service_x": phone_service,
    "Multiple Lines_x": multiple_lines,
    "Internet Service_x": internet_service,
    "Online Security_x": online_security,
    "Online Backup_x": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV_x": streaming_tv,
    "Streaming Movies_x": streaming_movies,
    "Contract_x": contract,
    "Paperless Billing_x": paperless_billing,
    "Payment Method_x": payment_method,
    "Monthly Charges": monthly_charges.round(2),
    "Total Charges_x": total_charges_str,
    "Churn": churn,
    "Age": age,
    "Satisfaction Score": satisfaction_score,
    "CLTV": cltv,
    "Number of Referrals": num_referrals,
})

raw.to_excel("Telco_Customer_Churn_Merged.xlsx", index=False)
print("Saved", raw.shape)
print(raw["Churn"].value_counts(normalize=True))
