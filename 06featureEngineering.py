# =============================================================================
#  PHASE 3 — FEATURE ENGINEERING
# =============================================================================
#
#  What this file does:
#  Takes the 3 raw tables and builds a single ML-ready DataFrame
#  where every row = one customer, every column = one feature.
#
#  Features we build:
#  From usage_logs    → login_drop_pct, avg_monthly_logins, avg_data_gb,
#                       login_trend_slope
#  From support_tickets→ total_tickets, tickets_per_month,
#                        high_priority_pct, unresolved_pct, billing_pct
#  From raw_customers → tenure, monthly_charges, total_charges,
#                       contract_encoded, payment_encoded,
#                       internet_encoded, charges_per_month_tenure
#  Target column      → churn (1 = churned, 0 = stayed)
#
# =============================================================================

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse
import os

# =============================================================================
#  0. CONNECTION AND RAW DATA LOAD
# =============================================================================

password = "Aditya@123"
safe_pwd = urllib.parse.quote_plus(password)
DB_URL   = f"postgresql://postgres:{safe_pwd}@localhost:5432/postgres"
engine   = create_engine(DB_URL)

print("="*55)
print("  PHASE 3 — FEATURE ENGINEERING")
print("="*55)
print("\nLoading raw tables...")

customers = pd.read_sql('SELECT * FROM raw_customers',   engine)
tickets   = pd.read_sql('SELECT * FROM support_tickets', engine)
usage     = pd.read_sql('SELECT * FROM usage_logs',      engine)

print(f"  raw_customers   : {customers.shape}")
print(f"  support_tickets : {tickets.shape}")
print(f"  usage_logs      : {usage.shape}")


# =============================================================================
#  1. USAGE FEATURES
#  Source: usage_logs (3 rows per customer — Jan, Feb, Mar)
#  Goal  : collapse to 1 row per customer with trend features
# =============================================================================

print("\n--- Building usage features ---")

# Pivot: one row per customer, months become columns
usage['month_date'] = pd.to_datetime(usage['month_date'])

usage_pivot = usage.pivot_table(
    index   = 'customerID',
    columns = 'month_date',
    values  = ['login_count', 'data_usage_gb'],
    aggfunc = 'first'
).reset_index()

# Flatten the multi-level column names pivot creates
# e.g. ('login_count', 2026-01-01) → 'login_jan'
usage_pivot.columns = [
    'customerID'   if col[0] == 'customerID'       else
    'login_jan'    if col == ('login_count',    pd.Timestamp('2026-01-01')) else
    'login_feb'    if col == ('login_count',    pd.Timestamp('2026-02-01')) else
    'login_mar'    if col == ('login_count',    pd.Timestamp('2026-03-01')) else
    'data_jan'     if col == ('data_usage_gb',  pd.Timestamp('2026-01-01')) else
    'data_feb'     if col == ('data_usage_gb',  pd.Timestamp('2026-02-01')) else
    'data_mar'     if col == ('data_usage_gb',  pd.Timestamp('2026-03-01')) else
    str(col)
    for col in usage_pivot.columns
]

# Feature 1: login_drop_pct
# How much did logins fall from Jan to Mar?
# Positive = dropped (bad sign), Negative = grew (good sign)
usage_pivot['login_drop_pct'] = (
    (usage_pivot['login_jan'] - usage_pivot['login_mar'])
    / usage_pivot['login_jan'].replace(0, np.nan)   # avoid division by zero
    * 100
).round(2)

# Feature 2: avg_monthly_logins
# Average logins across all 3 months — overall engagement level
usage_pivot['avg_monthly_logins'] = (
    usage_pivot[['login_jan', 'login_feb', 'login_mar']].mean(axis=1).round(2)
)

# Feature 3: avg_data_gb
# Average data usage across all 3 months
usage_pivot['avg_data_gb'] = (
    usage_pivot[['data_jan', 'data_feb', 'data_mar']].mean(axis=1).round(2)
)

# Feature 4: login_trend_slope
# Fits a straight line through Jan, Feb, Mar login counts
# Negative slope = consistently declining engagement (strong churn signal)
# We use numpy's polyfit(x, y, degree=1) — degree 1 = straight line
def compute_slope(row):
    y = [row['login_jan'], row['login_feb'], row['login_mar']]
    if any(pd.isna(y)):
        return np.nan
    x = [0, 1, 2]   # Jan=0, Feb=1, Mar=2
    slope, _ = np.polyfit(x, y, 1)
    return round(slope, 3)

usage_pivot['login_trend_slope'] = usage_pivot.apply(compute_slope, axis=1)

# Keep only the engineered features (drop raw monthly columns)
usage_features = usage_pivot[[
    'customerID', 'login_drop_pct', 'avg_monthly_logins',
    'avg_data_gb', 'login_trend_slope'
]]

print(f"  usage_features shape    : {usage_features.shape}")
print(f"  Sample login_drop_pct   : {usage_features['login_drop_pct'].describe().to_dict()}")


# =============================================================================
#  2. TICKET FEATURES
#  Source: support_tickets (multiple rows per customer)
#  Goal  : collapse to 1 row per customer with complaint pattern features
# =============================================================================

print("\n--- Building ticket features ---")

# Number of months in our data window (Jan–Mar = 3)
N_MONTHS = 3

ticket_features = (
    tickets
    .groupby('customerID')
    .agg(
        total_tickets     = ('ticket_id', 'count'),
        high_priority_cnt = ('priority',  lambda x: x.isin(['High','Critical']).sum()),
        unresolved_cnt    = ('status',    lambda x: x.isin(['Open','Pending']).sum()),
        billing_cnt       = ('category',  lambda x: (x == 'Billing').sum()),
    )
    .reset_index()
)

# Feature 5: tickets_per_month
# Normalises ticket count by time window so it's comparable across projects
ticket_features['tickets_per_month'] = (
    ticket_features['total_tickets'] / N_MONTHS
).round(3)

# Feature 6: high_priority_pct
# What fraction of their tickets are serious?
# 0 = all low priority, 1 = all high/critical
ticket_features['high_priority_pct'] = (
    ticket_features['high_priority_cnt'] / ticket_features['total_tickets']
).round(3)

# Feature 7: unresolved_pct
# Unresolved tickets signal ongoing frustration
ticket_features['unresolved_pct'] = (
    ticket_features['unresolved_cnt'] / ticket_features['total_tickets']
).round(3)

# Feature 8: billing_pct
# Billing complaints are strongly correlated with churn in telecom
ticket_features['billing_pct'] = (
    ticket_features['billing_cnt'] / ticket_features['total_tickets']
).round(3)

# Keep only engineered columns
ticket_features = ticket_features[[
    'customerID', 'total_tickets', 'tickets_per_month',
    'high_priority_pct', 'unresolved_pct', 'billing_pct'
]]

print(f"  ticket_features shape   : {ticket_features.shape}")
print(f"  Customers with tickets  : {len(ticket_features)}")
print(f"  Customers without any   : {customers.shape[0] - len(ticket_features)}")


# =============================================================================
#  3. CUSTOMER FEATURES
#  Source: raw_customers
#  Goal  : encode categoricals, engineer ratio features
# =============================================================================

print("\n--- Building customer features ---")

# Work on a copy — never modify the raw DataFrame in place
cust = customers[[
    'customerID', 'tenure', 'Contract', 'MonthlyCharges',
    'TotalCharges', 'PaymentMethod', 'InternetService',
    'TechSupport', 'OnlineSecurity', 'Churn'
]].copy()

# Feature 9: tenure (already numeric — keep as-is)
# Higher tenure = less likely to churn (loyal customers)

# Feature 10: monthly_charges (already numeric)
cust = cust.rename(columns={
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges'  : 'total_charges',
})

# Feature 11: charges_per_tenure
# Total paid divided by months — should equal monthly charges if no changes
# Outliers here flag pricing changes or data errors
cust['charges_per_tenure'] = (
    cust['total_charges'] / cust['tenure'].replace(0, np.nan)
).round(2)

# Feature 12: contract_encoded
# Ordinal encoding — longer contract = higher number = lower churn risk
contract_map = {
    'Month-to-month': 0,   # highest churn risk
    'One year'      : 1,
    'Two year'      : 2,   # lowest churn risk
}
cust['contract_encoded'] = cust['Contract'].map(contract_map)

# Feature 13: payment_encoded
# Electronic check customers churn more (no auto-renewal friction)
payment_map = {
    'Electronic check'          : 0,   # most likely to churn
    'Mailed check'              : 1,
    'Bank transfer (automatic)' : 2,
    'Credit card (automatic)'   : 3,   # least likely (auto-pay = sticky)
}
cust['payment_encoded'] = cust['PaymentMethod'].map(payment_map)

# Feature 14: internet_encoded
# Fiber optic customers churn more despite (or because of) higher price
internet_map = {
    'No'          : 0,
    'DSL'         : 1,
    'Fiber optic' : 2,
}
cust['internet_encoded'] = cust['InternetService'].map(internet_map)

# Feature 15: support_services_count
# How many protective services does the customer have?
# (TechSupport + OnlineSecurity — both reduce churn significantly)
cust['support_services_count'] = (
    (cust['TechSupport']    == 'Yes').astype(int) +
    (cust['OnlineSecurity'] == 'Yes').astype(int)
)

# Target: churn encoded as 0/1 integer
cust['churn'] = (cust['Churn'] == 'Yes').astype(int)

customer_features = cust[[
    'customerID', 'tenure', 'monthly_charges', 'total_charges',
    'charges_per_tenure', 'contract_encoded', 'payment_encoded',
    'internet_encoded', 'support_services_count', 'churn'
]]

print(f"  customer_features shape : {customer_features.shape}")
print(f"  Churn rate              : {cust['churn'].mean():.1%}")


# =============================================================================
#  4. MERGE ALL FEATURES INTO ONE DATAFRAME
# =============================================================================

print("\n--- Merging all features ---")

# Start with all customers as the base (LEFT JOIN everything onto it)
# This ensures every customer appears exactly once
df = customer_features.copy()

# Merge usage features — every customer has usage data so this is a full match
df = df.merge(usage_features, on='customerID', how='left')

# Merge ticket features — customers with NO tickets get NaN here
# We fill with 0 because 0 tickets = 0 frequency, 0 %, etc.
df = df.merge(ticket_features, on='customerID', how='left')

ticket_cols = ['total_tickets', 'tickets_per_month',
               'high_priority_pct', 'unresolved_pct', 'billing_pct']
df[ticket_cols] = df[ticket_cols].fillna(0)

print(f"  Final df shape          : {df.shape}")
print(f"  Columns                 : {list(df.columns)}")


# =============================================================================
#  5. FINAL CHECKS
# =============================================================================

print("\n--- Final feature checks ---")

# Check for any remaining nulls
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
if nulls.empty:
    print("  PASS — no nulls in final feature DataFrame")
else:
    print("  Columns with nulls (likely charges_per_tenure for tenure=0):")
    print(nulls)
    # Fill remaining nulls with median — safe default for tree-based models
    for col in nulls.index:
        median_val = df[col].median()
        df[col]    = df[col].fillna(median_val)
        print(f"    Filled '{col}' nulls with median={median_val:.2f}")

# Quick feature summary
print("\n  Feature summary:")
feature_cols = [c for c in df.columns if c not in ['customerID', 'churn']]
print(df[feature_cols].describe().round(2).to_string())

# Churn balance check
churn_rate = df['churn'].mean()
print(f"\n  Churn rate in final df  : {churn_rate:.1%}")
if churn_rate < 0.35:
    print("  NOTE: class imbalance detected — use SMOTE or class_weight in Phase 5")


# =============================================================================
#  6. SAVE TO CSV AND WRITE BACK TO POSTGRES
# =============================================================================

# Save locally as CSV — used by Phase 4 (EDA) and Phase 5 (model)
os.makedirs("data", exist_ok=True)
df.to_csv("data/features.csv", index=False)
print(f"\n  Saved → data/features.csv  ({df.shape[0]:,} rows, {df.shape[1]} columns)")

# Also write back to Postgres so you can query features with SQL later
df.to_sql('features', engine, if_exists='replace', index=False)
print("  Saved → PostgreSQL table 'features'")

print("\n" + "="*55)
print("  PHASE 3 COMPLETE")
print(f"  {df.shape[1] - 2} features built for {df.shape[0]:,} customers")
print("  Next → python phase4_eda.py")
print("="*55)