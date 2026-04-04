
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse

# =============================================================================
#  0. CONNECTION
# =============================================================================

password   = "Aditya@123"
safe_pwd   = urllib.parse.quote_plus(password)
DB_URL     = f"postgresql://postgres:{safe_pwd}@localhost:5432/postgres"
engine     = create_engine(DB_URL)

print("Loading tables — please wait...")

customers = pd.read_sql('SELECT * FROM raw_customers',   engine)
tickets   = pd.read_sql('SELECT * FROM support_tickets', engine)
usage     = pd.read_sql('SELECT * FROM usage_logs',      engine)

print(f"  raw_customers   : {customers.shape[0]:,} rows, {customers.shape[1]} columns")
print(f"  support_tickets : {tickets.shape[0]:,} rows, {tickets.shape[1]} columns")
print(f"  usage_logs      : {usage.shape[0]:,} rows, {usage.shape[1]} columns")


# =============================================================================
#  HELPER
# =============================================================================

def header(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# =============================================================================
#  1. NULL CHECKS
# =============================================================================

def check_nulls(df, name):
    header(f"NULL CHECK — {name}")
    null_counts = df.isnull().sum()
    null_pct    = (df.isnull().mean() * 100).round(2)
    result      = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
    result      = result[result['null_count'] > 0]

    if result.empty:
        print("  PASS — no nulls found")
    else:
        print(f"  FAIL — {len(result)} column(s) have nulls:\n")
        print(result.to_string())
    return result


null_customers = check_nulls(customers, "raw_customers")
null_tickets   = check_nulls(tickets,   "support_tickets")
null_usage     = check_nulls(usage,     "usage_logs")

# Hard assertions — script stops here if critical columns have nulls
critical_cols = ['customerID', 'Churn', 'tenure', 'MonthlyCharges']
for col in critical_cols:
    assert customers[col].isnull().sum() == 0, \
        f"\nCRITICAL: '{col}' has nulls — fix before proceeding!"
print("\n  All critical column assertions passed.")


# =============================================================================
#  2. DUPLICATE CHECKS
# =============================================================================

def check_duplicates(df, name, key_col):
    header(f"DUPLICATE CHECK — {name}")
    full_dups = df.duplicated().sum()
    key_dups  = df.duplicated(subset=[key_col]).sum()

    print(f"  Full duplicate rows     : {full_dups:<6} {'PASS' if full_dups == 0 else 'FAIL'}")
    print(f"  Duplicate '{key_col}'   : {key_dups:<6} {'PASS' if key_dups  == 0 else 'FAIL'}")

    if key_dups > 0:
        print("\n  Offending IDs:")
        print(df[df.duplicated(subset=[key_col], keep=False)][[key_col]].head(10))

    return full_dups, key_dups


check_duplicates(customers, "raw_customers",   "customerID")
check_duplicates(tickets,   "support_tickets", "ticket_id")

# usage_logs must have exactly 3 rows per customer (Jan, Feb, Mar)
header("DUPLICATE CHECK — usage_logs (3 months per customer)")
rows_per_customer = usage.groupby("customerID").size()
bad_months        = rows_per_customer[rows_per_customer != 3]
print(f"  Customers without exactly 3 months : {len(bad_months)}  "
      f"{'PASS' if len(bad_months) == 0 else 'FAIL'}")
if not bad_months.empty:
    print(bad_months.head())


# =============================================================================
#  3. REFERENTIAL INTEGRITY
# =============================================================================

def check_referential_integrity(parent_df, child_df, key, parent_name, child_name):
    header(f"REFERENTIAL INTEGRITY — {child_name} → {parent_name}")
    parent_ids = set(parent_df[key])
    child_ids  = set(child_df[key])
    orphans    = child_ids - parent_ids
    unmatched  = parent_ids - child_ids

    print(f"  {parent_name} unique IDs : {len(parent_ids):,}")
    print(f"  {child_name} unique IDs  : {len(child_ids):,}")
    print(f"  Orphan IDs (child not in parent)   : {len(orphans):<6} "
          f"{'PASS' if len(orphans) == 0 else 'FAIL'}")
    print(f"  Unmatched  (parent not in child)   : {len(unmatched):<6} "
          f"(ok — not all customers have tickets)")

    if orphans:
        print(f"  Sample orphan IDs: {list(orphans)[:5]}")

    return orphans, unmatched


check_referential_integrity(customers, tickets, "customerID",
                            "raw_customers", "support_tickets")
check_referential_integrity(customers, usage,   "customerID",
                            "raw_customers", "usage_logs")


# =============================================================================
#  4. SCHEMA AND VALUE CHECKS
# =============================================================================

header("SCHEMA AND VALUE CHECKS")

# Numeric dtype check
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
print("  Numeric dtype checks:")
for col in numeric_cols:
    ok = pd.api.types.is_numeric_dtype(customers[col])
    print(f"    {col:<20} dtype={str(customers[col].dtype):<10} "
          f"{'PASS' if ok else 'FAIL — not numeric'}")

# Allowed categorical values
print("\n  Allowed value checks (raw_customers):")
allowed_customers = {
    'Churn'        : {'Yes', 'No'},
    'Contract'     : {'Month-to-month', 'One year', 'Two year'},
    'PaymentMethod': {'Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)'},
}
for col, valid in allowed_customers.items():
    actual     = set(customers[col].dropna().unique())
    unexpected = actual - valid
    ok         = len(unexpected) == 0
    print(f"    {col:<20} {'PASS' if ok else f'FAIL — unexpected: {unexpected}'}")

print("\n  Allowed value checks (support_tickets):")
allowed_tickets = {
    'priority': {'Low', 'Medium', 'High', 'Critical'},
    'status'  : {'Open', 'Closed', 'Resolved', 'Pending'},
    'category': {'Technical', 'Billing', 'Outage', 'General'},
}
for col, valid in allowed_tickets.items():
    actual     = set(tickets[col].dropna().unique())
    unexpected = actual - valid
    ok         = len(unexpected) == 0
    print(f"    {col:<20} {'PASS' if ok else f'FAIL — unexpected: {unexpected}'}")

# No negative values
print("\n  Non-negative value checks:")
checks = [
    (customers, 'tenure',         'customers.tenure'),
    (customers, 'MonthlyCharges', 'customers.MonthlyCharges'),
    (usage,     'login_count',    'usage.login_count'),
    (usage,     'data_usage_gb',  'usage.data_usage_gb'),
]
for df, col, label in checks:
    neg = (df[col] < 0).sum()
    print(f"    {label:<30} negatives={neg}  {'PASS' if neg == 0 else 'FAIL'}")

# Tenure range sanity
t_min, t_max = customers['tenure'].min(), customers['tenure'].max()
print(f"\n  tenure range: {t_min} – {t_max} months  "
      f"{'PASS' if 0 <= t_min and t_max <= 120 else 'FAIL — out of range'}")


# =============================================================================
#  5. OUTLIER DETECTION (IQR METHOD)
# =============================================================================

def detect_outliers_iqr(df, col, label=""):
    Q1      = df[col].quantile(0.25)
    Q3      = df[col].quantile(0.75)
    IQR     = Q3 - Q1
    lower   = Q1 - 1.5 * IQR
    upper   = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    pct     = len(outliers) / len(df) * 100
    print(f"  {label or col}:")
    print(f"    Q1={Q1:.1f}  Q3={Q3:.1f}  IQR={IQR:.1f}  "
          f"bounds=[{lower:.1f}, {upper:.1f}]")
    print(f"    Outliers: {len(outliers)} ({pct:.1f}%)")
    return outliers


header("OUTLIER DETECTION — raw_customers")
detect_outliers_iqr(customers, 'tenure',         'tenure (months)')
detect_outliers_iqr(customers, 'MonthlyCharges', 'MonthlyCharges ($)')
detect_outliers_iqr(customers, 'TotalCharges',   'TotalCharges ($)')

header("OUTLIER DETECTION — usage_logs")
detect_outliers_iqr(usage, 'login_count',   'login_count')
detect_outliers_iqr(usage, 'data_usage_gb', 'data_usage_gb')


# =============================================================================
#  6. FINAL SCORECARD
# =============================================================================

def final_scorecard(customers, tickets, usage):
    header("FINAL VALIDATION SCORECARD")
    results = {}

    # Row counts
    results['customers_loaded'] = len(customers) > 0
    results['tickets_loaded']   = len(tickets)   > 0
    results['usage_loaded']     = len(usage)      > 0

    # Nulls
    crit_nulls = customers[
        ['customerID','Churn','tenure','MonthlyCharges']
    ].isnull().sum().sum()
    results['no_critical_nulls']  = (crit_nulls == 0)
    results['no_ticket_nulls']    = (tickets.isnull().sum().sum() == 0)
    results['no_usage_nulls']     = (usage.isnull().sum().sum()   == 0)

    # Duplicates
    results['no_duplicate_customers'] = (
        customers.duplicated(subset=['customerID']).sum() == 0)
    results['no_duplicate_tickets']   = (
        tickets.duplicated(subset=['ticket_id']).sum() == 0)
    results['usage_3_months_each']    = (
        (usage.groupby('customerID').size() != 3).sum() == 0)

    # Referential integrity
    cust_ids = set(customers['customerID'])
    results['no_orphan_tickets'] = (
        len(set(tickets['customerID']) - cust_ids) == 0)
    results['no_orphan_usage']   = (
        len(set(usage['customerID'])   - cust_ids) == 0)

    # Values
    results['no_negative_tenure']   = (customers['tenure']         >= 0).all()
    results['no_negative_charges']  = (customers['MonthlyCharges'] >= 0).all()
    results['no_negative_logins']   = (usage['login_count']        >= 0).all()
    results['no_negative_data']     = (usage['data_usage_gb']      >= 0).all()
    results['valid_churn_values']   = (
        set(customers['Churn'].dropna().unique()) <= {'Yes','No'})
    results['valid_contract_values']= (
        set(customers['Contract'].dropna().unique()) <=
        {'Month-to-month','One year','Two year'})

    # Print scorecard
    passed = sum(results.values())
    total  = len(results)

    for check, ok in results.items():
        icon = "PASS" if ok else "FAIL"
        print(f"  [{icon}]  {check}")

    print(f"\n{'='*55}")
    print(f"  RESULT: {passed}/{total} checks passed")
    print(f"{'='*55}")

    if passed == total:
        print("  ALL CHECKS PASSED — safe to proceed to Phase 3")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  {total - passed} check(s) FAILED:")
        for f in failed:
            print(f"    - {f}")
        print("  Fix the above before feature engineering.")

    return results


results = final_scorecard(customers, tickets, usage)