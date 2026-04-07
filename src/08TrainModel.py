# =============================================================================
#  PHASE 5 — MODEL TRAINING (LOGISTIC REGRESSION)
#  Churn Prediction Project
#
#  What this file does:
#  1. Loads features.csv from Phase 3
#  2. Splits into train / test (80/20, stratified)
#  3. Handles class imbalance with class_weight='balanced'
#     (no SMOTE needed — logistic regression handles it natively)
#  4. Trains Logistic Regression with StandardScaler in a Pipeline
#  5. Cross-validates with 5-fold StratifiedKFold
#  6. Prints every metric with a plain-English explanation
#  7. Saves model to models/lr_model.pkl for Phase 6
#
#  Install:
#  pip install scikit-learn joblib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# =============================================================================
#  HELPER
# =============================================================================

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# =============================================================================
#  0. LOAD DATA
# =============================================================================

header("PHASE 5 — LOGISTIC REGRESSION TRAINING")

df = pd.read_csv("data/features.csv")
print(f"\n  Loaded features.csv : {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  Churn rate          : {df['churn'].mean():.1%}  ← imbalanced (~27%)")

# =============================================================================
#  1. FEATURE SELECTION
# =============================================================================

header("1. FEATURE SELECTION")

# These columns are not features — drop them
DROP_COLS = ['customerID', 'churn', 'Churn Label',
             'Contract', 'Internet', 'ticket_bucket']

FEATURE_COLS = [c for c in df.columns
                if c not in DROP_COLS and c in df.columns]

X = df[FEATURE_COLS].copy()
y = df['churn'].copy()

# Fill any remaining nulls with column median
# (charges_per_tenure is NaN when tenure = 0)
for col in X.columns:
    if X[col].isnull().any():
        median = X[col].median()
        X[col] = X[col].fillna(median)
        print(f"  Filled nulls in '{col}' with median = {median:.2f}")

print(f"\n  Features ({len(FEATURE_COLS)} total):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {col}")

# =============================================================================
#  2. TRAIN / TEST SPLIT
#
#  Why 80/20?
#  80% gives the model enough data to learn patterns.
#  20% is held out completely — the model never sees it during training.
#  This held-out set gives us an honest estimate of real-world performance.
#
#  Why stratify=y?
# When you see stratify=y in your Python code, you are telling the computer:
# "Look at the y column (the churners). When you split the data into Train and Test, make sure both sides still have exactly 27% churners."
#  Without stratify, the random split might give you 20% churn in train
#  but 35% churn in test — making evaluation unfair.
#  stratify=y guarantees both sets have the same churn ratio (~27%).
# =============================================================================

header("2. TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    # random state => test and train data will be different always when I press run so always differnet result so we set the random state 42 is a random number .
    stratify     = y
)

print(f"\n  Train : {X_train.shape[0]:,} rows  "
      f"| churn rate = {y_train.mean():.1%}")
print(f"  Test  : {X_test.shape[0]:,} rows  "
      f"| churn rate = {y_test.mean():.1%}")
print(f"\n  Both sets have the same churn ratio — stratify worked correctly.")

# =============================================================================
#  3. BUILD THE PIPELINE
#
#  A Pipeline chains steps so they run in order automatically.
#  Step 1 — StandardScaler:
#    Logistic Regression is sensitive to feature scale.
#    tenure ranges 0-72, high_priority_pct ranges 0-1.
#    Without scaling, tenure dominates just because its numbers are bigger.
#    StandardScaler converts every feature to mean=0, std=1.
#
#  Step 2 — LogisticRegression:
#    class_weight='balanced' automatically adjusts for imbalance.
#    It gives churners (~27%) more weight so the model does not
#    just learn to predict "stayed" for everyone.
#    C=1.0 is the regularisation strength — higher C = less regularisation.
#    max_iter=1000 gives the solver enough steps to converge.
# =============================================================================

header("3. BUILDING THE PIPELINE")
# StandardScaler converts every column so they all have a Mean of 0 and a Standard Deviation of 1. Now, every feature speaks the same "language."
# B. C=1.0 (The "Strictness" Filter)
# This is called Regularization.
# High C (e.g., 100): The model is very "Loose." it tries to fit every single data point perfectly (Danger: Overfitting).
# Low C (e.g., 0.01): The model is very "Strict." It ignores small details to find the big, simple patterns.
# C=1.0 is the "Goldilocks" zone—not too strict, not too loose.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        class_weight = 'balanced',
        C            = 1.0,
        
        max_iter     = 1000,
        random_state = 42,
        solver       = 'lbfgs'
    ))
])

print("""
  Pipeline steps:
    1. StandardScaler      — scale all features to mean=0, std=1
    2. LogisticRegression  — class_weight='balanced' handles imbalance
                             no SMOTE needed — LR does this natively
""")

# =============================================================================
#  4. TRAIN THE MODEL
# =============================================================================

header("4. TRAINING")

pipeline.fit(X_train, y_train)
print("  Model trained successfully.")

# Get predictions and probabilities on the test set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of churn

# =============================================================================
#  5. METRICS — WITH PLAIN ENGLISH EXPLANATIONS
#
#  Why not just use accuracy?
#  If we predict "nobody churns" we get ~73% accuracy.
#  That model is useless — it catches zero churners.
#  We need metrics that care about HOW WELL we catch the minority class.
# =============================================================================

header("5. MODEL PERFORMANCE ON TEST SET")

accuracy  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

cm        = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  Metric        Value    What it means                   │
  ├─────────────────────────────────────────────────────────┤
  │  Accuracy      {accuracy:.4f}   {accuracy*100:.1f}% of all predictions correct       │
  │                         (misleading — use ROC-AUC)      │
  │                                                         │
  │  ROC-AUC       {roc_auc:.4f}   How well model RANKS churners     │
  │                         above non-churners (0.5=random, │
  │                         1.0=perfect)  ← PRIMARY METRIC  │
  │                                                         │
  │  Precision     {precision:.4f}   Of customers we flagged as      │
  │                         churners, {precision*100:.1f}% actually churned  │
  │                                                         │
  │  Recall        {recall:.4f}   Of ALL actual churners, we       │
  │                         caught {recall*100:.1f}% of them           │
  │                                                         │
  │  F1-Score      {f1:.4f}   Harmonic mean of P and R.      │
  │                         Use when P and R both matter     │
  └─────────────────────────────────────────────────────────┘

  Confusion Matrix breakdown:
    True Negatives  (correctly said "stayed")  : {TN:,}
    False Positives (wrongly flagged as churner): {FP:,}
    False Negatives (missed actual churners)    : {FN:,}  ← most costly
    True Positives  (correctly caught churners) : {TP:,}

  Plain English:
    Out of {y_test.sum():,} actual churners in the test set,
    the model caught {TP:,} and missed {FN:,}.
    It sent {FP:,} unnecessary retention alerts to loyal customers.
""")

print("  Full classification report:")
print(classification_report(y_test, y_pred,
                            target_names=['Stayed', 'Churned']))

# =============================================================================
#  6. CROSS-VALIDATION
#
#  A single 80/20 split can be lucky or unlucky depending on which
#  customers ended up in each set.
#  Cross-validation splits the TRAINING data into 5 folds.
#  It trains on 4 folds and validates on 1 fold — 5 times.
#  The average across all 5 folds is a much more reliable performance estimate.
#
#  We use StratifiedKFold to preserve the churn ratio in every fold.
# =============================================================================

header("6. 5-FOLD STRATIFIED CROSS-VALIDATION")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv      = cv,
    scoring = ['roc_auc', 'f1', 'precision', 'recall'],
    n_jobs  = -1
    # n_jobs = -1 tells Python: "Use all the CPU cores in my laptop at the same time."
)

print(f"\n  {'Metric':<12}  {'Mean':>7}  {'Std':>7}  {'All 5 folds'}")
print(f"  {'-'*58}")
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cv_results[f'test_{metric}']
    folds  = '  '.join(f'{s:.3f}' for s in scores)
    print(f"  {metric:<12}  {scores.mean():>7.4f}  "
          f"{scores.std():>7.4f}  {folds}")

print(f"""
  How to read this:
    Mean  = average performance across 5 folds (the number to report)
    Std   = how consistent the model is (lower = more stable)
    If std is very high (>0.05), the model is unstable — investigate.
    If CV mean >> test set score, the model may be overfitting.
""")

# =============================================================================
#  7. COEFFICIENT ANALYSIS
#
#  This is the biggest advantage of Logistic Regression over Random Forest.
#  Every feature has a coefficient that tells you:
#    Positive coefficient → higher value pushes toward churn
#    Negative coefficient → higher value pushes away from churn
#
#  The raw coefficient is log-odds.
#  exp(coefficient) = odds ratio — easier to explain to business.
#
#  Example: contract_encoded coefficient = -1.2
#  exp(-1.2) = 0.30 → a one-unit increase in contract (e.g. month-to-month
#  to one-year) multiplies churn odds by 0.30 — a 70% reduction in odds.
# =============================================================================
# ===========================================================
# This is the "Why" of your model. While other AI models (like Random Forest) are "Black Boxes" that just give you an answer, Logistic Regression is an "Open Book." It tells you exactly how much every feature—like tenure or monthly charges—influences a customer's decision to leave.

# 1. The Coefficient (The "Weight")
# Every feature in your model gets assigned a number called a Coefficient. Think of it as a "Vote" for or against Churn.
# Positive Coefficient (+): These are Risk Factors. As this number goes up, the chance of Churn goes up.
# Example: If total_tickets has a positive coefficient, it means more complaints = more churn.
# Negative Coefficient (-): These are Protective Factors. As this number goes up, the chance of Churn goes down.
# Example: If tenure has a negative coefficient, it means more months of loyalty = less churn.

# ///////////////////////////////////
# /////////////////////////
# Imagine your contract_encoded has a coefficient of -1.2.The Math: e^{-1.2} approx 0.30.The Interpretation: Moving from "Month-to-Month" to "One-Year" multiplies the "Odds of Churning" by 0.30.
# The Result: Since 0.30 is 70% less than 1.0, you can tell your manager: "If we can move a customer to a one-year contract, we reduce their risk of leaving by 70%!"
# ///////////////////////////
# ///////////////////////////////////
# ===========================================================
header("7. COEFFICIENT ANALYSIS — WHY THE MODEL DECIDES")

lr_model     = pipeline.named_steps['model']
scaler       = pipeline.named_steps['scaler']
coefficients = lr_model.coef_[0]

coef_df = pd.DataFrame({
    'feature'    : FEATURE_COLS,
    'coefficient': coefficients,
    'odds_ratio' : np.exp(coefficients),
}).sort_values('coefficient', ascending=False).reset_index(drop=True)

print(f"\n  {'Feature':<28} {'Coef':>8}  {'Odds Ratio':>11}  Meaning")
print(f"  {'-'*72}")

for _, row in coef_df.iterrows():
    coef = row['coefficient']
    OR   = row['odds_ratio']
    if coef > 0.1:
        meaning = f"↑ churn risk  (OR={OR:.2f}x more likely)"
    elif coef < -0.1:
        meaning = f"↓ churn risk  (OR={OR:.2f}x less likely)"
    else:
        meaning = "  negligible effect"
    print(f"  {row['feature']:<28} {coef:>8.4f}  {OR:>11.4f}  {meaning}")

# Top 3 insights in plain English
top3_churn    = coef_df.head(3)
top3_protect  = coef_df.tail(3).iloc[::-1]

print(f"""
  ─────────────────────────────────────────────────
  TOP 3 FACTORS DRIVING CHURN:
  (positive coefficient = pushes toward churn)
""")
for _, row in top3_churn.iterrows():
    print(f"    {row['feature']:<28} "
          f"odds ratio = {row['odds_ratio']:.2f}x more likely to churn")

print(f"""
  TOP 3 FACTORS PROTECTING AGAINST CHURN:
  (negative coefficient = pushes away from churn)
""")
for _, row in top3_protect.iterrows():
    print(f"    {row['feature']:<28} "
          f"odds ratio = {row['odds_ratio']:.2f}x less likely to churn")

# =============================================================================
#  8. COEFFICIENT PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 7))

colors = [C_RED if c > 0 else C_BLUE
          for C_RED, C_BLUE, c
          in [('#E05C5C', '#4A90D9', c) for c in coef_df['coefficient']]]

bars = ax.barh(
    coef_df['feature'],
    coef_df['coefficient'],
    color  = colors,
    edgecolor = 'white',
    height = 0.6
)

ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (log-odds)\n'
              'Red = pushes toward churn   |   '
              'Blue = pushes away from churn')
ax.set_title('Logistic Regression Coefficients\n'
             'What drives churn in this model?',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Value labels on bars
for bar, val in zip(bars, coef_df['coefficient']):
    ax.text(
        val + (0.02 if val >= 0 else -0.02),
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=8
    )

plt.tight_layout()
plt.savefig('plots/09_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved → plots/09_coefficients.png")

# =============================================================================
#  9. SAVE MODEL AND ARTIFACTS FOR PHASE 6
# =============================================================================

header("8. SAVING MODEL")

joblib.dump({
    'pipeline'    : pipeline,
    'feature_cols': FEATURE_COLS,
    'X_test'      : X_test,
    'y_test'      : y_test,
    'y_pred'      : y_pred,
    'y_prob'      : y_prob,
    'coef_df'     : coef_df,
    'metrics'     : {
        'accuracy' : accuracy,
        'roc_auc'  : roc_auc,
        'f1'       : f1,
        'precision': precision,
        'recall'   : recall,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
    }
}, 'models/lr_model.pkl')

print(f"""
  Saved → models/lr_model.pkl
  Contains: pipeline, feature list, test data,
            predictions, coefficients, all metrics
""")

print("="*60)
print(f"  PHASE 5 COMPLETE")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"  Recall   : {recall:.4f}  ({TP:,} churners caught, {FN:,} missed)")
print("="*60)