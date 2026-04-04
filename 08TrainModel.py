# =============================================================================
#  PHASE 5 — MODEL TRAINING
#  Churn Prediction Project
#  Run: python phase5_model.py
#
#  What this file does:
#  1. Loads features.csv from Phase 3
#  2. Handles class imbalance (SMOTE + class_weight)
#  3. Trains Logistic Regression baseline
#  4. Trains Random Forest
#  5. Cross-validates both models properly
#  6. Compares results in a clean scorecard
#  7. Saves the best model to disk for Phase 6
#
#  Install dependencies first:
#  pip install scikit-learn imbalanced-learn joblib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline           import Pipeline
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline as ImbPipeline

os.makedirs("plots",  exist_ok=True)
os.makedirs("models", exist_ok=True)

# =============================================================================
#  0. LOAD DATA
# =============================================================================

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

header("PHASE 5 — MODEL TRAINING")

df = pd.read_csv("data/features.csv")
print(f"\n  Loaded features.csv : {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  Churn rate          : {df['churn'].mean():.1%}  ← imbalanced")

# =============================================================================
#  1. FEATURE SELECTION AND TRAIN/TEST SPLIT
# =============================================================================

header("1. FEATURE SELECTION AND SPLIT")

# Drop non-feature columns
DROP_COLS = ['customerID', 'churn', 'Churn Label', 'Contract',
             'Internet', 'ticket_bucket']
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS
                and c in df.columns]

X = df[FEATURE_COLS].copy()
y = df['churn'].copy()

print(f"\n  Features used ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {col}")

# Impute any remaining nulls with column median
# (charges_per_tenure can be NaN for tenure=0 customers)
for col in X.columns:
    if X[col].isnull().any():
        median = X[col].median()
        X[col] = X[col].fillna(median)
        print(f"  Imputed nulls in '{col}' with median={median:.2f}")

# Stratified split — preserves churn ratio in both train and test
# test_size=0.2 means 80% train, 20% test
# random_state=42 makes results reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y      # ← crucial for imbalanced data
)

print(f"\n  Train set : {X_train.shape[0]:,} rows "
      f"(churn rate={y_train.mean():.1%})")
print(f"  Test set  : {X_test.shape[0]:,} rows  "
      f"(churn rate={y_test.mean():.1%})")

# =============================================================================
#  2. CLASS IMBALANCE — SMOTE
#  SMOTE = Synthetic Minority Over-sampling TEchnique
#  Creates synthetic churn examples so the model sees balanced classes
#  IMPORTANT: apply SMOTE only to training data, NEVER to test data
#  Applying it to test data would be data leakage — a critical mistake
# =============================================================================

header("2. HANDLING CLASS IMBALANCE (SMOTE)")

print(f"\n  Before SMOTE — Train set:")
print(f"    Stayed  (0): {(y_train==0).sum():,}")
print(f"    Churned (1): {(y_train==1).sum():,}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\n  After SMOTE — Train set:")
print(f"    Stayed  (0): {(y_train_sm==0).sum():,}")
print(f"    Churned (1): {(y_train_sm==1).sum():,}")
print(f"  Classes are now balanced — model will learn both equally")

# =============================================================================
#  3. MODEL A — LOGISTIC REGRESSION (BASELINE)
#  Always start with the simplest model.
#  If a complex model barely beats logistic regression,
#  the simple model wins (easier to explain, faster, less likely to overfit)
# =============================================================================

header("3A. LOGISTIC REGRESSION — BASELINE")

# Pipeline = scaler + model in one object
# Scaler is critical for logistic regression — features must be on same scale
# Without scaling, tenure (0-72) dominates monthly_charges (18-118)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        class_weight = 'balanced',  # alternative to SMOTE — penalises misclassifying minority
        max_iter     = 1000,        # more iterations for convergence
        random_state = 42,
        C            = 1.0          # regularisation strength (1/lambda) — default is fine
    ))
])

# Train on SMOTE-balanced data
lr_pipeline.fit(X_train_sm, y_train_sm)

# Predict on UNTOUCHED test set
y_pred_lr    = lr_pipeline.predict(X_test)
y_prob_lr    = lr_pipeline.predict_proba(X_test)[:, 1]  # probability of churn

print(f"\n  Logistic Regression — Test Set Results:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(y_test, y_prob_lr):.4f}  ← primary metric")
print(f"    F1-Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred_lr):.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_lr,
                            target_names=['Stayed','Churned']))

# =============================================================================
#  4. CROSS-VALIDATION — LOGISTIC REGRESSION
#  Single train/test split can be lucky or unlucky.
#  Cross-validation splits data 5 ways and trains/tests 5 times,
#  giving a much more reliable estimate of real-world performance.
# =============================================================================

header("3B. CROSS-VALIDATION — LOGISTIC REGRESSION")

# StratifiedKFold preserves churn ratio in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We cross-validate on the ORIGINAL (non-SMOTE) training data
# and use an ImbPipeline that applies SMOTE inside each fold
# This is the CORRECT way — prevents data leakage from SMOTE
lr_cv_pipeline = ImbPipeline([
    ('smote',  SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ))
])

lr_cv_results = cross_validate(
    lr_cv_pipeline, X_train, y_train, cv=cv,
    scoring  = ['roc_auc', 'f1', 'precision', 'recall'],
    n_jobs   = -1    # use all CPU cores
)

print(f"\n  Logistic Regression — 5-Fold CV Results:")
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = lr_cv_results[f'test_{metric}']
    print(f"    {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"(folds: {', '.join(f'{s:.3f}' for s in scores)})")

# =============================================================================
#  5. MODEL B — RANDOM FOREST
#  Ensemble of decision trees — handles non-linearity, feature interactions,
#  and is naturally resistant to outliers.
#  No scaling needed — trees split on thresholds not distances.
# =============================================================================

header("4A. RANDOM FOREST")

rf_model = RandomForestClassifier(
    n_estimators = 200,        # 200 trees — more = better up to a point
    max_depth    = 15,         # limit depth to prevent overfitting
    min_samples_leaf = 10,     # each leaf needs at least 10 samples
    class_weight = 'balanced', # handle imbalance (in addition to SMOTE)
    random_state = 42,
    n_jobs       = -1          # use all CPU cores
)

# Random forest does NOT need scaling — train directly on SMOTE data
rf_model.fit(X_train_sm, y_train_sm)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"\n  Random Forest — Test Set Results:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(y_test, y_prob_rf):.4f}  ← primary metric")
print(f"    F1-Score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred_rf):.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_rf,
                            target_names=['Stayed','Churned']))

# =============================================================================
#  6. CROSS-VALIDATION — RANDOM FOREST
# =============================================================================

header("4B. CROSS-VALIDATION — RANDOM FOREST")

rf_cv_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=10, class_weight='balanced',
        random_state=42, n_jobs=-1
    ))
])

rf_cv_results = cross_validate(
    rf_cv_pipeline, X_train, y_train, cv=cv,
    scoring = ['roc_auc', 'f1', 'precision', 'recall'],
    n_jobs  = -1
)

print(f"\n  Random Forest — 5-Fold CV Results:")
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = rf_cv_results[f'test_{metric}']
    print(f"    {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"(folds: {', '.join(f'{s:.3f}' for s in scores)})")

# =============================================================================
#  7. HEAD-TO-HEAD COMPARISON SCORECARD
# =============================================================================

header("5. MODEL COMPARISON SCORECARD")

lr_auc = roc_auc_score(y_test, y_prob_lr)
rf_auc = roc_auc_score(y_test, y_prob_rf)
lr_f1  = f1_score(y_test, y_pred_lr)
rf_f1  = f1_score(y_test, y_pred_rf)
lr_rec = recall_score(y_test, y_pred_lr)
rf_rec = recall_score(y_test, y_pred_rf)
lr_pre = precision_score(y_test, y_pred_lr)
rf_pre = precision_score(y_test, y_pred_rf)

def winner(a, b):
    return "LR  <--" if a > b else "RF  <--"

print(f"""
  {'Metric':<15} {'Log. Regression':>17} {'Random Forest':>15} {'Better':>10}
  {'-'*60}
  {'ROC-AUC':<15} {lr_auc:>17.4f} {rf_auc:>15.4f} {winner(lr_auc,rf_auc):>10}
  {'F1-Score':<15} {lr_f1:>17.4f} {rf_f1:>15.4f} {winner(lr_f1,rf_f1):>10}
  {'Recall':<15} {lr_rec:>17.4f} {rf_rec:>15.4f} {winner(lr_rec,rf_rec):>10}
  {'Precision':<15} {lr_pre:>17.4f} {rf_pre:>15.4f} {winner(lr_pre,rf_pre):>10}
  {'-'*60}

  PRIMARY METRIC = ROC-AUC (not accuracy — data is imbalanced)

  Why ROC-AUC?
  A model predicting nobody churns gets ~73% accuracy but 0% recall.
  ROC-AUC measures how well the model RANKS churners above non-churners.
  A score of 0.5 = random guessing, 1.0 = perfect.

  Why Recall matters for churn?
  Missing a churner (False Negative) = lost customer revenue.
  A wrong alert (False Positive) = wasted retention offer ($5-10).
  In churn, missing churners costs MORE, so we favour high Recall.
""")

# =============================================================================
#  8. FEATURE IMPORTANCE (RANDOM FOREST)
#  Shows which features the model found most useful.
#  This is what you explain to a business stakeholder.
# =============================================================================

header("6. FEATURE IMPORTANCE — RANDOM FOREST")

importance_df = pd.DataFrame({
    'feature'   : FEATURE_COLS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print(f"\n  Top 10 features by importance:")
for _, row in importance_df.head(10).iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"    {row['feature']:<28} {row['importance']:.4f}  {bar}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(9, 6))
top10 = importance_df.head(10)
colors = ['#E05C5C' if i < 3 else '#4A90D9' for i in range(len(top10))]
ax.barh(top10['feature'][::-1], top10['importance'][::-1],
        color=colors[::-1], edgecolor='white')
ax.set_xlabel('Feature importance (Gini impurity reduction)')
ax.set_title('Top 10 features — Random Forest\n'
             '(red = top 3 most important)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved → plots/09_feature_importance.png")

# =============================================================================
#  9. CONFUSION MATRICES — SIDE BY SIDE
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices — Test Set', fontsize=13, fontweight='bold')

for ax, y_pred, title in zip(
    axes,
    [y_pred_lr, y_pred_rf],
    ['Logistic Regression', 'Random Forest']
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed','Churned'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title)

    # Annotate TN, FP, FN, TP
    labels = [['TN\n(correct)', 'FP\n(false alarm)'],
              ['FN\n(missed!)',  'TP\n(caught!)']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, labels[i][j],
                    ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('plots/10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → plots/10_confusion_matrices.png")

# =============================================================================
#  10. SAVE BEST MODEL
# =============================================================================

header("7. SAVING BEST MODEL")

best_model_name = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
best_model      = rf_model         if rf_auc >= lr_auc else lr_pipeline
best_auc        = max(rf_auc, lr_auc)

# Save model, feature list, test data for Phase 6
joblib.dump({
    'model'        : best_model,
    'model_name'   : best_model_name,
    'feature_cols' : FEATURE_COLS,
    'X_test'       : X_test,
    'y_test'       : y_test,
    'y_prob'       : y_prob_rf if rf_auc >= lr_auc else y_prob_lr,
    'y_pred'       : y_pred_rf if rf_auc >= lr_auc else y_pred_lr,
    'lr_pipeline'  : lr_pipeline,
    'rf_model'     : rf_model,
    'y_prob_lr'    : y_prob_lr,
    'y_prob_rf'    : y_prob_rf,
    'y_pred_lr'    : y_pred_lr,
    'y_pred_rf'    : y_pred_rf,
}, 'models/best_model.pkl')

print(f"""
  Best model : {best_model_name}
  ROC-AUC    : {best_auc:.4f}
  Saved to   : models/best_model.pkl

  Next → python phase6_evaluation.py
""")

print("="*60)
print("  PHASE 5 COMPLETE")
print("="*60)