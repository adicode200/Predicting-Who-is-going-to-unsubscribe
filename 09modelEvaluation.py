# =============================================================================
#  PHASE 6 — MODEL EVALUATION AND INTERPRETATION
#  Churn Prediction Project
#  Run: python phase6_evaluation.py
#
#  What this file does:
#  1. Loads the saved model from Phase 5
#  2. Plots detailed confusion matrix with business cost framing
#  3. Plots ROC-AUC curve for both models
#  4. Plots Precision-Recall curve and finds optimal threshold
#  5. SHAP values — explains WHY the model makes each prediction
#  6. Prints a final business summary with actionable numbers
#
#  Install:
#  pip install shap matplotlib scikit-learn joblib
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import shap

os.makedirs("plots",   exist_ok=True)
os.makedirs("reports", exist_ok=True)

# =============================================================================
#  STYLE
# =============================================================================

plt.rcParams.update({
    'figure.dpi'       : 150,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
})

C_BLUE   = '#4A90D9'
C_RED    = '#E05C5C'
C_ORANGE = '#F0A070'
C_GREEN  = '#5CB85C'
C_GRAY   = '#888888'

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def save(filename):
    path = f"plots/{filename}"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

# =============================================================================
#  0. LOAD SAVED MODEL AND DATA FROM PHASE 5
# =============================================================================

header("PHASE 6 — EVALUATION AND INTERPRETATION")

print("\n  Loading model from Phase 5...")
bundle        = joblib.load('models/best_model.pkl')
best_model    = bundle['model']
best_name     = bundle['model_name']
feature_cols  = bundle['feature_cols']
X_test        = bundle['X_test']
y_test        = bundle['y_test']
y_prob_rf     = bundle['y_prob_rf']
y_prob_lr     = bundle['y_prob_lr']
y_pred_rf     = bundle['y_pred_rf']
y_pred_lr     = bundle['y_pred_lr']
rf_model      = bundle['rf_model']

print(f"  Best model          : {best_name}")
print(f"  Test set size       : {len(y_test):,} customers")
print(f"  Actual churn rate   : {y_test.mean():.1%}")
print(f"  RF   ROC-AUC        : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  LR   ROC-AUC        : {roc_auc_score(y_test, y_prob_lr):.4f}")

# =============================================================================
#  1. CONFUSION MATRIX — WITH BUSINESS COST FRAMING
#  A confusion matrix alone is just numbers.
#  The insight is translating TP/FP/FN/TN into dollars.
# =============================================================================

header("1. CONFUSION MATRIX WITH BUSINESS COST")

# Business assumptions (adjust to your domain)
REVENUE_PER_CUSTOMER_MONTHLY = 65   # avg monthly revenue per customer ($)
RETENTION_OFFER_COST         = 20   # cost of a retention offer ($)
MONTHS_SAVED                 = 6    # avg months retained after intervention

y_pred = y_pred_rf
y_prob = y_prob_rf

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

cost_fn  = FN * REVENUE_PER_CUSTOMER_MONTHLY * MONTHS_SAVED
cost_fp  = FP * RETENTION_OFFER_COST
value_tp = TP * (REVENUE_PER_CUSTOMER_MONTHLY * MONTHS_SAVED - RETENTION_OFFER_COST)
net_value = value_tp - cost_fn - cost_fp

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: visual confusion matrix
disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed', 'Churned'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion matrix — Random Forest (test set)')

for i, (row_label, col_label, label, color) in enumerate([
    (0, 0, 'True Negative\n(Correct: stayed)',       C_BLUE),
    (0, 1, 'False Positive\n(Wrong alarm)',           C_ORANGE),
    (1, 0, 'False Negative\n(Missed churner!)',       C_RED),
    (1, 1, 'True Positive\n(Caught churner)',         C_GREEN),
]):
    r, c = divmod(i, 2)
    axes[0].text(c, r + 0.38, label,
                 ha='center', fontsize=8, color=color, fontweight='bold')

# Right: business cost breakdown bar chart
categories  = ['TP: Revenue\nsaved',
               'FP: Wasted\noffers',
               'FN: Missed\nchurners',
               'Net value']
values      = [value_tp, -cost_fp, -cost_fn, net_value]
bar_colors  = [C_GREEN, C_ORANGE, C_RED,
               C_GREEN if net_value > 0 else C_RED]

bars = axes[1].bar(categories, values, color=bar_colors,
                   edgecolor='white', width=0.5)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title('Business value of model predictions\n'
                  f'(assumptions: ${REVENUE_PER_CUSTOMER_MONTHLY}/mo, '
                  f'${RETENTION_OFFER_COST} offer cost)')
axes[1].set_ylabel('Estimated value ($)')
axes[1].yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f'${x:,.0f}')
)
for bar, val in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (500 if val >= 0 else -2500),
        f'${abs(val):,.0f}',
        ha='center', fontsize=9, fontweight='bold'
    )

save("11_confusion_matrix_business.png")

print(f"""
  Confusion Matrix Breakdown:
    True Negatives  (correctly predicted stayed) : {TN:,}
    False Positives (wrongly flagged as churner) : {FP:,}
    False Negatives (missed actual churners)     : {FN:,}  ← most costly
    True Positives  (correctly caught churners)  : {TP:,}

  Business Cost Estimate (with model):
    Revenue saved (TP × ${REVENUE_PER_CUSTOMER_MONTHLY}/mo × {MONTHS_SAVED}mo) : ${value_tp:,.0f}
    Wasted offers (FP × ${RETENTION_OFFER_COST})                  : ${cost_fp:,.0f}
    Missed revenue (FN × ${REVENUE_PER_CUSTOMER_MONTHLY}/mo × {MONTHS_SAVED}mo): ${cost_fn:,.0f}
    ─────────────────────────────────────────────────────
    Net estimated value of model                 : ${net_value:,.0f}
""")

# =============================================================================
#  2. ROC-AUC CURVE — BOTH MODELS ON SAME PLOT
#  ROC curve shows the tradeoff between True Positive Rate and
#  False Positive Rate at every possible threshold.
#  AUC = area under that curve. Higher = better.
#  Random baseline (coin flip) = diagonal line with AUC = 0.5
# =============================================================================

header("2. ROC-AUC CURVE")

fig, ax = plt.subplots(figsize=(8, 6))

for y_prob_plot, label, color in [
    (y_prob_rf, 'Random Forest',       C_BLUE),
    (y_prob_lr, 'Logistic Regression', C_ORANGE),
]:
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_plot)
    auc = roc_auc_score(y_test, y_prob_plot)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{label}  (AUC = {auc:.4f})')

# Diagonal = random baseline
ax.plot([0, 1], [0, 1], color=C_GRAY, linewidth=1.2,
        linestyle='--', label='Random baseline (AUC = 0.50)')

# Shade area under RF curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
ax.fill_between(fpr_rf, tpr_rf, alpha=0.08, color=C_BLUE)

ax.set_xlabel('False Positive Rate  (1 - Specificity)\n'
              '← How often we wrongly flag a loyal customer')
ax.set_ylabel('True Positive Rate  (Sensitivity / Recall)\n'
              '← How often we correctly catch a churner')
ax.set_title('ROC-AUC Curve — Logistic Regression vs Random Forest')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# Annotate perfect point
ax.annotate('Perfect\nclassifier', xy=(0, 1), xytext=(0.1, 0.88),
            arrowprops=dict(arrowstyle='->', color=C_GRAY),
            fontsize=9, color=C_GRAY)

save("12_roc_auc_curve.png")

print(f"  Random Forest  AUC : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  Logistic Reg.  AUC : {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"  Baseline (random)  : 0.5000")

# =============================================================================
#  3. PRECISION-RECALL CURVE + OPTIMAL THRESHOLD
#  PR curve is more informative than ROC when classes are imbalanced.
#  It shows: if I lower my threshold to catch more churners (higher Recall),
#  how many false alarms do I introduce (lower Precision)?
#
#  FINDING THE OPTIMAL THRESHOLD:
#  Default threshold = 0.5 (predict churn if prob > 50%)
#  But for churn, catching more churners is worth some false alarms.
#  We find the threshold that maximises F1-Score (harmonic mean of P and R).
# =============================================================================

header("3. PRECISION-RECALL CURVE AND OPTIMAL THRESHOLD")

precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
    y_test, y_prob_rf
)
avg_precision = average_precision_score(y_test, y_prob_rf)

# Find threshold that maximises F1
f1_scores  = 2 * (precision_vals * recall_vals) / (
    precision_vals + recall_vals + 1e-9
)
best_idx       = np.argmax(f1_scores[:-1])
best_threshold = pr_thresholds[best_idx]
best_precision = precision_vals[best_idx]
best_recall    = recall_vals[best_idx]
best_f1        = f1_scores[best_idx]

print(f"""
  Default threshold (0.5):
    Precision : {precision_score(y_test, y_pred_rf):.4f}
    Recall    : {recall_score(y_test, y_pred_rf):.4f}
    F1        : {f1_score(y_test, y_pred_rf):.4f}

  Optimal threshold ({best_threshold:.3f}):
    Precision : {best_precision:.4f}
    Recall    : {best_recall:.4f}
    F1        : {best_f1:.4f}
    → Lower threshold = catch MORE churners at cost of more false alarms
""")

# Apply optimal threshold
y_pred_optimal = (y_prob_rf >= best_threshold).astype(int)
TN_o, FP_o, FN_o, TP_o = confusion_matrix(y_test, y_pred_optimal).ravel()

print(f"  With optimal threshold {best_threshold:.3f}:")
print(f"    Extra churners caught (vs default) : "
      f"{TP_o - TP:+,}")
print(f"    Extra false alarms   (vs default)  : "
      f"{FP_o - FP:+,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: PR curve
axes[0].plot(recall_vals, precision_vals, color=C_BLUE,
             linewidth=2.5, label=f'Random Forest (AP={avg_precision:.3f})')
axes[0].axhline(y_test.mean(), color=C_GRAY, linestyle='--',
                linewidth=1.2,
                label=f'Baseline (AP={y_test.mean():.3f})')
axes[0].scatter(best_recall, best_precision, color=C_RED,
                s=120, zorder=5,
                label=f'Optimal threshold={best_threshold:.2f}\n'
                      f'(P={best_precision:.2f}, R={best_recall:.2f})')
axes[0].set_xlabel('Recall  (fraction of churners caught)')
axes[0].set_ylabel('Precision  (fraction of alerts that are real)')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.02)

# Right: P, R, F1 vs threshold
threshold_range = np.linspace(0.1, 0.9, 200)
prec_vals, rec_vals, f1_vals = [], [], []
for t in threshold_range:
    y_t = (y_prob_rf >= t).astype(int)
    if y_t.sum() == 0:
        prec_vals.append(0); rec_vals.append(0); f1_vals.append(0)
    else:
        prec_vals.append(precision_score(y_test, y_t, zero_division=0))
        rec_vals.append(recall_score(y_test, y_t))
        f1_vals.append(f1_score(y_test, y_t))

axes[1].plot(threshold_range, prec_vals, color=C_BLUE,
             linewidth=2, label='Precision')
axes[1].plot(threshold_range, rec_vals, color=C_RED,
             linewidth=2, label='Recall')
axes[1].plot(threshold_range, f1_vals, color=C_GREEN,
             linewidth=2.5, label='F1-Score', linestyle='-.')
axes[1].axvline(0.5, color=C_GRAY, linestyle=':', linewidth=1.2,
                label='Default threshold (0.5)')
axes[1].axvline(best_threshold, color=C_ORANGE, linestyle='--',
                linewidth=1.8,
                label=f'Optimal threshold ({best_threshold:.2f})')
axes[1].set_xlabel('Classification threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision, Recall and F1 vs Threshold')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0.1, 0.9)
axes[1].set_ylim(0, 1.02)

save("13_precision_recall_threshold.png")

# =============================================================================
#  4. SHAP VALUES
#  SHAP = SHapley Additive exPlanations
#  Answers: for THIS specific customer, why did the model predict churn?
#  Each feature gets a SHAP value = how much it pushed the prediction
#  up (toward churn) or down (away from churn).
#  Red bars = pushed toward churn, Blue bars = pushed away from churn.
# =============================================================================

header("4. SHAP VALUES — MODEL INTERPRETATION")

print("  Computing SHAP values (this takes ~30 seconds)...")

# TreeExplainer is fast and exact for tree-based models like Random Forest
explainer   = shap.TreeExplainer(rf_model)

# Use a sample of test data for speed — 300 is enough for reliable SHAP values
sample_size = min(300, len(X_test))
X_sample    = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

# shap_values has shape [n_classes, n_samples, n_features]
# Index [1] = SHAP values for class 1 (churn)
shap_churn = shap_values[1] if isinstance(shap_values, list) else shap_values

print(f"  SHAP values computed for {sample_size} customers")

# ── Plot A: SHAP Summary (Beeswarm) ──────────────────────────────────────────
# Every dot = one customer
# X position = SHAP value (how much this feature pushed toward/away from churn)
# Colour = feature value (red = high value, blue = low value)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_churn, X_sample,
    feature_names = feature_cols,
    show          = False,
    plot_type     = 'dot',
    max_display   = 15,
    color_bar     = True,
)
plt.title('SHAP Summary — Feature impact on churn prediction\n'
          'Red dot = high feature value,  '
          'Right of 0 = pushes toward churn',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/14_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → plots/14_shap_summary.png")

# ── Plot B: SHAP Bar Chart (Mean Absolute) ───────────────────────────────────
# Average |SHAP value| per feature = overall importance, ignoring direction
mean_abs_shap = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': np.abs(shap_churn).mean(axis=0)
}).sort_values('importance', ascending=False).head(12)

fig, ax = plt.subplots(figsize=(9, 6))
colors  = [C_RED if i < 3 else C_BLUE for i in range(len(mean_abs_shap))]
ax.barh(mean_abs_shap['feature'][::-1],
        mean_abs_shap['importance'][::-1],
        color=colors[::-1], edgecolor='white')
ax.set_xlabel('Mean |SHAP value|  (average impact on prediction)')
ax.set_title('SHAP Feature Importance\n(red = top 3 most impactful)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/15_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → plots/15_shap_bar.png")

print(f"\n  Top 10 features by mean |SHAP value|:")
for _, row in mean_abs_shap.head(10).iterrows():
    bar = '█' * int(row['importance'] * 400)
    print(f"    {row['feature']:<28} {row['importance']:.4f}  {bar}")

# ── Plot C: Single Customer Waterfall ────────────────────────────────────────
# Pick the highest-risk customer — show exactly WHY they are high risk
# This is the most powerful plot for business stakeholders
highest_risk_idx = np.argmax(y_prob_rf[X_sample.index - X_sample.index.min()]
                              if hasattr(X_sample.index, 'min')
                              else y_prob_rf[:sample_size])
highest_risk_idx = int(np.argmax(
    y_prob_rf[X_test.index.get_indexer(X_sample.index)]
))

fig, ax = plt.subplots(figsize=(10, 6))
shap_exp = shap.Explanation(
    values        = shap_churn[highest_risk_idx],
    base_values   = explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
    data          = X_sample.iloc[highest_risk_idx].values,
    feature_names = feature_cols,
)
shap.waterfall_plot(shap_exp, max_display=12, show=False)
plt.title(f'SHAP Waterfall — Highest-Risk Customer\n'
          f'Predicted churn probability: '
          f'{y_prob_rf[X_test.index.get_indexer(X_sample.index)[highest_risk_idx]]:.1%}',
          fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/16_shap_waterfall_high_risk.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved → plots/16_shap_waterfall_high_risk.png")

# =============================================================================
#  5. FINAL BUSINESS SUMMARY REPORT
# =============================================================================

header("5. FINAL BUSINESS SUMMARY")

n_high_risk   = (y_prob_rf >= 0.7).sum()
n_medium_risk = ((y_prob_rf >= 0.4) & (y_prob_rf < 0.7)).sum()
n_low_risk    = (y_prob_rf < 0.4).sum()

top_feature = mean_abs_shap.iloc[0]['feature']

summary = f"""
╔══════════════════════════════════════════════════════════════╗
║           PHASE 6 — FINAL EVALUATION SUMMARY                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MODEL PERFORMANCE (Random Forest on held-out test set)      ║
║  ─────────────────────────────────────────────────────────   ║
║  ROC-AUC          : {roc_auc_score(y_test, y_prob_rf):.4f}                            ║
║  F1-Score         : {f1_score(y_test, y_pred_rf):.4f}  (default threshold 0.5)  ║
║  Precision        : {precision_score(y_test, y_pred_rf):.4f}                            ║
║  Recall           : {recall_score(y_test, y_pred_rf):.4f}                            ║
║  Optimal threshold: {best_threshold:.3f}  (maximises F1)              ║
║                                                              ║
║  RISK SEGMENTATION (test set, {len(y_prob_rf):,} customers)           ║
║  ─────────────────────────────────────────────────────────   ║
║  High risk   (prob >= 0.70) : {n_high_risk:>4,} customers              ║
║  Medium risk (0.40 – 0.70) : {n_medium_risk:>4,} customers              ║
║  Low risk    (prob <  0.40) : {n_low_risk:>4,} customers              ║
║                                                              ║
║  BUSINESS VALUE (test set estimates)                         ║
║  ─────────────────────────────────────────────────────────   ║
║  Net estimated model value  : ${net_value:>8,.0f}                ║
║  Churners caught (TP)       : {TP:>4,}                          ║
║  Missed churners (FN)       : {FN:>4,}  ← priority to reduce   ║
║                                                              ║
║  TOP CHURN DRIVERS (from SHAP)                               ║
║  ─────────────────────────────────────────────────────────   ║
║  #{mean_abs_shap.iloc[0]['feature']:<28} (SHAP={mean_abs_shap.iloc[0]['importance']:.4f})   ║
║  #{mean_abs_shap.iloc[1]['feature']:<28} (SHAP={mean_abs_shap.iloc[1]['importance']:.4f})   ║
║  #{mean_abs_shap.iloc[2]['feature']:<28} (SHAP={mean_abs_shap.iloc[2]['importance']:.4f})   ║
║                                                              ║
║  NEXT STEP → python phase7_score.py                          ║
╚══════════════════════════════════════════════════════════════╝
"""
print(summary)

# Save summary to text file
with open('reports/phase6_summary.txt', 'w') as f:
    f.write(summary)
print("  Saved → reports/phase6_summary.txt")

print("\n  Plots generated:")
for i, name in enumerate([
    '11_confusion_matrix_business.png',
    '12_roc_auc_curve.png',
    '13_precision_recall_threshold.png',
    '14_shap_summary.png',
    '15_shap_bar.png',
    '16_shap_waterfall_high_risk.png',
], 1):
    print(f"    {i}. plots/{name}")

print(f"\n{'='*60}")
print("  PHASE 6 COMPLETE")
print(f"{'='*60}")