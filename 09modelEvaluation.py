# =============================================================================
#  PHASE 6 — MODEL EVALUATION AND INTERPRETATION (LOGISTIC REGRESSION)
#  Churn Prediction Project
#  Run: python phase6_evaluation.py
#
#  What this file does:
#  1. Loads lr_model.pkl saved by Phase 5
#  2. Confusion matrix with business cost framing
#  3. ROC-AUC curve
#  4. Precision-Recall curve + optimal threshold finder
#  5. Coefficient plot — replaces SHAP, explains model in plain English
#  6. Final business summary report
#
#  No new installs needed — uses scikit-learn and matplotlib only
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

os.makedirs("plots",   exist_ok=True)
os.makedirs("reports", exist_ok=True)

# =============================================================================
#  STYLE — consistent with Phase 4
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
#  0. LOAD MODEL AND DATA FROM PHASE 5
# =============================================================================

header("PHASE 6 — EVALUATION AND INTERPRETATION")

print("\n  Loading model from Phase 5...")
bundle       = joblib.load('models/lr_model.pkl')
pipeline     = bundle['pipeline']
feature_cols = bundle['feature_cols']
X_test       = bundle['X_test']
y_test       = bundle['y_test']
y_pred       = bundle['y_pred']
y_prob       = bundle['y_prob']
coef_df      = bundle['coef_df']
metrics      = bundle['metrics']

TP = metrics['TP']
TN = metrics['TN']
FP = metrics['FP']
FN = metrics['FN']

print(f"  Model           : Logistic Regression")
print(f"  Test set size   : {len(y_test):,} customers")
print(f"  Actual churn    : {y_test.mean():.1%}")
print(f"  ROC-AUC         : {metrics['roc_auc']:.4f}")
print(f"  Recall          : {metrics['recall']:.4f}  "
      f"({TP:,} churners caught, {FN:,} missed)")

# =============================================================================
#  1. CONFUSION MATRIX WITH BUSINESS COST
#
#  The confusion matrix alone is just 4 numbers.
#  The insight is translating those numbers into dollars.
#  This is what makes a data scientist useful to a business.
#
#  Business assumptions (change these to match your domain):
#    Revenue per customer per month : $65
#    Cost of one retention offer    : $20
#    Avg months saved after offer   : 6
# =============================================================================

header("1. CONFUSION MATRIX WITH BUSINESS COST")

REVENUE_MONTHLY  = 65
OFFER_COST       = 20
MONTHS_SAVED     = 6

value_tp  = TP * (REVENUE_MONTHLY * MONTHS_SAVED - OFFER_COST)
cost_fp   = FP * OFFER_COST
cost_fn   = FN * REVENUE_MONTHLY * MONTHS_SAVED
net_value = value_tp - cost_fp - cost_fn

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — confusion matrix heatmap
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed', 'Churned'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion matrix — Logistic Regression (test set)')

annotation = [
    (0, 0, 'True Negative\n(Correct: stayed)',  C_BLUE),
    (0, 1, 'False Positive\n(Wrong alarm)',      C_ORANGE),
    (1, 0, 'False Negative\n(Missed churner!)',  C_RED),
    (1, 1, 'True Positive\n(Caught churner)',    C_GREEN),
]
for col, row, label, color in annotation:
    axes[0].text(col, row + 0.38, label,
                 ha='center', fontsize=8,
                 color=color, fontweight='bold')

# Right — business value bar chart
categories = ['TP: Revenue\nsaved',
              'FP: Wasted\noffers',
              'FN: Missed\nchurners',
              'Net value']
values     = [value_tp, -cost_fp, -cost_fn, net_value]
bar_colors = [C_GREEN, C_ORANGE, C_RED,
              C_GREEN if net_value >= 0 else C_RED]

bars = axes[1].bar(categories, values,
                   color=bar_colors, edgecolor='white', width=0.5)
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title(f'Business value of predictions\n'
                  f'(${REVENUE_MONTHLY}/mo revenue, '
                  f'${OFFER_COST} offer cost, '
                  f'{MONTHS_SAVED} months saved)')
axes[1].set_ylabel('Estimated value ($)')
axes[1].yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f'${x:,.0f}')
)
for bar, val in zip(bars, values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (300 if val >= 0 else -2000),
        f'${abs(val):,.0f}',
        ha='center', fontsize=9, fontweight='bold'
    )

save("11_confusion_matrix_business.png")

print(f"""
  Confusion matrix:
    True Negatives  : {TN:,}   correctly predicted stayed
    False Positives : {FP:,}   wrongly flagged as churner
    False Negatives : {FN:,}    missed actual churners  ← most costly
    True Positives  : {TP:,}   correctly caught churners

  Business value estimate:
    Revenue saved (TP)  : ${value_tp:>10,.0f}
    Wasted offers (FP)  : ${cost_fp:>10,.0f}
    Missed revenue (FN) : ${cost_fn:>10,.0f}
    ─────────────────────────────────
    Net model value     : ${net_value:>10,.0f}
""")

# =============================================================================
#  2. ROC-AUC CURVE
#
#  ROC = Receiver Operating Characteristic
#  X axis = False Positive Rate (how often we wrongly flag loyal customers)
#  Y axis = True Positive Rate  (how often we correctly catch churners)
#
#  Every point on the curve = one possible threshold.
#  AUC = area under the curve.
#  0.5 = random guessing (diagonal line)
#  1.0 = perfect model
#
#  A good model hugs the top-left corner.
# =============================================================================

header("2. ROC-AUC CURVE")

fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr, tpr, color=C_BLUE, linewidth=2.5,
        label=f'Logistic Regression  (AUC = {auc:.4f})')
ax.fill_between(fpr, tpr, alpha=0.08, color=C_BLUE)
ax.plot([0, 1], [0, 1], color=C_GRAY, linewidth=1.2,
        linestyle='--', label='Random baseline  (AUC = 0.50)')

# Mark the point closest to top-left corner (best threshold on ROC)
distances      = np.sqrt(fpr**2 + (1 - tpr)**2)
best_roc_idx   = np.argmin(distances)
best_roc_fpr   = fpr[best_roc_idx]
best_roc_tpr   = tpr[best_roc_idx]
best_roc_thresh= roc_thresholds[best_roc_idx]

ax.scatter(best_roc_fpr, best_roc_tpr,
           color=C_RED, s=100, zorder=5,
           label=f'Best threshold = {best_roc_thresh:.2f}\n'
                 f'(FPR={best_roc_fpr:.2f}, TPR={best_roc_tpr:.2f})')

ax.annotate('Closer to here\n= better model',
            xy=(0, 1), xytext=(0.15, 0.82),
            arrowprops=dict(arrowstyle='->', color=C_GRAY),
            fontsize=9, color=C_GRAY)

ax.set_xlabel('False Positive Rate  (wrongly flagging loyal customers)')
ax.set_ylabel('True Positive Rate  (correctly catching churners)')
ax.set_title('ROC-AUC Curve — Logistic Regression')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

save("12_roc_auc_curve.png")
print(f"  ROC-AUC : {auc:.4f}")
print(f"  Best threshold on ROC curve : {best_roc_thresh:.3f}")

# =============================================================================
#  3. PRECISION-RECALL CURVE + OPTIMAL THRESHOLD
#
#  PR curve is more informative than ROC for imbalanced data.
#  It answers: if I lower my threshold to catch more churners,
#  how many false alarms do I accept?
#
#  Two subplots:
#    Left  — the PR curve with the optimal point marked
#    Right — Precision, Recall, F1 plotted against every threshold
#            so you can SEE the tradeoff and pick the right one
# =============================================================================

header("3. PRECISION-RECALL CURVE AND THRESHOLD ANALYSIS")

prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Find threshold that maximises F1
f1_curve  = (2 * prec_curve * rec_curve /
             (prec_curve + rec_curve + 1e-9))
best_idx   = np.argmax(f1_curve[:-1])
best_thresh= pr_thresholds[best_idx]
best_prec  = prec_curve[best_idx]
best_rec   = rec_curve[best_idx]
best_f1    = f1_curve[best_idx]

print(f"""
  Default threshold (0.50):
    Precision : {metrics['precision']:.4f}
    Recall    : {metrics['recall']:.4f}
    F1        : {metrics['f1']:.4f}

  Optimal threshold ({best_thresh:.3f})  — maximises F1:
    Precision : {best_prec:.4f}
    Recall    : {best_rec:.4f}
    F1        : {best_f1:.4f}

  Interpretation:
    Lowering threshold from 0.5 to {best_thresh:.2f} means:
    → We flag a customer as at-risk if churn probability > {best_thresh:.0%}
    → We catch MORE churners (higher Recall)
    → We also send more unnecessary offers (lower Precision)
    → Whether this is worth it depends on offer cost vs revenue saved
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — PR curve
axes[0].plot(rec_curve, prec_curve, color=C_BLUE,
             linewidth=2.5,
             label=f'Logistic Regression (AP={avg_precision:.3f})')
axes[0].axhline(y_test.mean(), color=C_GRAY, linestyle='--',
                linewidth=1.2,
                label=f'Baseline (AP={y_test.mean():.3f})')
axes[0].scatter(best_rec, best_prec, color=C_RED, s=120, zorder=5,
                label=f'Optimal threshold = {best_thresh:.2f}\n'
                      f'(P={best_prec:.2f}, R={best_rec:.2f}, '
                      f'F1={best_f1:.2f})')
axes[0].set_xlabel('Recall  (fraction of churners caught)')
axes[0].set_ylabel('Precision  (fraction of alerts that are real churners)')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.02)

# Right — P, R, F1 vs threshold
threshold_range = np.linspace(0.1, 0.9, 200)
p_vals, r_vals, f1_vals = [], [], []
for t in threshold_range:
    y_t = (y_prob >= t).astype(int)
    if y_t.sum() == 0:
        p_vals.append(0); r_vals.append(0); f1_vals.append(0)
    else:
        p_vals.append(precision_score(y_test, y_t, zero_division=0))
        r_vals.append(recall_score(y_test, y_t))
        f1_vals.append(f1_score(y_test, y_t))

axes[1].plot(threshold_range, p_vals,  color=C_BLUE,
             linewidth=2,   label='Precision')
axes[1].plot(threshold_range, r_vals,  color=C_RED,
             linewidth=2,   label='Recall')
axes[1].plot(threshold_range, f1_vals, color=C_GREEN,
             linewidth=2.5, label='F1-Score', linestyle='-.')
axes[1].axvline(0.5, color=C_GRAY, linestyle=':',
                linewidth=1.2, label='Default threshold (0.5)')
axes[1].axvline(best_thresh, color=C_ORANGE, linestyle='--',
                linewidth=1.8,
                label=f'Optimal threshold ({best_thresh:.2f})')
axes[1].set_xlabel('Classification threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision, Recall and F1 vs Threshold\n'
                  'Slide threshold left → catch more churners')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0.1, 0.9)
axes[1].set_ylim(0, 1.02)

save("13_precision_recall_threshold.png")

# =============================================================================
#  4. COEFFICIENT PLOT
#
#  This is the most important plot for Logistic Regression.
#  No SHAP library needed — the model explains itself.
#
#  Left plot  — raw coefficients (log-odds scale)
#               positive = pushes toward churn
#               negative = pushes away from churn
#
#  Right plot — odds ratios = exp(coefficient)
#               odds ratio > 1 → increases churn risk
#               odds ratio < 1 → decreases churn risk
#               odds ratio = 1 → no effect
#
#  The odds ratio is what you explain to a business stakeholder.
#  "A customer on month-to-month contract is 3.2x more likely to
#   churn than a two-year contract customer, all else being equal."
# =============================================================================

header("4. COEFFICIENT PLOT — WHY THE MODEL DECIDES")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sort by coefficient for both plots
plot_df = coef_df.sort_values('coefficient', ascending=True)

# Left — raw coefficients
colors_coef = [C_RED if c > 0 else C_BLUE
               for c in plot_df['coefficient']]
bars = axes[0].barh(plot_df['feature'], plot_df['coefficient'],
                    color=colors_coef, edgecolor='white', height=0.6)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_xlabel('Coefficient (log-odds)\n'
                   'Red = pushes toward churn  |  '
                   'Blue = pushes away from churn')
axes[0].set_title('Logistic Regression Coefficients')

for bar, val in zip(bars, plot_df['coefficient']):
    axes[0].text(
        val + (0.01 if val >= 0 else -0.01),
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}',
        va='center',
        ha='left' if val >= 0 else 'right',
        fontsize=8
    )

# Right — odds ratios
colors_or = [C_RED if o > 1 else C_BLUE
             for o in plot_df['odds_ratio']]
bars2 = axes[1].barh(plot_df['feature'], plot_df['odds_ratio'],
                     color=colors_or, edgecolor='white', height=0.6)
axes[1].axvline(1, color='black', linewidth=0.8,
                label='OR=1 (no effect)')
axes[1].set_xlabel('Odds Ratio  exp(coefficient)\n'
                   'Red > 1 = more likely to churn  |  '
                   'Blue < 1 = less likely to churn')
axes[1].set_title('Odds Ratios\n(easier to explain to business)')

for bar, val in zip(bars2, plot_df['odds_ratio']):
    axes[1].text(
        val + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.2f}x',
        va='center', ha='left', fontsize=8
    )

save("14_coefficients_and_odds_ratios.png")

# Print top insights
top_churn   = coef_df.nlargest(3,  'coefficient')
top_protect = coef_df.nsmallest(3, 'coefficient')

print(f"\n  Top 3 factors DRIVING churn:")
for _, row in top_churn.iterrows():
    print(f"    {row['feature']:<28} "
          f"coef={row['coefficient']:+.3f}  "
          f"odds ratio={row['odds_ratio']:.2f}x more likely")

print(f"\n  Top 3 factors PROTECTING against churn:")
for _, row in top_protect.iterrows():
    print(f"    {row['feature']:<28} "
          f"coef={row['coefficient']:+.3f}  "
          f"odds ratio={row['odds_ratio']:.2f}x less likely")

# =============================================================================
#  5. FINAL BUSINESS SUMMARY REPORT
# =============================================================================

header("5. FINAL BUSINESS SUMMARY")

n_high   = (y_prob >= 0.70).sum()
n_medium = ((y_prob >= 0.40) & (y_prob < 0.70)).sum()
n_low    = (y_prob <  0.40).sum()

top1 = coef_df.nlargest(1, 'coefficient').iloc[0]
top2 = coef_df.nlargest(2, 'coefficient').iloc[1]
top3 = coef_df.nlargest(3, 'coefficient').iloc[2]

summary = f"""
╔══════════════════════════════════════════════════════════════╗
║        PHASE 6 — FINAL EVALUATION SUMMARY                   ║
║        Model: Logistic Regression                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MODEL PERFORMANCE (held-out test set)                       ║
║  ─────────────────────────────────────────────────────────   ║
║  ROC-AUC          : {metrics['roc_auc']:.4f}   ← primary metric           ║
║  Accuracy         : {metrics['accuracy']:.4f}   ← misleading, ignore       ║
║  Precision        : {metrics['precision']:.4f}                              ║
║  Recall           : {metrics['recall']:.4f}                              ║
║  F1-Score         : {metrics['f1']:.4f}                              ║
║  Optimal threshold: {best_thresh:.3f}   (maximises F1)              ║
║                                                              ║
║  CONFUSION MATRIX                                            ║
║  ─────────────────────────────────────────────────────────   ║
║  Churners caught (TP)        : {TP:>4,}                         ║
║  Churners missed (FN)        : {FN:>4,}   ← priority to reduce  ║
║  False alarms    (FP)        : {FP:>4,}                         ║
║  Correct negatives (TN)      : {TN:>4,}                         ║
║                                                              ║
║  BUSINESS VALUE ESTIMATE (test set)                          ║
║  ─────────────────────────────────────────────────────────   ║
║  Revenue saved               : ${value_tp:>8,.0f}                 ║
║  Wasted offers               : ${cost_fp:>8,.0f}                 ║
║  Missed revenue              : ${cost_fn:>8,.0f}                 ║
║  Net model value             : ${net_value:>8,.0f}                 ║
║                                                              ║
║  RISK SEGMENTS (test set, {len(y_prob):,} customers)              ║
║  ─────────────────────────────────────────────────────────   ║
║  High risk   (prob >= 0.70)  : {n_high:>4,} customers             ║
║  Medium risk (0.40 to 0.70)  : {n_medium:>4,} customers             ║
║  Low risk    (prob <  0.40)  : {n_low:>4,} customers             ║
║                                                              ║
║  TOP CHURN DRIVERS (from coefficients)                       ║
║  ─────────────────────────────────────────────────────────   ║
║  1. {top1['feature']:<28} OR = {top1['odds_ratio']:.2f}x          ║
║  2. {top2['feature']:<28} OR = {top2['odds_ratio']:.2f}x          ║
║  3. {top3['feature']:<28} OR = {top3['odds_ratio']:.2f}x          ║
║                                                              ║
║  NEXT STEP → python phase7_score.py                          ║
╚══════════════════════════════════════════════════════════════╝
"""
print(summary)

with open('reports/phase6_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("  Saved → reports/phase6_summary.txt")

print("\n  Plots generated:")
for name in [
    '11_confusion_matrix_business.png',
    '12_roc_auc_curve.png',
    '13_precision_recall_threshold.png',
    '14_coefficients_and_odds_ratios.png',
]:
    print(f"    plots/{name}")

print(f"\n{'='*60}")
print("  PHASE 6 COMPLETE")
print(f"{'='*60}")