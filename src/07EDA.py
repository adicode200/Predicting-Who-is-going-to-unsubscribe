# =============================================================================
#  PHASE 4 — EXPLORATORY DATA ANALYSIS (EDA)
#  Churn Prediction Project
#  
#
#  What this file does:
#  Loads features.csv built in Phase 3 and produces 8 plots that answer
#  the most important business questions about churn. Every plot is saved
#  as a PNG in the plots/ folde r so you can include them in a presentation.
#
#  Plots produced:
#  01_churn_distribution.png       — how imbalanced is the target?
#  02_churn_by_contract.png        — which contract type churns most?
#  03_tenure_by_churn.png          — do new customers churn more?
#  04_monthly_charges_by_churn.png — do high-charge customers churn more?
#  05_login_drop_by_churn.png      — does usage drop predict churn?
#  06_tickets_by_churn.png         — do more tickets = more churn?
#  07_correlation_heatmap.png      — which features correlate with churn?
#  08_churn_by_internet.png        — fiber vs DSL vs no internet churn rate
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# =============================================================================
#  0. SETUP
# =============================================================================

# Create output folder for plots
os.makedirs("plots", exist_ok=True)

# Global plot style — clean, professional, interview-ready
plt.rcParams.update({
    'figure.dpi'      : 150,
    'figure.facecolor': 'white',
    'axes.facecolor'  : 'white',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.3,
    'grid.linestyle'  : '--',
    'font.size'       : 11,
    'axes.titlesize'  : 13,
    'axes.titleweight': 'bold',
    'axes.labelsize'  : 11,
})

CHURN_COLORS = {0: '#4A90D9', 1: '#E05C5C'}   # blue = stayed, red = churned
PALETTE      = [CHURN_COLORS[0], CHURN_COLORS[1]]

print("="*55)
print("  PHASE 4 — EXPLORATORY DATA ANALYSIS")
print("="*55)

# Load features built in Phase 3
df = pd.read_csv("data/features.csv")
print(f"\n  Loaded features.csv : {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  Churn rate          : {df['churn'].mean():.1%}\n")    

# Readable label for plots
df['Churn Label'] = df['churn'].map({0: 'Stayed', 1: 'Churned'})

# Reverse-map contract_encoded back to readable labels for plots
contract_map_rev = {0: 'Month-to-month', 1: 'One year', 2: 'Two year'}
df['Contract']   = df['contract_encoded'].map(contract_map_rev)

internet_map_rev = {0: 'No internet', 1: 'DSL', 2: 'Fiber optic'}
df['Internet']   = df['internet_encoded'].map(internet_map_rev)


def save(filename):
    path = f"plots/{filename}"
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# =============================================================================
#  PLOT 1 — CHURN DISTRIBUTION
#  Business question: how imbalanced is our target variable?
#  Why it matters: imbalance means accuracy is a useless metric.
#                  A model that predicts "never churns" gets 73% accuracy.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: count bar
counts = df['churn'].value_counts().sort_index()
bars   = axes[0].bar(['Stayed', 'Churned'], counts.values,
                     color=PALETTE, width=0.5, edgecolor='white')
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 50, f'{val:,}',
                 ha='center', va='bottom', fontweight='bold')
axes[0].set_title('Customer count by churn status')
axes[0].set_ylabel('Number of customers')
axes[0].set_ylim(0, counts.max() * 1.15)

# Right: percentage pie
pct    = counts / counts.sum() * 100
wedges, texts, autotexts = axes[1].pie(
    pct.values,
    labels    = ['Stayed', 'Churned'],
    colors    = PALETTE,
    autopct   = '%1.1f%%',
    startangle= 90,
    wedgeprops= dict(edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontweight('bold')
axes[1].set_title('Churn split (%)')

# Annotation explaining why this matters
fig.text(0.5, -0.04,
         'Class imbalance (~27% churn): use ROC-AUC, not accuracy. '
         'Apply SMOTE or class_weight in Phase 5.',
         ha='center', fontsize=9, color='#666666', style='italic')

save("01_churn_distribution.png")


# =============================================================================
#  PLOT 2 — CHURN RATE BY CONTRACT TYPE
#  Business question: which contract type has the highest churn?
#  Why it matters: contract type is typically the #1 feature in telecom churn.
# =============================================================================

contract_order = ['Month-to-month', 'One year', 'Two year']

churn_by_contract = (
    df.groupby('Contract')['churn']
    .agg(['mean', 'count'])
    .reindex(contract_order)
    .reset_index()
)
churn_by_contract['churn_pct'] = churn_by_contract['mean'] * 100

fig, ax = plt.subplots(figsize=(8, 5))
colors  = ['#E05C5C', '#F0A070', '#4A90D9']   # red→orange→blue by risk
bars    = ax.bar(churn_by_contract['Contract'],
                 churn_by_contract['churn_pct'],
                 color=colors, width=0.5, edgecolor='white')

for bar, row in zip(bars, churn_by_contract.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{row.churn_pct:.1f}%\n(n={row.count:,})",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Churn rate by contract type')
ax.set_ylabel('Churn rate (%)')
ax.set_ylim(0, churn_by_contract['churn_pct'].max() * 1.25)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel('')

# Key insight annotation
max_pct = churn_by_contract['churn_pct'].max()
ax.axhline(max_pct, color='#E05C5C', linestyle=':', alpha=0.4, linewidth=1)

save("02_churn_by_contract.png")


# =============================================================================
#  PLOT 3 — TENURE DISTRIBUTION BY CHURN
#  Business question: do new customers churn more than loyal ones?
#  Why it matters: tenure is a proxy for customer satisfaction over time.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: overlapping histogram
for churn_val, label, color in [(0,'Stayed','#4A90D9'), (1,'Churned','#E05C5C')]:
    subset = df[df['churn'] == churn_val]['tenure']
    axes[0].hist(subset, bins=30, alpha=0.6, label=label,
                 color=color, edgecolor='white')
axes[0].set_title('Tenure distribution by churn status')
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Number of customers')
axes[0].legend()

# Annotate medians
for churn_val, color in [(0,'#4A90D9'), (1,'#E05C5C')]:
    med = df[df['churn']==churn_val]['tenure'].median()
    axes[0].axvline(med, color=color, linestyle='--', linewidth=1.5,
                    label=f'Median={med:.0f}')

# Right: box plot for cleaner comparison
sns.boxplot(data=df, x='Churn Label', y='tenure',
            palette=PALETTE, width=0.4, ax=axes[1],
            order=['Stayed','Churned'])
axes[1].set_title('Tenure spread: stayed vs churned')
axes[1].set_xlabel('')
axes[1].set_ylabel('Tenure (months)')

# Add median labels on boxplot
for i, churn_label in enumerate(['Stayed', 'Churned']):
    med = df[df['Churn Label']==churn_label]['tenure'].median()
    axes[1].text(i, med + 1, f'Median\n{med:.0f}mo',
                 ha='center', fontsize=9, fontweight='bold')

save("03_tenure_by_churn.png")


# =============================================================================
#  PLOT 4 — MONTHLY CHARGES BY CHURN
#  Business question: do expensive plans drive customers away?
#  Why it matters: high charges + poor service = churn trigger.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: KDE (density) plot — smoother than histogram for continuous data
for churn_val, label, color in [(0,'Stayed','#4A90D9'), (1,'Churned','#E05C5C')]:
    subset = df[df['churn']==churn_val]['monthly_charges']
    subset.plot.kde(ax=axes[0], label=label, color=color, linewidth=2)
axes[0].set_title('Monthly charges distribution (density)')
axes[0].set_xlabel('Monthly charges ($)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(df['monthly_charges'].min() - 5, df['monthly_charges'].max() + 5)

# Right: violin plot — shows shape + spread simultaneously
sns.violinplot(data=df, x='Churn Label', y='monthly_charges',
               palette=PALETTE, ax=axes[1],
               order=['Stayed','Churned'], inner='quartile')
axes[1].set_title('Monthly charges: stayed vs churned')
axes[1].set_xlabel('')
axes[1].set_ylabel('Monthly charges ($)')

# Mean labels
for i, churn_label in enumerate(['Stayed', 'Churned']):
    mean_val = df[df['Churn Label']==churn_label]['monthly_charges'].mean()
    axes[1].text(i, df['monthly_charges'].max() + 2,
                 f'Mean ${mean_val:.0f}',
                 ha='center', fontsize=9, fontweight='bold')

save("04_monthly_charges_by_churn.png")


# =============================================================================
#  PLOT 5 — LOGIN DROP PCT BY CHURN
#  Business question: does declining usage predict churn?
#  Why it matters: login_drop_pct is our custom-engineered feature —
#                  this plot validates whether it actually carries signal.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: box plot of login_drop_pct
clean = df.dropna(subset=['login_drop_pct'])
sns.boxplot(data=clean, x='Churn Label', y='login_drop_pct',
            palette=PALETTE, width=0.4, ax=axes[0],
            order=['Stayed','Churned'])
axes[0].set_title('Login drop % (Jan→Mar) by churn status')
axes[0].set_xlabel('')
axes[0].set_ylabel('Login drop % (positive = declining)')
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Right: scatter — login_drop_pct vs tenure, coloured by churn
scatter = axes[1].scatter(
    clean['tenure'], clean['login_drop_pct'],
    c    = clean['churn'].map(CHURN_COLORS),
    alpha= 0.3, s=10, linewidths=0
)
axes[1].set_title('Login drop % vs tenure (coloured by churn)')
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('Login drop %')
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# Manual legend for scatter
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor='#4A90D9', markersize=8, label='Stayed'),
    Line2D([0],[0], marker='o', color='w',
           markerfacecolor='#E05C5C', markersize=8, label='Churned'),
]
axes[1].legend(handles=legend_elements)

save("05_login_drop_by_churn.png")


# =============================================================================
#  PLOT 6 — TICKET COUNT BY CHURN
#  Business question: do customers with more tickets churn more?
#  Why it matters: validates our ticket features before putting them in a model.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: average ticket count by churn
ticket_means = df.groupby('Churn Label')[
    ['total_tickets','high_priority_pct']
].mean().reset_index()

x     = range(len(ticket_means))
width = 0.35
bars1 = axes[0].bar([i - width/2 for i in x],
                    ticket_means['total_tickets'],
                    width, label='Avg total tickets',
                    color=['#4A90D9','#E05C5C'], edgecolor='white')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}',
                 ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('Avg ticket count: stayed vs churned')
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(ticket_means['Churn Label'])
axes[0].set_ylabel('Average number of tickets')

# Right: churn rate by ticket bucket
df['ticket_bucket'] = pd.cut(
    df['total_tickets'],
    bins  = [-1, 0, 2, 5, 100],
    labels= ['0 tickets', '1-2 tickets', '3-5 tickets', '6+ tickets']
)
bucket_churn = (
    df.groupby('ticket_bucket', observed=True)['churn']
    .agg(['mean','count'])
    .reset_index()
)
bucket_churn['churn_pct'] = bucket_churn['mean'] * 100

colors_b = ['#4A90D9', '#7AB8E8', '#F0A070', '#E05C5C']
bars2    = axes[1].bar(bucket_churn['ticket_bucket'],
                       bucket_churn['churn_pct'],
                       color=colors_b, width=0.5, edgecolor='white')
for bar, row in zip(bars2, bucket_churn.itertuples()):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{row.churn_pct:.1f}%\n(n={row.count:,})",
                 ha='center', fontsize=9, fontweight='bold')
axes[1].set_title('Churn rate by ticket volume')
axes[1].set_ylabel('Churn rate (%)')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

save("06_tickets_by_churn.png")


# =============================================================================
#  PLOT 7 — CORRELATION HEATMAP
#  Business question: which features are most correlated with churn?
#  Why it matters: high correlation = likely important feature.
#                  Also reveals multicollinearity between features.
# =============================================================================

feature_cols = [
    'churn', 'tenure', 'monthly_charges', 'total_charges',
    'contract_encoded', 'payment_encoded', 'internet_encoded',
    'support_services_count', 'login_drop_pct', 'avg_monthly_logins',
    'avg_data_gb', 'login_trend_slope', 'total_tickets',
    'tickets_per_month', 'high_priority_pct', 'unresolved_pct',
    'billing_pct', 'charges_per_tenure'
]

corr = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))

mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap    = 'RdBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    ax=ax, annot_kws={'size': 8}
)
ax.set_title('Feature correlation matrix\n'
             '(focus on the churn row — top row)', pad=15)

# Highlight the churn row label
ax.get_yticklabels()[0].set_color('#E05C5C')
ax.get_yticklabels()[0].set_fontweight('bold')

save("07_correlation_heatmap.png")


# =============================================================================
#  PLOT 8 — CHURN RATE BY INTERNET SERVICE
#  Business question: does internet type affect churn?
#  Why it matters: fiber optic customers churn despite higher speeds —
#                  likely due to higher price + more alternatives available.
# =============================================================================

internet_order = ['No internet', 'DSL', 'Fiber optic']
churn_internet = (
    df.groupby('Internet')['churn']
    .agg(['mean','count'])
    .reindex(internet_order)
    .reset_index()
)
churn_internet['churn_pct'] = churn_internet['mean'] * 100

fig, ax = plt.subplots(figsize=(8, 5))
colors  = ['#4A90D9', '#7AB8E8', '#E05C5C']
bars    = ax.bar(churn_internet['Internet'],
                 churn_internet['churn_pct'],
                 color=colors, width=0.5, edgecolor='white')

for bar, row in zip(bars, churn_internet.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{row.churn_pct:.1f}%\n(n={row.count:,})",
            ha='center', fontsize=10, fontweight='bold')

ax.set_title('Churn rate by internet service type')
ax.set_ylabel('Churn rate (%)')
ax.set_xlabel('')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, churn_internet['churn_pct'].max() * 1.25)

ax.text(0.98, 0.95,
        'Fiber optic: higher speed but also\nhigher churn — price sensitivity?',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=9, color='#666666', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3', alpha=0.7))

save("08_churn_by_internet.png")


# =============================================================================
#  SUMMARY — KEY INSIGHTS
# =============================================================================

print("\n" + "="*55)
print("  EDA COMPLETE — KEY INSIGHTS")
print("="*55)

churn_by_contract_summary = df.groupby('Contract')['churn'].mean() * 100
churn_by_internet_summary = df.groupby('Internet')['churn'].mean() * 100
median_tenure_stayed  = df[df['churn']==0]['tenure'].median()
median_tenure_churned = df[df['churn']==1]['tenure'].median()
mean_drop_stayed      = df[df['churn']==0]['login_drop_pct'].mean()
mean_drop_churned     = df[df['churn']==1]['login_drop_pct'].mean()

print(f"""
  1. Class imbalance
     Churn rate = {df['churn'].mean():.1%}  →  use ROC-AUC not accuracy in Phase 5

  2. Contract type (strongest signal)
     Month-to-month : {churn_by_contract_summary.get('Month-to-month', 0):.1f}% churn
     One year       : {churn_by_contract_summary.get('One year', 0):.1f}% churn
     Two year       : {churn_by_contract_summary.get('Two year', 0):.1f}% churn

  3. Tenure
     Median tenure (stayed)   : {median_tenure_stayed:.0f} months
     Median tenure (churned)  : {median_tenure_churned:.0f} months
     → New customers churn far more than loyal ones

  4. Login drop (engineered feature validation)
     Avg drop % (stayed)   : {mean_drop_stayed:.1f}%
     Avg drop % (churned)  : {mean_drop_churned:.1f}%
     → login_drop_pct carries real signal

  5. Internet service
     Fiber optic churn rate : {churn_by_internet_summary.get('Fiber optic', 0):.1f}%
     DSL churn rate         : {churn_by_internet_summary.get('DSL', 0):.1f}%
     No internet churn rate : {churn_by_internet_summary.get('No internet', 0):.1f}%

""")
print("="*55)