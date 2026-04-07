# =============================================================================
#  CHURN PREDICTION DASHBOARD — Streamlit App
#  Run locally : streamlit run app.py
#  Run via Docker: docker-compose up
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
#  PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title  = "Churn Prediction Dashboard",
    page_icon   = "📉",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================================
#  CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 4px;
    }
    .risk-high   { background:#fff0f0; border-left:4px solid #E05C5C;
                   padding:12px; border-radius:6px; margin:6px 0; }
    .risk-medium { background:#fff8f0; border-left:4px solid #F0A070;
                   padding:12px; border-radius:6px; margin:6px 0; }
    .risk-low    { background:#f0fff4; border-left:4px solid #5CB85C;
                   padding:12px; border-radius:6px; margin:6px 0; }
    .reason-tag {
        display:inline-block;
        background:#fee2e2;
        color:#991b1b;
        padding:3px 10px;
        border-radius:99px;
        font-size:0.78rem;
        font-weight:600;
        margin:2px;
    }
    .reason-tag-blue {
        background:#dbeafe;
        color:#1e40af;
    }
    .stAlert { border-radius: 10px; }
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  LOAD MODEL — cached so it only loads once
# =============================================================================

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "lr_model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def load_features():
    path = os.path.join("data", "features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

bundle       = load_model()
df_features  = load_features()

# =============================================================================
#  SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
st.sidebar.title("Churn Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 EDA Explorer",
     "🤖 Model Performance",
     "🔮 Live Prediction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("Model: Logistic Regression")
st.sidebar.markdown("Dataset: Telco Churn (7,043 customers)")
st.sidebar.markdown("Built with: Python + Streamlit")

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def get_churn_reasons(feature_values, coef_df, feature_cols, top_n=3):
    """
    For a single customer, multiply their feature values by the model
    coefficients to find which features contributed most to churn prediction.
    Returns top_n reasons pushing toward churn and top_n protecting.
    """
    contribs = {}
    for feat, val in zip(feature_cols, feature_values):
        row = coef_df[coef_df['feature'] == feat]
        if len(row) == 0:
            continue
        coef = row.iloc[0]['coefficient']
        contribs[feat] = coef * val   # contribution = coef × feature value

    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

    # Positive contributions = pushing toward churn
    churn_drivers   = [(f, v) for f, v in sorted_contribs if v > 0][:top_n]
    # Negative contributions = protecting against churn
    churn_protectors= [(f, v) for f, v in sorted_contribs if v < 0][:top_n]

    return churn_drivers, churn_protectors


FEATURE_LABELS = {
    'tenure'               : 'Tenure (months)',
    'monthly_charges'      : 'Monthly charges ($)',
    'total_charges'        : 'Total charges ($)',
    'charges_per_tenure'   : 'Charges per month of tenure',
    'contract_encoded'     : 'Contract type',
    'payment_encoded'      : 'Payment method',
    'internet_encoded'     : 'Internet service type',
    'support_services_count': 'Support services subscribed',
    'login_drop_pct'       : 'Login drop % (Jan→Mar)',
    'avg_monthly_logins'   : 'Avg monthly logins',
    'avg_data_gb'          : 'Avg data usage (GB)',
    'login_trend_slope'    : 'Login trend (slope)',
    'total_tickets'        : 'Total support tickets',
    'tickets_per_month'    : 'Tickets per month',
    'high_priority_pct'    : '% High/Critical tickets',
    'unresolved_pct'       : '% Unresolved tickets',
    'billing_pct'          : '% Billing tickets',
}


# =============================================================================
#  PAGE 1 — OVERVIEW
# =============================================================================

if page == "🏠 Overview":
    st.title("📉 Customer Churn Prediction")
    st.markdown("End-to-end ML project — Logistic Regression on Telco dataset")
    st.markdown("---")

    if df_features is None:
        st.error("features.csv not found. Run src/06featureEngineering.py first.")
        st.stop()

    # Top KPI metrics
    total      = len(df_features)
    churned    = df_features['churn'].sum()
    churn_rate = df_features['churn'].mean()

    if bundle:
        X = df_features[[c for c in bundle['feature_cols']
                          if c in df_features.columns]].copy()
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        probs = bundle['pipeline'].predict_proba(X)[:, 1]
        n_high   = (probs >= 0.70).sum()
        n_medium = ((probs >= 0.40) & (probs < 0.70)).sum()
        n_low    = (probs <  0.40).sum()
    else:
        n_high = n_medium = n_low = 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",  f"{total:,}")
    c2.metric("Churned",          f"{int(churned):,}")
    c3.metric("Churn Rate",       f"{churn_rate:.1%}")
    c4.metric("🔴 High Risk",     f"{n_high:,}")
    c5.metric("🟡 Medium Risk",   f"{n_medium:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df_features['churn'].value_counts().sort_index()
        ax.bar(['Stayed', 'Churned'], counts.values,
               color=['#4A90D9', '#E05C5C'], width=0.5, edgecolor='white')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 30, f'{v:,}', ha='center', fontweight='bold')
        ax.set_ylabel('Customers')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Risk level breakdown")
        if bundle:
            fig, ax = plt.subplots(figsize=(5, 3))
            risk_counts = [n_high, n_medium, n_low]
            risk_labels = ['High Risk', 'Medium Risk', 'Low Risk']
            risk_colors = ['#E05C5C', '#F0A070', '#5CB85C']
            wedges, texts, autotexts = ax.pie(
                risk_counts, labels=risk_labels, colors=risk_colors,
                autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2)
            )
            for at in autotexts:
                at.set_fontweight('bold')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run Phase 5 to see risk breakdown.")

    st.markdown("---")
    st.subheader("Project pipeline")
    steps = [
        ("01", "Data Upload",        "Raw CSV → PostgreSQL",           "✅"),
        ("02", "Ticket Generation",  "Synthetic support tickets",      "✅"),
        ("03", "Usage Generation",   "3 months usage logs",            "✅"),
        ("04", "SQL Exploration",    "JOINs, aggregations, insights",  "✅"),
        ("05", "Data Validation",    "Nulls, duplicates, integrity",   "✅"),
        ("06", "Feature Engineering","15 ML-ready features",           "✅"),
        ("07", "EDA",                "8 analytical plots",             "✅"),
        ("08", "Model Training",     "Logistic Regression + CV",       "✅"),
        ("09", "Evaluation",         "ROC-AUC, coefficients, costs",   "✅"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc, status) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:white;border-radius:10px;padding:14px;
                        margin:6px 0;box-shadow:0 2px 6px rgba(0,0,0,0.06)'>
                <div style='font-size:0.75rem;color:#888'>Phase {num}</div>
                <div style='font-weight:600;margin:2px 0'>{status} {title}</div>
                <div style='font-size:0.8rem;color:#555'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
#  PAGE 2 — EDA EXPLORER
# =============================================================================

elif page == "📊 EDA Explorer":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Visual insights from the Telco dataset")
    st.markdown("---")

    if df_features is None:
        st.error("features.csv not found. Run Phase 3 first.")
        st.stop()

    df = df_features.copy()
    contract_map = {0:'Month-to-month', 1:'One year', 2:'Two year'}
    internet_map = {0:'No internet',    1:'DSL',      2:'Fiber optic'}
    df['Contract'] = df['contract_encoded'].map(contract_map)
    df['Internet'] = df['internet_encoded'].map(internet_map)
    df['Churn Label'] = df['churn'].map({0:'Stayed', 1:'Churned'})

    tab1, tab2, tab3, tab4 = st.tabs([
        "Contract & Charges", "Tenure & Usage",
        "Support Tickets",    "Correlation"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn rate by contract type")
            contract_order = ['Month-to-month','One year','Two year']
            cb = (df.groupby('Contract')['churn']
                    .mean().reindex(contract_order) * 100)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#E05C5C','#F0A070','#4A90D9']
            bars = ax.bar(cb.index, cb.values,
                          color=colors, width=0.5, edgecolor='white')
            for bar, val in zip(bars, cb.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5,
                        f'{val:.1f}%', ha='center', fontweight='bold')
            ax.set_ylabel('Churn rate (%)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Monthly charges by churn status")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for label, color in [('Stayed','#4A90D9'),('Churned','#E05C5C')]:
                df[df['Churn Label']==label]['monthly_charges'].plot.kde(
                    ax=ax, label=label, color=color, linewidth=2)
            ax.set_xlabel('Monthly charges ($)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tenure distribution by churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for label, color in [('Stayed','#4A90D9'),('Churned','#E05C5C')]:
                df[df['Churn Label']==label]['tenure'].hist(
                    bins=25, alpha=0.6, label=label,
                    color=color, ax=ax, edgecolor='white')
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Customers')
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Login drop % by churn status")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            clean = df.dropna(subset=['login_drop_pct'])
            stayed  = clean[clean['Churn Label']=='Stayed']['login_drop_pct']
            churned = clean[clean['Churn Label']=='Churned']['login_drop_pct']
            ax.boxplot([stayed, churned], labels=['Stayed','Churned'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#E6F1FB'),
                       medianprops=dict(color='#E05C5C', linewidth=2))
            ax.set_ylabel('Login drop % (Jan→Mar)')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn rate by ticket volume")
            df['ticket_bucket'] = pd.cut(
                df['total_tickets'],
                bins=[-1,0,2,5,100],
                labels=['0 tickets','1-2','3-5','6+']
            )
            bucket_churn = (df.groupby('ticket_bucket', observed=True)['churn']
                              .mean() * 100)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#4A90D9','#7AB8E8','#F0A070','#E05C5C']
            bars = ax.bar(bucket_churn.index, bucket_churn.values,
                          color=colors, width=0.5, edgecolor='white')
            for bar, val in zip(bars, bucket_churn.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3,
                        f'{val:.1f}%', ha='center', fontweight='bold',
                        fontsize=9)
            ax.set_ylabel('Churn rate (%)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Churn rate by internet service")
            internet_order = ['No internet','DSL','Fiber optic']
            ci = (df.groupby('Internet')['churn']
                    .mean().reindex(internet_order) * 100)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#4A90D9','#7AB8E8','#E05C5C']
            bars = ax.bar(ci.index, ci.values,
                          color=colors, width=0.5, edgecolor='white')
            for bar, val in zip(bars, ci.values):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.3,
                        f'{val:.1f}%', ha='center', fontweight='bold')
            ax.set_ylabel('Churn rate (%)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Feature correlation matrix")
        num_cols = ['churn','tenure','monthly_charges','contract_encoded',
                    'login_drop_pct','avg_monthly_logins',
                    'total_tickets','high_priority_pct','billing_pct']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor='white', ax=ax,
                    annot_kws={'size':9})
        ax.set_title('Correlation matrix — focus on the churn row')
        st.pyplot(fig); plt.close()


# =============================================================================
#  PAGE 3 — MODEL PERFORMANCE
# =============================================================================

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("Logistic Regression — evaluation on held-out test set")
    st.markdown("---")

    if bundle is None:
        st.error("Model not found. Run src/08TrainModel.py first.")
        st.stop()

    metrics  = bundle['metrics']
    coef_df  = bundle['coef_df']
    y_test   = bundle['y_test']
    y_pred   = bundle['y_pred']
    y_prob   = bundle['y_prob']

    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}",  "Primary metric")
    c2.metric("Recall",    f"{metrics['recall']:.4f}",   "Churners caught")
    c3.metric("Precision", f"{metrics['precision']:.4f}")
    c4.metric("F1-Score",  f"{metrics['f1']:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC-AUC curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#4A90D9', linewidth=2.5,
                label=f'Logistic Regression (AUC={auc:.4f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color='#4A90D9')
        ax.plot([0,1],[0,1], color='#888', linestyle='--',
                label='Random baseline (0.50)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Confusion matrix")
        from sklearn.metrics import ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Stayed','Churned'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        TN,FP,FN,TP = cm.ravel()
        labels = [(0,0,'True Negative\n(Correct)'   ,'#4A90D9'),
                  (0,1,'False Positive\n(Wrong alarm)','#F0A070'),
                  (1,0,'False Negative\n(Missed!)'   ,'#E05C5C'),
                  (1,1,'True Positive\n(Caught!)'    ,'#5CB85C')]
        for col_i, row_i, lbl, clr in labels:
            ax.text(col_i, row_i+0.38, lbl,
                    ha='center', fontsize=7, color=clr, fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Coefficient analysis — what drives churn")
    st.markdown("**Red bars** push toward churn · **Blue bars** protect against churn")

    plot_df = coef_df.sort_values('coefficient', ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    colors_c = ['#E05C5C' if c > 0 else '#4A90D9'
                for c in plot_df['coefficient']]
    axes[0].barh(plot_df['feature'], plot_df['coefficient'],
                 color=colors_c, edgecolor='white', height=0.6)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel('Coefficient (log-odds)')
    axes[0].set_title('Raw coefficients')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    colors_o = ['#E05C5C' if o > 1 else '#4A90D9'
                for o in plot_df['odds_ratio']]
    axes[1].barh(plot_df['feature'], plot_df['odds_ratio'],
                 color=colors_o, edgecolor='white', height=0.6)
    axes[1].axvline(1, color='black', linewidth=0.8)
    axes[1].set_xlabel('Odds ratio  exp(coefficient)')
    axes[1].set_title('Odds ratios — easier to explain')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    for bar, val in zip(axes[1].patches, plot_df['odds_ratio']):
        axes[1].text(val+0.01, bar.get_y()+bar.get_height()/2,
                     f'{val:.2f}x', va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Business cost estimate")
    bc1, bc2, bc3 = st.columns(3)
    REVENUE = 65; OFFER = 20; MONTHS = 6
    value_tp  = TP * (REVENUE * MONTHS - OFFER)
    cost_fp   = FP * OFFER
    cost_fn   = FN * REVENUE * MONTHS
    net       = value_tp - cost_fp - cost_fn
    bc1.metric("Revenue saved (TP)",  f"${value_tp:,.0f}")
    bc2.metric("Missed revenue (FN)", f"-${cost_fn:,.0f}")
    bc3.metric("Net model value",     f"${net:,.0f}",
               delta="positive" if net > 0 else "negative")


# =============================================================================
#  PAGE 4 — LIVE PREDICTION
# =============================================================================

elif page == "🔮 Live Prediction":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Enter a customer's details to get their churn probability "
                "and the exact reasons why they are at risk.")
    st.markdown("---")

    if bundle is None:
        st.error("Model not found. Run src/08TrainModel.py first.")
        st.stop()

    pipeline     = bundle['pipeline']
    feature_cols = bundle['feature_cols']
    coef_df      = bundle['coef_df']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer profile")
        tenure           = st.slider("Tenure (months)",       0, 72, 12)
        monthly_charges  = st.slider("Monthly charges ($)",   18, 120, 65)
        contract         = st.selectbox("Contract type",
                            ["Month-to-month","One year","Two year"])
        payment          = st.selectbox("Payment method",
                            ["Electronic check","Mailed check",
                             "Bank transfer (automatic)",
                             "Credit card (automatic)"])

    with col2:
        st.subheader("Internet and support")
        internet         = st.selectbox("Internet service",
                            ["No internet","DSL","Fiber optic"])
        tech_support     = st.checkbox("Has Tech Support",    value=False)
        online_security  = st.checkbox("Has Online Security", value=False)
        total_tickets    = st.slider("Total support tickets", 0, 15, 0)
        high_priority_pct= st.slider("% High priority tickets", 0.0, 1.0, 0.0)

    with col3:
        st.subheader("Usage (last 3 months)")
        jan_logins = st.slider("Jan logins", 0, 60, 20)
        mar_logins = st.slider("Mar logins", 0, 60, 20)
        avg_data   = st.slider("Avg data usage (GB)", 0, 500, 200)
        unresolved_pct = st.slider("% Unresolved tickets", 0.0, 1.0, 0.0)
        billing_pct    = st.slider("% Billing tickets",    0.0, 1.0, 0.0)

    st.markdown("---")

    # Encode inputs to match training features
    contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
    payment_map  = {"Electronic check":0, "Mailed check":1,
                    "Bank transfer (automatic)":2,
                    "Credit card (automatic)":3}
    internet_map = {"No internet":0, "DSL":1, "Fiber optic":2}

    contract_enc  = contract_map[contract]
    payment_enc   = payment_map[payment]
    internet_enc  = internet_map[internet]
    support_count = int(tech_support) + int(online_security)
    total_charges = monthly_charges * tenure
    charges_per_t = total_charges / tenure if tenure > 0 else monthly_charges
    login_drop    = ((jan_logins - mar_logins) / jan_logins * 100
                     if jan_logins > 0 else 0)
    avg_logins    = (jan_logins + mar_logins) / 2
    trend_slope   = (mar_logins - jan_logins) / 2
    tickets_pm    = total_tickets / 3

    # Build feature vector in same order as training
    feature_values_map = {
        'tenure'               : tenure,
        'monthly_charges'      : monthly_charges,
        'total_charges'        : total_charges,
        'charges_per_tenure'   : charges_per_t,
        'contract_encoded'     : contract_enc,
        'payment_encoded'      : payment_enc,
        'internet_encoded'     : internet_enc,
        'support_services_count': support_count,
        'login_drop_pct'       : login_drop,
        'avg_monthly_logins'   : avg_logins,
        'avg_data_gb'          : avg_data,
        'login_trend_slope'    : trend_slope,
        'total_tickets'        : total_tickets,
        'tickets_per_month'    : tickets_pm,
        'high_priority_pct'    : high_priority_pct,
        'unresolved_pct'       : unresolved_pct,
        'billing_pct'          : billing_pct,
    }

    feature_values = [feature_values_map.get(f, 0) for f in feature_cols]
    X_input = pd.DataFrame([feature_values], columns=feature_cols)

    # Predict
    churn_prob  = pipeline.predict_proba(X_input)[0][1]
    churn_pred  = pipeline.predict(X_input)[0]

    # Risk level
    if churn_prob >= 0.70:
        risk = "HIGH"
        risk_color = "#E05C5C"
        risk_bg    = "#fff0f0"
        risk_emoji = "🔴"
    elif churn_prob >= 0.40:
        risk = "MEDIUM"
        risk_color = "#F0A070"
        risk_bg    = "#fff8f0"
        risk_emoji = "🟡"
    else:
        risk = "LOW"
        risk_color = "#5CB85C"
        risk_bg    = "#f0fff4"
        risk_emoji = "🟢"

    # Display prediction result
    st.markdown(f"""
    <div style='background:{risk_bg};border:2px solid {risk_color};
                border-radius:14px;padding:24px;text-align:center;
                margin-bottom:20px'>
        <div style='font-size:1rem;color:#555;margin-bottom:6px'>
            Churn probability
        </div>
        <div style='font-size:3.5rem;font-weight:800;color:{risk_color}'>
            {churn_prob:.1%}
        </div>
        <div style='font-size:1.4rem;font-weight:600;color:{risk_color}'>
            {risk_emoji} {risk} RISK
        </div>
        <div style='font-size:0.9rem;color:#777;margin-top:8px'>
            {'This customer is likely to cancel their subscription'
             if churn_pred == 1
             else 'This customer is likely to stay'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge bar
    st.markdown("**Churn probability gauge**")
    fig, ax = plt.subplots(figsize=(8, 0.8))
    ax.barh([0], [1],   color='#eee',     height=0.5)
    ax.barh([0], [0.4], color='#5CB85C',  height=0.5)
    ax.barh([0], [0.3], color='#F0A070',  height=0.5, left=0.4)
    ax.barh([0], [0.3], color='#E05C5C',  height=0.5, left=0.7)
    ax.barh([0], [churn_prob], color='black', height=0.08, left=0)
    ax.axvline(churn_prob, color='black', linewidth=3)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.4, 0.7, 1.0])
    ax.set_xticklabels(['0%', '40%\n(Low→Med)', '70%\n(Med→High)', '100%'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── CHURN REASONS — the key feature you requested ──────────────────────
    churn_drivers, churn_protectors = get_churn_reasons(
        feature_values, coef_df, feature_cols, top_n=4
    )

    col_reason1, col_reason2 = st.columns(2)

    with col_reason1:
        if churn_pred == 1 and churn_drivers:
            st.markdown("### ⚠️ Why this customer may churn")
            st.markdown("These features are **pushing toward churn:**")
            for feat, contrib in churn_drivers:
                label     = FEATURE_LABELS.get(feat, feat)
                raw_val   = feature_values_map.get(feat, 0)
                intensity = abs(contrib)
                bar_width = min(int(intensity * 80), 100)
                st.markdown(f"""
                <div class='risk-high'>
                    <div style='font-weight:600;color:#991b1b'>{label}</div>
                    <div style='font-size:0.82rem;color:#555;margin:3px 0'>
                        Value: <b>{raw_val:.2f}</b> &nbsp;·&nbsp;
                        Churn contribution: <b>+{contrib:.3f}</b>
                    </div>
                    <div style='background:#fee2e2;border-radius:4px;
                                height:6px;margin-top:5px'>
                        <div style='background:#E05C5C;height:6px;
                                    border-radius:4px;
                                    width:{bar_width}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### ✅ No strong churn signals detected")
            st.markdown("This customer does not show high-risk patterns.")

    with col_reason2:
        if churn_protectors:
            st.markdown("### 🛡️ What is protecting this customer")
            st.markdown("These features are **reducing churn risk:**")
            for feat, contrib in churn_protectors:
                label     = FEATURE_LABELS.get(feat, feat)
                raw_val   = feature_values_map.get(feat, 0)
                intensity = abs(contrib)
                bar_width = min(int(intensity * 80), 100)
                st.markdown(f"""
                <div class='risk-low'>
                    <div style='font-weight:600;color:#166534'>{label}</div>
                    <div style='font-size:0.82rem;color:#555;margin:3px 0'>
                        Value: <b>{raw_val:.2f}</b> &nbsp;·&nbsp;
                        Protection contribution: <b>{contrib:.3f}</b>
                    </div>
                    <div style='background:#dcfce7;border-radius:4px;
                                height:6px;margin-top:5px'>
                        <div style='background:#5CB85C;height:6px;
                                    border-radius:4px;
                                    width:{bar_width}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All feature values for this customer")
    input_display = pd.DataFrame({
        'Feature'    : [FEATURE_LABELS.get(f, f) for f in feature_cols],
        'Value'      : [f'{v:.3f}' for v in feature_values],
        'Coefficient': [f'{coef_df[coef_df.feature==f].coefficient.values[0]:+.3f}'
                        if len(coef_df[coef_df.feature==f]) > 0 else '—'
                        for f in feature_cols],
        'Contribution': [f'{feature_values_map.get(f,0) * coef_df[coef_df.feature==f].coefficient.values[0]:+.3f}'
                         if len(coef_df[coef_df.feature==f]) > 0 else '—'
                         for f in feature_cols],
    })
    st.dataframe(input_display, use_container_width=True, hide_index=True)