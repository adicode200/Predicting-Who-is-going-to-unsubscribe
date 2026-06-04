 <div align="center">

<img src="https://img.icons8.com/fluency/96/combo-chart.png" width="80"/>

# 📉 Predicting Who Is Going To Unsubscribe

**An end-to-end AI system that predicts customer churn and explains exactly why**

[![Live Demo](https://predicting-who-is-going-to-unsubscribe-dbr5zbksgvvm6rigmaampd.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-adicode200-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/adicode200/Predicting-Who-is-going-to-unsubscribe)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 🔴 The Problem

Every month, telecom companies lose customers silently.

A customer does not cancel in one day. They stop logging in. They raise billing complaints. Their usage drops. Then one morning they are gone — and the business finds out in next month's report.

**By then it is too late.**

Acquiring a new customer costs **5 to 10 times more** than retaining an existing one. The real cost of churn is not just the lost subscription — it is the marketing spend, the onboarding cost, and the lifetime value that never gets realised.

The problem is not that businesses do not care about churn. The problem is they have no system that tells them **who is about to leave, and why, before it happens.**

---

## 🟢 The Solution

This system watches three signals simultaneously — how a customer's usage is trending, how many complaints they are raising, and what kind of contract they are on — and produces a single churn probability score for every customer.

But a probability score alone is not enough. A business manager cannot act on "this customer has a 78% churn probability." They need to know **why.**

This system tells them:

> *"This customer's login activity dropped 48% over the last 3 months, they have raised 2 unresolved billing tickets, and they are on a month-to-month contract. These three factors make them 2.3× more likely to leave than the average customer."*

That is actionable. A retention offer can be sent today, before the customer cancels.

**Live demo:** [telco-churn-insight.streamlit.app](https://predicting-who-is-going-to-unsubscribe-dbr5zbksgvvm6rigmaampd.streamlit.app/)

---

## 📊 Results

| Metric | Score | Plain English |
|--------|-------|---------------|
| ROC-AUC | **0.8269** | Model correctly ranks churners above non-churners 83% of the time |
| Recall | **0.7674** | 76.7% of customers who were going to leave were caught |
| Precision | **0.5026** | 1 in 2 flagged customers actually churned |
| F1-Score | **0.6074** | Balanced score between catching churners and avoiding false alarms |
| Estimated business value | **$66,580** | Net recoverable revenue on the test set alone |

> **Why not accuracy?** A model that predicts nobody churns gets 73% accuracy but catches zero customers. ROC-AUC and Recall are the metrics that matter for this problem.

---

## 🏗️ How It Works — Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        RAW DATA                             │
│         7,043 customers · 3 tables · PostgreSQL             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA VALIDATION                           │
│         15 automated checks before any modelling            │
│    nulls · duplicates · referential integrity · outliers    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING                          │
│      15 features built from 3 raw tables from scratch       │
│   login drop % · ticket frequency · contract encoding ...   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING                             │
│    Logistic Regression · StandardScaler · class_weight      │
│         5-fold cross-validation · threshold tuning          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION                               │
│    ROC-AUC · Precision-Recall · Coefficient analysis        │
│              Business value quantification                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               STREAMLIT DASHBOARD                           │
│    Live prediction · Churn reasons · EDA · Model metrics    │
│              Containerised with Docker                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── app.py                        # Streamlit dashboard — 4 pages
├── Dockerfile                    # Container configuration
├── docker-compose.yml            # PostgreSQL + Streamlit together
├── requirements.txt              # All dependencies
├── .env                          # DB credentials (not in git)
├── .gitignore
├── README.md
│
├── src/                          # Pipeline — run in order
│   ├── 01oad_data.py             # Upload raw CSV to PostgreSQL
│   ├── 02generate_ticket.py      # Generate synthetic support tickets
│   ├── 03generate_usage.py       # Generate 3-month usage logs
│   ├── 04exploration.py          # SQL exploration and joins
│   ├── 05validation.py           # 15 automated data quality checks
│   ├── 06featureEngineering.py   # Build all 15 ML features
│   ├── 07EDA.py                  # Generate 8 analytical plots
│   ├── 08TrainModel.py           # Train model and cross-validate
│   └── 09modelEvaluation.py      # Evaluate, interpret, save report
│
├── data/
│   ├── customer_data.csv         # Raw IBM Telco dataset (7,043 rows)
│   └── features.csv              # Engineered ML-ready features
│
├── models/
│   └── lr_model.pkl              # Trained model + all metadata
│
├── plots/                        # All generated visualisations (14 PNGs)
└── reports/                      # Phase 6 evaluation summary
```

---

## 🗄️ Database Schema

Three tables in PostgreSQL — joined together for analysis:

```
raw_customers (7,043 rows)
├── customerID       — unique identifier
├── tenure           — months as a customer
├── Contract         — Month-to-month / One year / Two year
├── MonthlyCharges   — current monthly bill
├── TotalCharges     — total spend to date
└── Churn            — Yes / No  ← TARGET VARIABLE

support_tickets (5,000 rows)
├── ticket_id        — unique ticket identifier
├── customerID       — links to raw_customers
├── category         — Technical / Billing / Outage / General
├── priority         — Low / Medium / High / Critical
├── status           — Open / Pending / Resolved / Closed
└── created_at       — ticket date

usage_logs (21,129 rows)
├── customerID       — links to raw_customers
├── month_date       — Jan / Feb / Mar 2026
├── login_count      — number of logins that month
└── data_usage_gb    — data consumed that month
```

**SQL used to join them:**
```sql
WITH ticket_summary AS (
    SELECT customerID,
           COUNT(*) AS total_tickets,
           SUM(CASE WHEN priority IN ('High','Critical') THEN 1 ELSE 0 END)
               AS high_pri_tickets
    FROM support_tickets
    GROUP BY customerID
),
usage_summary AS (
    SELECT customerID,
           MAX(CASE WHEN month_date='2026-01-01' THEN login_count END) AS jan_logins,
           MAX(CASE WHEN month_date='2026-03-01' THEN login_count END) AS mar_logins
    FROM usage_logs
    GROUP BY customerID
)
SELECT c.*, t.total_tickets, u.jan_logins, u.mar_logins
FROM raw_customers c
LEFT JOIN ticket_summary t USING (customerID)
LEFT JOIN usage_summary  u USING (customerID)
```

---

## 🔧 Feature Engineering — All 15 Features

None of these 15 features exist in the original CSV. Every single one was built from scratch:

### From usage_logs — Usage behaviour signals
| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `login_drop_pct` | (Jan − Mar) / Jan × 100 | Declining engagement before cancellation |
| `avg_monthly_logins` | Mean of Jan + Feb + Mar | Overall engagement level |
| `login_trend_slope` | np.polyfit across 3 months | Consistent decline vs one-time drop |
| `avg_data_gb` | Mean data usage across 3 months | Service utilisation level |

### From support_tickets — Complaint pattern signals
| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `total_tickets` | COUNT per customer | Raw complaint volume |
| `tickets_per_month` | Total tickets / 3 | Normalised complaint frequency |
| `high_priority_pct` | High+Critical / total | Severity of issues |
| `unresolved_pct` | Open+Pending / total | Ongoing frustration level |
| `billing_pct` | Billing tickets / total | Financial dissatisfaction signal |

### From raw_customers — Profile signals
| Feature | Encoding | Why it matters |
|---------|----------|----------------|
| `tenure` | Numeric (months) | Loyal customers churn far less |
| `monthly_charges` | Numeric ($) | Higher charges increase sensitivity |
| `total_charges` | Numeric ($) | Total lifetime value |
| `contract_encoded` | 0=Month-to-month → 2=Two year | Strongest single predictor |
| `support_services_count` | TechSupport + OnlineSecurity | Embedded customers are stickier |
| `charges_per_tenure` | TotalCharges / tenure | Detects pricing changes over time |

---

## ✅ Data Validation — 15 Automated Checks

Dirty data produces confident wrong answers. These checks run before any modelling:

```
Null checks
  ✅ No nulls in customerID, Churn, tenure, MonthlyCharges
  ✅ No nulls in support_tickets
  ✅ No nulls in usage_logs

Duplicate checks
  ✅ No duplicate customerIDs in raw_customers
  ✅ No duplicate ticket_ids in support_tickets
  ✅ Exactly 3 usage months per customer

Referential integrity
  ✅ No orphan IDs in support_tickets
  ✅ No orphan IDs in usage_logs

Value checks
  ✅ Churn column contains only Yes / No
  ✅ Contract contains only valid values
  ✅ Priority and status contain only valid values
  ✅ No negative tenure or charges
  ✅ No negative login counts or data usage
  ✅ Tenure range between 0 and 120 months

Outlier detection
  ✅ IQR-based outlier detection on 5 numeric columns
```

---

## 🤖 Model — Why Logistic Regression

I originally built this with Random Forest — better AUC on paper. But when I tried to explain why a specific customer was flagged, I needed SHAP — a 30-second computation and a library most business people have never heard of.

I switched to Logistic Regression. The model now explains itself through odds ratios:

```
Feature                   Coefficient    Odds Ratio    Business meaning
──────────────────────    ───────────    ──────────    ─────────────────────────────────
login_drop_pct              +0.842          2.32x       High drop → 2.3x more likely to churn
contract_encoded            −1.240          0.29x       Longer contract → 71% less likely
billing_pct                 +0.631          1.88x       Billing complaints → 1.9x more likely
support_services_count      −0.580          0.56x       More services → 44% less likely
```

No SHAP. No black box. Coefficient × feature value = contribution. Any business person can read it.

**Pipeline:**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(
        class_weight = 'balanced',   # handles 27% churn imbalance natively
        C            = 1.0,
        max_iter     = 1000,
        solver       = 'lbfgs'
    ))
])
```

---

## 🌐 Dashboard — 4 Pages

### 🏠 Overview
KPIs at a glance — total customers, churn rate, risk segment counts, full pipeline status card

### 📊 EDA Explorer
4 interactive tabs — contract analysis, tenure and usage patterns, support ticket breakdown, full correlation heatmap

### 🤖 Model Performance
ROC-AUC curve, confusion matrix with business cost framing, coefficient plot with odds ratios, net business value estimate

### 🔮 Live Prediction
The most important page. Enter any customer's details and instantly see:
- Churn probability with a visual gauge
- Risk level — High / Medium / Low
- **Exact reasons why they may churn** with contribution bars
- What features are protecting them
- Full feature contribution table

---

## 📈 Key Insights from the Data

```
1. Contract type is the strongest predictor
   Month-to-month customers churn at ~42%
   Two-year contract customers churn at ~11%
   3.8× difference from one variable alone

2. New customers are the most vulnerable
   Median tenure of churned customers  : ~10 months
   Median tenure of retained customers : ~38 months
   Get them past month 12 and they become loyal

3. Login drop predicts churn weeks in advance
   Customers who reduced logins by 50%+ churn significantly more
   This is the earliest detectable signal in the data

4. Fiber optic customers churn more despite better service
   Likely price sensitivity — they have more alternatives
   Churn rate: Fiber optic ~42% vs DSL ~19%

5. Billing tickets are a warning sign
   A customer with majority billing tickets
   is showing financial dissatisfaction before they act on it
```

---

## 🚀 Running Locally

### Option 1 — Streamlit only (quickest)

```bash
# 1. Clone the repo
git clone https://github.com/adicode200/Predicting-Who-is-going-to-unsubscribe
cd Predicting-Who-is-going-to-unsubscribe

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

# 4. Run the pipeline in order
python src/01oad_data.py
python src/02generate_ticket.py
python src/03generate_usage.py
python src/05validation.py
python src/06featureEngineering.py
python src/07EDA.py
python src/08TrainModel.py
python src/09modelEvaluation.py

# 5. Launch dashboard
streamlit run app.py
```

### Option 2 — Docker (recommended)

```bash
# 1. Clone the repo
git clone https://github.com/adicode200/Predicting-Who-is-going-to-unsubscribe
cd Predicting-Who-is-going-to-unsubscribe

# 2. Create .env with your credentials
# 3. Build and run
docker-compose up --build

# Open http://localhost:8501
```

---

## ⚙️ Environment Variables

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

---

## 🔭 Where This Can Be Extended

This project is built in a way that makes each of these extensions a natural next step — not a rewrite:

### Extend the data
```
More months of usage logs
→ login_trend_slope becomes more powerful with 6 or 12 months
→ seasonal patterns become detectable

Real support ticket data
→ add NLP sentiment analysis on ticket descriptions
→ angry ticket text becomes a feature itself

Add product usage data
→ which features of the product does the customer use?
→ customers using fewer features are more at risk
```

### Extend the model
```
Add XGBoost as a second model
→ compare against Logistic Regression on the same test set
→ build a model selection framework

Add probability calibration
→ Platt scaling to make probabilities more reliable
→ better threshold tuning

Add time-series features
→ rolling 6-month window instead of fixed Jan-Mar
→ model updates automatically as new months arrive
```

### Extend the system
```
Replace static CSV with live DB query
→ dashboard always shows fresh data
→ one line change: pd.read_sql() instead of pd.read_csv()

Add a FastAPI prediction endpoint
→ any CRM or mobile app can call your model via REST API
→ sales team gets churn scores inside their existing tools

Automate retraining with a scheduler
→ new data arrives each month
→ pipeline runs automatically, model updates, dashboard refreshes
→ you do nothing

Add authentication to the dashboard
→ different teams see different customer segments
→ managers see aggregates, agents see individual customers

Deploy on AWS or GCP
→ move from Streamlit Community Cloud to a real server
→ handles concurrent users, custom domain, SSL
```

### Extend to other industries
```
This exact pipeline works for:
→ SaaS companies     (replace usage_logs with feature adoption logs)
→ Banking            (replace tickets with transaction complaints)
→ Streaming services (replace logins with watch time)
→ Insurance          (replace charges with premium data)

Only src/06featureEngineering.py needs changing.
Everything else — validation, model, dashboard — stays identical.
```

---

## 🛠️ Tech Stack

| Category | Technology | Why this choice |
|----------|-----------|-----------------|
| Language | Python 3.11 | Industry standard for ML |
| Database | PostgreSQL 15 | Real relational DB — not just CSV |
| Data | pandas, numpy | Standard data manipulation |
| ML | scikit-learn | Clean pipeline API |
| Visualisation | matplotlib, seaborn | Full control over plots |
| Dashboard | Streamlit | Fastest path from model to UI |
| Containers | Docker, Docker Compose | Reproducible environment |
| Deployment | Streamlit Community Cloud | Free public URL for portfolio |
| Env management | python-dotenv | Keeps credentials out of code |

---

## 📚 Concepts Covered

```
SQL              JOINs · CTEs · GROUP BY · CASE WHEN · Window functions
                 NULLIF · COALESCE · Pivot with MAX(CASE WHEN)

Data Engineering Multi-table schema design · Referential integrity
                 Synthetic data generation · ETL pipeline

Validation       Automated quality checks · Data contracts
                 IQR outlier detection · Assert statements

Feature Eng.     Signal engineering from timestamps and counts
                 Ordinal encoding · Ratio features · Trend slopes

Statistics       Class imbalance · Correlation analysis
                 Distribution analysis · IQR method

Machine Learning Logistic Regression · StandardScaler · Pipelines
                 Cross-validation · class_weight · Regularisation

Evaluation       ROC-AUC · Confusion matrix · Precision-Recall curve
                 Optimal threshold tuning · Business cost framing

Explainability   Log-odds · Odds ratios · Coefficient interpretation
                 Feature contribution = value × coefficient

MLOps            Docker · docker-compose · Environment management
                 Model serialisation with joblib · .env pattern

Deployment       Streamlit Cloud · Public URL · Container deployment
```

---

## 👤 Author

**Aditya**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/aditya-yadav-71250125a/
)
[![GitHub](https://img.shields.io/badge/GitHub-adicode200-181717?style=flat&logo=github)](https://github.com/adicode200)
[![Live App](https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=flat&logo=streamlit)](https://telco-churn-insight.streamlit.app)

---

<div align="center">

**If this project helped you or you found it interesting, consider giving it a ⭐**

*Built entirely from scratch — no tutorials, no shortcuts.*

</div>
