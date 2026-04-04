import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
password = "Aditya@123"
safe_password = urllib.parse.quote_plus(password)
DB_URL = f"postgresql://postgres:{safe_password}@localhost:5432/postgres"
engine = create_engine(DB_URL)
query = """
    WITH ticket_summary AS (
        SELECT "customerID",
               COUNT(*) AS total_tickets,
               SUM(CASE WHEN priority IN ('High','Critical') THEN 1 ELSE 0 END) AS high_pri_tickets
        FROM support_tickets
        GROUP BY "customerID"
    ),
    usage_summary AS (
        SELECT "customerID",
               MAX(CASE WHEN month_date='2026-01-01' THEN login_count END) AS jan_logins,
               MAX(CASE WHEN month_date='2026-03-01' THEN login_count END) AS mar_logins,
               AVG(data_usage_gb) AS avg_data_gb
        FROM usage_logs
        GROUP BY "customerID"
    )
    SELECT c."customerID", c."tenure", c."Contract",
           c."MonthlyCharges", c."TotalCharges", c."Churn",
           COALESCE(t.total_tickets, 0)    AS total_tickets,
           COALESCE(t.high_pri_tickets, 0) AS high_pri_tickets,
           u.jan_logins, u.mar_logins, u.avg_data_gb
    FROM raw_customers c
    LEFT JOIN ticket_summary t USING ("customerID")
    LEFT JOIN usage_summary  u USING ("customerID")
"""

df = pd.read_sql(query, engine)
print(df.shape)
print(df)
df.head()