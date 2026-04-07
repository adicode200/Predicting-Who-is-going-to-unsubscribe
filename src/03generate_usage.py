import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse
import os 
from dotenv import load_dotenv  


# 1. Setup Connection
load_dotenv()
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')

safe_password = urllib.parse.quote_plus(password)
DB_URL = f"postgresql://{user}:{safe_password}@{host}:{port}/{dbname}"
engine = create_engine(DB_URL)

def create_usage_logs():
    print("Fetching customer IDs...")
    ids_df = pd.read_sql("SELECT \"customerID\" FROM raw_customers", engine)
    customer_ids = ids_df['customerID'].tolist()

    usage_data = []
    months = ["2026-01-01", "2026-02-01", "2026-03-01"]

    print("Generating 3 months of usage for each customer...")
    for cid in customer_ids:
        # We simulate a "Trend" - some stay steady, some drop (Churn signal!)
        # it pusnishes the less activity in the 3 month => shows the downfall
        trend_factor = np.random.choice([0.9, 1.0, 1.1, 0.5]) # 0.5 represents a 50% drop
        
        base_logins = np.random.randint(10, 30)
        base_gb = np.random.randint(50, 500)

        for i, month in enumerate(months):
            # Apply the trend factor more heavily in the final month (March)
            current_multiplier = (trend_factor ** i) 
            
            usage_data.append({
                "customerID": cid,
                "month_date": month,
                "login_count": int(base_logins * current_multiplier),
                "data_usage_gb": int(base_gb * current_multiplier)
            })

    usage_df = pd.DataFrame(usage_data)

    print("Uploading 'usage_logs' to PostgreSQL...")
    usage_df.to_sql('usage_logs', engine, if_exists='replace', index=False)
    print("✅ Success! Your 3rd table is ready.")

if __name__ == "__main__":
    create_usage_logs()