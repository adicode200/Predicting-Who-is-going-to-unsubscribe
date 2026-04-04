import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib.parse

# 1. Setup Connection (Using the encoded password logic)
password = "Aditya@123"
safe_password = urllib.parse.quote_plus(password)
DB_URL = f"postgresql://postgres:{safe_password}@localhost:5432/postgres"
engine = create_engine(DB_URL)

def create_support_tickets():
    print("Fetching existing customer IDs...")
    # Get IDs from the table you already uploaded
    ids_df = pd.read_sql("SELECT \"customerID\" FROM raw_customers", engine)
    customer_ids = ids_df['customerID'].tolist() 

    # 2. Generate Synthetic Ticket Data
    num_tickets = 5000  # Let's create 5000 random complaints
    data = {
        "ticket_id": [f"TKT-{i}" for i in range(1, num_tickets + 1)],
        "customerID": np.random.choice(customer_ids, num_tickets),
        "category": np.random.choice(["Technical", "Billing", "Outage", "General"], num_tickets),
        "priority": np.random.choice(["Low", "Medium", "High", "Critical"], num_tickets, p=[0.4, 0.3, 0.2, 0.1]),
        "status": np.random.choice(["Resolved", "Closed", "Open", "Pending"], num_tickets),
        "created_at": pd.to_datetime(np.random.choice(pd.date_range('2026-01-01', '2026-03-31'), num_tickets))
    }

    tickets_df = pd.DataFrame(data)

    print("Uploading 'support_tickets' table to PostgreSQL...")
    # 3. Push to Postgres
    tickets_df.to_sql('support_tickets', engine, if_exists='replace', index=False)
    print("✅ Success! You now have a second table for JOIN practice.")

if __name__ == "__main__":
    create_support_tickets()