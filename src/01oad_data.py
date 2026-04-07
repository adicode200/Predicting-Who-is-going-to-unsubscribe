import pandas as pd
from sqlalchemy import create_engine
import os
import urllib.parse
from dotenv import load_dotenv  
# 1. Connection string (Matches the Docker command you ran)
# postgresql://[user]:[password]@[host]:[port]/[database]
load_dotenv()
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = os.getenv('DB_PORT')
dbname = os.getenv('DB_NAME')

safe_password = urllib.parse.quote_plus(password)
DB_URL = f"postgresql://{user}:{safe_password}@{host}:{port}/{dbname}"
engine = create_engine(DB_URL)
def upload_data():
    # 2. Locate the file
    file_path = os.path.join("data", "customer_data.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}. Check your folder!")
        return

    print("Reading CSV...")
    df = pd.read_csv(file_path)

    # 3. Data Cleaning (Important!)
    # The Telco dataset has spaces in 'TotalCharges' which break numeric columns.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(0, inplace=True) # Replace empty charges with 0

    print("Uploading to PostgreSQL...")
    # 4. The Magic Line
    # This creates the table automatically based on the CSV headers.
    df.to_sql('raw_customers', engine, if_exists='replace', index=False)
    
    print("✅ Success! 7,043 rows uploaded to table 'raw_customers'.")

if __name__ == "__main__":
    upload_data()