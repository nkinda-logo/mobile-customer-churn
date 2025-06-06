import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('database/churn_predictions.db')
    c = conn.cursor()
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  customer_id INTEGER,
                  telecom_company TEXT,
                  region TEXT,
                  age INTEGER,
                  gender TEXT,
                  contract_type TEXT,
                  contract_duration INTEGER,
                  tenure_months INTEGER,
                  monthly_charges REAL,
                  data_usage_gb REAL,
                  call_duration_minutes INTEGER,
                  complaints_filed INTEGER,
                  customer_support_calls INTEGER,
                  payment_method TEXT,
                  internet_service TEXT,
                  additional_services TEXT,
                  discount_offer_used TEXT,
                  billing_issues_reported INTEGER,
                  prediction INTEGER,
                  probability REAL,
                  prediction_date TEXT)''')
    
    # Create users table for authentication
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
                  role TEXT DEFAULT 'user')''')
    
    # Insert admin user if not exists
    try:
        c.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                  ('admin', 'pbkdf2:sha256:260000$X8D3...', 'admin@example.com', 'admin'))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()