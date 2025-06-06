import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def repair_models():
    print("🔧 Repairing model artifacts...")
    
    # Create sample data with all possible categories
    sample_data = {
        'TelecomCompany': ['Airtel', 'Tigo', 'Vodacom', 'Halotel', 'TTCL', 'Zantel'],
        'Region': ['Arusha', 'Dar es Salaam', 'Dodoma', 'Geita', 'Iringa', 'Kagera',
                  'Katavi', 'Kigoma', 'Kilimanjaro', 'Lindi', 'Manyara', 'Mara',
                  'Mbeya', 'Morogoro', 'Mtwara', 'Mwanza', 'Njombe', 'Pemba North',
                  'Pemba South', 'Pwani', 'Rukwa', 'Ruvuma', 'Shinyanga', 'Simiyu',
                  'Singida', 'Songwe', 'Tabora', 'Tanga', 'Unguja North', 'Unguja South'],
        'Gender': ['Male', 'Female'],
        'ContractType': ['Prepaid', 'Postpaid', 'Hybrid'],
        'ContractDuration': ['1 Month', '3 Months', '6 Months', '12 Months', '24 Months'],
        'PaymentMethod': ['Credit Card', 'Bank Transfer', 'Mobile Money', 'Cash', 'Voucher'],
        'InternetService': ['Mobile Data', 'Fiber', 'DSL', 'WiMAX', 'None'],
        'AdditionalServices': ['Streaming', 'VPN', 'Cloud Storage', 'Gaming', 'None'],
        'DiscountOfferUsed': ['Yes', 'No'],
        'Age': [30],
        'TenureMonths': [12],
        'MonthlyCharges': [50000],
        'DataUsageGB': [5],
        'CallDurationMinutes': [300],
        'ComplaintsFiled': [0],
        'CustomerSupportCalls': [0],
        'BillingIssuesReported': [0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create and save label encoders
    label_encoders = {}
    categorical_cols = [
        'TelecomCompany', 'Region', 'Gender', 'ContractType',
        'ContractDuration', 'PaymentMethod', 'InternetService',
        'AdditionalServices', 'DiscountOfferUsed'
    ]
    
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    # Create and save scaler
    numerical_cols = [
        'Age', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
        'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
        'BillingIssuesReported'
    ]
    
    scaler = StandardScaler()
    scaler.fit(df[numerical_cols])
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Process the data to get expected columns
    processed = df.copy()
    for col in categorical_cols:
        processed[col] = label_encoders[col].transform(processed[col])
    
    processed[numerical_cols] = scaler.transform(processed[numerical_cols])
    expected_columns = processed.columns.tolist()
    joblib.dump(expected_columns, 'models/expected_columns.pkl')
    
    print("✅ Successfully repaired all model artifacts")
    print(f"Expected features: {len(expected_columns)} columns")
    return True

if __name__ == '__main__':
    repair_models()