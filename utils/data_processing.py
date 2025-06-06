import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.utils import column_or_1d

# Standard categories without 'Unknown'
STANDARD_CATEGORIES = {
    'TelecomCompany': ['Airtel', 'Tigo', 'Vodacom', 'Halotel', 'TTCL', 'Zantel'],
    'InternetService': ['Mobile Data', 'Fiber', 'DSL', 'Broadband', 'None'],
    'PaymentMethod': ['Credit Card', 'Bank Transfer', 'Mobile Money', 'Cash'],
    'AdditionalServices': ['Streaming', 'VPN', 'Cloud Storage', 'None'],
    'ContractDuration': ['1 Month', '6 Months', '12 Months', '24 Months']
}

def preprocess_data(df, training_mode=False):
    """Preprocess the input data without Unknown category"""
    try:
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Handle missing values using mode (most frequent category)
        for col, default in [('InternetService', 'None'), 
                           ('AdditionalServices', 'None'),
                           ('PaymentMethod', 'Mobile Money'),  # Most common payment method
                           ('TelecomCompany', 'Airtel')]:      # Most common telecom company
            if col in df.columns:
                df[col] = df[col].fillna(default)
                if col in STANDARD_CATEGORIES:
                    # Replace any non-standard values with the default
                    df[col] = df[col].apply(lambda x: x if x in STANDARD_CATEGORIES[col] else default)

        # Apply label encoding
        categorical_cols = [
            'TelecomCompany', 'Region', 'Gender', 'ContractType', 
            'ContractDuration', 'PaymentMethod', 'InternetService', 
            'AdditionalServices', 'DiscountOfferUsed'
        ]
        
        if training_mode:
            # Initialize and fit label encoders during training
            label_encoders = {}
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    le.fit(df[col].unique())
                    df[col] = le.transform(df[col])
                    label_encoders[col] = le
            
            # Initialize and fit scaler
            numerical_cols = [
                'Age', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
                'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
                'BillingIssuesReported'
            ]
            
            scaler = StandardScaler()
            if numerical_cols[0] in df.columns:
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            
            return df, label_encoders, scaler
        
        else:
            # Load preprocessing objects for prediction
            label_encoders = joblib.load('models/label_encoders.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            for col in categorical_cols:
                if col in df.columns and col in label_encoders:
                    le = label_encoders[col]
                    # Replace any unseen categories with the most frequent category
                    mask = ~df[col].isin(le.classes_)
                    if mask.any():
                        most_frequent = pd.Series(le.classes_).mode()[0]
                        df.loc[mask, col] = most_frequent
                    df[col] = le.transform(df[col])
                elif col in df.columns:
                    raise ValueError(f"Label encoder for {col} not found")

            # Scale numerical features
            numerical_cols = [
                'Age', 'TenureMonths', 'MonthlyCharges', 'DataUsageGB',
                'CallDurationMinutes', 'ComplaintsFiled', 'CustomerSupportCalls',
                'BillingIssuesReported'
            ]
            
            if numerical_cols[0] in df.columns:
                df[numerical_cols] = scaler.transform(df[numerical_cols])
            
            return df
    
    except Exception as e:
        raise RuntimeError(f"Data preprocessing failed: {str(e)}")

def prepare_training_data():
    """Prepare training data without Unknown category"""
    try:
        # Load dataset
        df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
        
        # Drop CustomerID
        df = df.drop('CustomerID', axis=1)
        
        # Convert Churn to binary
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Preprocess data in training mode
        X, label_encoders, scaler = preprocess_data(df.drop('Churn', axis=1), training_mode=True)
        y = df['Churn']
        
        # Save preprocessing objects
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test
    
    except Exception as e:
        raise RuntimeError(f"Training data preparation failed: {str(e)}")