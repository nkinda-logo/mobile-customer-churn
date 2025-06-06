import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from utils.model_training import get_feature_importance

def plot_to_base64(plt):
    """Convert matplotlib plot to base64 encoded image"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

def create_visualizations():
    """Create all visualizations for the dashboard"""
    try:
        # Load the data
        df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
        
        # Create visualizations dictionary
        viz_data = {}
        
        # 1. Churn Distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Churn', data=df, palette='viridis')
        plt.title('Customer Churn Distribution')
        plt.xlabel('Churn Status')
        plt.ylabel('Count')
        viz_data['churn_dist'] = plot_to_base64(plt)
        
        # 2. Churn by Telecom Company
        plt.figure(figsize=(12, 6))
        sns.countplot(x='TelecomCompany', hue='Churn', data=df, palette='viridis')
        plt.title('Churn Distribution by Telecom Company')
        plt.xlabel('Telecom Company')
        plt.ylabel('Count')
        plt.legend(title='Churn Status')
        plt.xticks(rotation=45)
        viz_data['churn_by_company'] = plot_to_base64(plt)
        
        # 3. Churn by Contract Type
        plt.figure(figsize=(10, 6))
        sns.countplot(x='ContractType', hue='Churn', data=df, palette='viridis')
        plt.title('Churn Distribution by Contract Type')
        plt.xlabel('Contract Type')
        plt.ylabel('Count')
        plt.legend(title='Churn Status')
        viz_data['churn_by_contract'] = plot_to_base64(plt)
        
        # 4. Churn by Payment Method
        plt.figure(figsize=(12, 6))
        sns.countplot(x='PaymentMethod', hue='Churn', data=df, palette='viridis')
        plt.title('Churn Distribution by Payment Method')
        plt.xlabel('Payment Method')
        plt.ylabel('Count')
        plt.legend(title='Churn Status')
        plt.xticks(rotation=45)
        viz_data['churn_by_payment'] = plot_to_base64(plt)
        
        # 5. Age Distribution by Churn
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Churn', y='Age', data=df, palette='viridis')
        plt.title('Age Distribution by Churn Status')
        plt.xlabel('Churn Status')
        plt.ylabel('Age')
        viz_data['age_dist'] = plot_to_base64(plt)
        
        # 6. Tenure Distribution by Churn
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Churn', y='TenureMonths', data=df, palette='viridis')
        plt.title('Tenure Distribution by Churn Status')
        plt.xlabel('Churn Status')
        plt.ylabel('Tenure (Months)')
        viz_data['tenure_dist'] = plot_to_base64(plt)
        
        # 7. Monthly Charges by Churn
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='viridis')
        plt.title('Monthly Charges Distribution by Churn Status')
        plt.xlabel('Churn Status')
        plt.ylabel('Monthly Charges')
        viz_data['charges_dist'] = plot_to_base64(plt)
        
        # 8. Feature Importance
        feature_importance = get_feature_importance()
        if feature_importance is not None:
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', 
                       data=feature_importance.head(10), palette='viridis')
            plt.title('Top 10 Most Important Features for Churn Prediction')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            viz_data['feature_imp'] = plot_to_base64(plt)
        
        return viz_data
    
    except Exception as e:
        raise RuntimeError(f"Visualization creation failed: {str(e)}")