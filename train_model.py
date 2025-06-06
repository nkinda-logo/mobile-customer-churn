# train_model.py
from utils.model_training import train_and_evaluate_models, get_feature_importance
import pandas as pd

if __name__ == '__main__':
    print("Training models...")
    results = train_and_evaluate_models()
    print("\nModel Evaluation Results:")
    print(results.to_string(index=False))
    
    print("\nFeature Importance:")
    feature_importance = get_feature_importance()
    print(feature_importance.to_string(index=False))
    
    print("\nModel training completed. Best model saved to 'models/churn_model.pkl'")