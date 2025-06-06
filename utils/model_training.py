from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
import pandas as pd
import joblib
from utils.data_processing import prepare_training_data

def train_and_evaluate_models():
    """Train and evaluate multiple models with robust categorical handling"""
    try:
        # Prepare the data
        X_train, X_test, y_train, y_test = prepare_training_data()
        
        # Get categorical features indices for CatBoost
        categorical_features = [
            'TelecomCompany', 'Region', 'Gender', 'ContractType',
            'ContractDuration', 'PaymentMethod', 'InternetService',
            'AdditionalServices', 'DiscountOfferUsed'
        ]
        cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, 
                                   eval_metric='logloss', scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])),
            'CatBoost': CatBoostClassifier(
                cat_features=cat_indices,
                random_state=42,
                verbose=0,
                iterations=500,
                auto_class_weights='Balanced'
            ),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
        }
        
        # Train and evaluate models
        results = []
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                
                results.append({
                    'Model': name,
                    'CV Mean AUC': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Test Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'ROC AUC': roc_auc
                })
                
                # Save the best model (prioritize CatBoost for categorical handling)
                if name == 'CatBoost':
                    joblib.dump(model, 'models/churn_model.pkl')
                    # Save backup model
                    joblib.dump(models['XGBoost'], 'models/backup_churn_model.pkl')
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv('models/model_performance.csv', index=False)
        
        return results_df
    
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")

def get_feature_importance():
    """Get feature importance from the best model"""
    try:
        # Try loading CatBoost first
        model = joblib.load('models/churn_model.pkl')
    except:
        try:
            model = joblib.load('models/backup_churn_model.pkl')
        except:
            return None
    
    # Get feature importance
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = model.coef_[0]
    else:
        return None
    
    # Get feature names
    df = pd.read_csv('data/Tanzania_Telecom_Churn_10K.csv')
    features = df.drop(['CustomerID', 'Churn'], axis=1).columns
    
    # Create DataFrame of feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return feature_importance
