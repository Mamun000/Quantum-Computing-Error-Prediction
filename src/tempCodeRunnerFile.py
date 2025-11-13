import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            
            para = param[model_name]
            
            gs = GridSearchCV(model, para, cv=3, scoring='roc_auc', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            # Calculate ROC-AUC
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_test_proba)
            except:
                roc_auc = test_model_score
            
            report[model_name] = {
                'train_accuracy': train_model_score,
                'test_accuracy': test_model_score,
                'roc_auc': roc_auc
            }
            
            logging.info(f"{model_name} - Test Accuracy: {test_model_score:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

def calculate_business_impact(prediction, probability, actual=None):
    """
    Calculate business impact based on prediction and probability
    
    Args:
        prediction: Model prediction (0 or 1)
        probability: Probability of error
        actual: Actual outcome (if known)
    
    Returns:
        dict with impact details
    """
    # Business cost constants
    COST_FP = 100      # Cost of false positive (unnecessary intervention)
    COST_FN = 1000     # Cost of false negative (missed error, system failure)
    BENEFIT_TP = 500   # Benefit of catching an error
    COST_TN = 0        # No cost for true negative
    
    # Calculate expected impact based on probability
    if prediction == 1:  # Predicted error
        expected_benefit = probability * BENEFIT_TP
        expected_cost = (1 - probability) * COST_FP
        net_expected = expected_benefit - expected_cost
        recommendation = "High Risk - Immediate intervention recommended"
        confidence_level = "High" if probability > 0.8 else "Medium" if probability > 0.6 else "Low"
    else:  # Predicted no error
        expected_cost = probability * COST_FN
        expected_benefit = 0
        net_expected = -expected_cost
        recommendation = "Low Risk - Continue monitoring"
        confidence_level = "High" if probability < 0.2 else "Medium" if probability < 0.4 else "Low"
    
    result = {
        'prediction': 'Error Expected' if prediction == 1 else 'No Error',
        'probability': probability,
        'confidence_level': confidence_level,
        'expected_benefit': expected_benefit,
        'expected_cost': expected_cost,
        'net_impact': net_expected,
        'recommendation': recommendation
    }
    
    # If actual outcome is known, calculate actual impact
    if actual is not None:
        if actual == 1 and prediction == 1:  # True Positive
            actual_impact = BENEFIT_TP
            outcome_type = "True Positive"
        elif actual == 0 and prediction == 1:  # False Positive
            actual_impact = -COST_FP
            outcome_type = "False Positive"
        elif actual == 1 and prediction == 0:  # False Negative
            actual_impact = -COST_FN
            outcome_type = "False Negative"
        else:  # True Negative
            actual_impact = COST_TN
            outcome_type = "True Negative"
        
        result['actual_impact'] = actual_impact
        result['outcome_type'] = outcome_type
    
    return result