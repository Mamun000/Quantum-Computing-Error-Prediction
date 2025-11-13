import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved: {file_path}")
        
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
            logging.info(f"Training: {model_name}")
            
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
            
            logging.info(f"{model_name} - Accuracy: {test_model_score:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

def calculate_business_impact(prediction, probability, actual=None):
    """
    FIXED Business impact calculation with proper logic
    Now high-risk scenarios show POSITIVE net impact
    """
    try:
        # Ensure proper types
        prediction = int(prediction)
        probability = float(probability)
        
        logging.info(f"Calculating business impact - Prediction: {prediction}, Probability: {probability:.4f}")
        
        # REVISED Business cost constants (in millions) - FIXED LOGIC
        COST_PREVENTION = 0.5      # Cost of preventive maintenance - $0.5M
        COST_FAILURE = 12.0        # Cost of quantum system failure - $12M
        BENEFIT_PREVENTION = 10.0  # Benefit of preventing failure - $10M
        COST_MONITORING = 0.1      # Cost of normal monitoring - $0.1M
        
        # Calculate expected impact based on prediction and probability
        if prediction == 1:  # Predicted error - TAKE PREVENTIVE ACTION
            # If we prevent an error: We AVOID failure cost and GET prevention benefit
            expected_benefit = probability * (BENEFIT_PREVENTION + COST_FAILURE)
            # Cost: We always pay prevention cost
            expected_cost = COST_PREVENTION
            net_expected = expected_benefit - expected_cost
            
            # Enhanced recommendations
            if probability > 0.7:
                recommendation = "🚨 CRITICAL RISK - Immediate preventive action recommended. Potential net benefit: $%.1fM" % (net_expected)
                confidence_level = "Very High"
            elif probability > 0.5:
                recommendation = "⚠️ HIGH RISK - Schedule preventive maintenance. Expected net benefit: $%.1fM" % (net_expected)
                confidence_level = "High"
            else:
                recommendation = "🔶 MEDIUM RISK - Consider preventive measures. Net impact: $%.1fM" % (net_expected)
                confidence_level = "Medium"
                
        else:  # Predicted no error - CONTINUE MONITORING
            # Benefit: We avoid unnecessary prevention cost
            expected_benefit = COST_PREVENTION * (1 - probability)
            # Cost: Small risk of failure + monitoring cost
            expected_cost = (probability * COST_FAILURE) + COST_MONITORING
            net_expected = expected_benefit - expected_cost
            
            if probability < 0.2:
                recommendation = "✅ LOW RISK - System stable. Continue normal operations with monitoring."
                confidence_level = "Very High"
            elif probability < 0.4:
                recommendation = "🔷 MODERATE RISK - Enhanced monitoring recommended."
                confidence_level = "High"
            else:
                recommendation = "💡 ELEVATED RISK - Close monitoring advised. Consider preventive action."
                confidence_level = "Medium"
        
        # Format financial values in millions
        def format_millions(value):
            value = float(value)
            if value >= 0:
                if abs(value) >= 1.0:
                    return "$%.1fM" % value
                else:
                    return "$%.1fK" % (value * 1000)
            else:
                if abs(value) >= 1.0:
                    return "-$%.1fM" % abs(value)
                else:
                    return "-$%.1fK" % (abs(value) * 1000)
        
        result = {
            'prediction': 'Error Expected 🚨' if prediction == 1 else 'No Error ✅',
            'probability': f"{probability * 100:.2f}%",
            'confidence_level': confidence_level,
            'expected_benefit': format_millions(expected_benefit),
            'expected_cost': format_millions(expected_cost),
            'net_impact': format_millions(net_expected),
            'roi_potential': format_millions(net_expected),  # Same as net impact for simplicity
            'recommendation': recommendation
        }
        
        logging.info(f"Business Impact Results:")
        logging.info(f"  Expected Benefit: {result['expected_benefit']}")
        logging.info(f"  Expected Cost: {result['expected_cost']}")
        logging.info(f"  Net Impact: {result['net_impact']}")
        logging.info(f"  Recommendation: {result['recommendation']}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in business impact calculation: {e}")
        # Return default values in case of error
        return {
            'prediction': 'Error',
            'probability': '0%',
            'confidence_level': 'Unknown',
            'expected_benefit': '$0',
            'expected_cost': '$0', 
            'net_impact': '$0',
            'roi_potential': '$0',
            'recommendation': 'Calculation error'
        }