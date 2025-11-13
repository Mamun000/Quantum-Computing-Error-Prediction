import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "SVM": SVC(random_state=42, probability=True)
            }
            
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2']
                },
                "SVM": {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            # Get best model based on ROC-AUC score
            best_model_name = max(model_report, key=lambda k: model_report[k]['roc_auc'])
            best_model_score = model_report[best_model_name]['roc_auc']
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance", sys)
            
            logging.info(f"Best model: {best_model_name} with ROC-AUC: {best_model_score}")
            
            # Train best model on full training data
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            
            logging.info(f"Model training completed. Final accuracy: {accuracy}")
            
            return accuracy, best_model_name
            
        except Exception as e:
            raise CustomException(e, sys)