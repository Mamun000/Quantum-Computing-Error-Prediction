import sys
import os
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def start_training(self):
        try:
            print("\n" + "="*60)
            print("🚀 QUANTUM ERROR PREDICTION - TRAINING PIPELINE")
            print("="*60 + "\n")
            
            logging.info("Training pipeline started")
            
            # Data Ingestion
            print("📊 Step 1: Data Ingestion")
            print("-" * 40)
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            print(f"✅ Training data saved: {train_data_path}")
            print(f"✅ Test data saved: {test_data_path}\n")
            
            # Data Transformation
            print("🔧 Step 2: Data Transformation & Feature Engineering")
            print("-" * 40)
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            print(f"✅ Features engineered and scaled")
            print(f"✅ Preprocessor saved: {preprocessor_path}\n")
            
            # Model Training
            print("🤖 Step 3: Model Training & Evaluation")
            print("-" * 40)
            model_trainer = ModelTrainer()
            accuracy, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
            print(f"✅ Model training completed")
            print(f"✅ Best Model: {best_model}")
            print(f"✅ Model Accuracy: {accuracy:.4f}\n")
            
            logging.info(f"Training pipeline completed. Best Model: {best_model}, Accuracy: {accuracy}")
            
            print("="*60)
            print("✨ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\n📁 Artifacts saved in: {os.path.abspath('artifacts')}/")
            print("   - model.pkl")
            print("   - preprocessor.pkl")
            print("   - data.csv")
            print("   - train.csv")
            print("   - test.csv\n")
            print("🎯 Next step: Run 'python app.py' to start the web application\n")
            
            return accuracy, best_model
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}\n")
            logging.error(f"Training pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        obj = TrainPipeline()
        accuracy, model_name = obj.start_training()
    except Exception as e:
        print(f"\n💥 Training failed with error: {str(e)}")
        print("Check the logs folder for detailed error information.")