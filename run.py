"""
Complete setup and run script - handles warnings automatically
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys

def check_artifacts_exist():
    """Check if model artifacts already exist"""
    required_files = [
        'artifacts/model.pkl',
        'artifacts/preprocessor.pkl'
    ]
    return all(os.path.exists(f) for f in required_files)

def train_model():
    """Train the model"""
    print("=" * 70)
    print("🔬 QUANTUM ERROR PREDICTION SYSTEM")
    print("=" * 70)
    print()
    
    if check_artifacts_exist():
        print("✅ Model artifacts found!")
        print("   - artifacts/model.pkl")
        print("   - artifacts/preprocessor.pkl")
        print()
        response = input("Do you want to retrain the model? (y/n): ").lower()
        if response != 'y':
            print("\n✅ Using existing model. Starting application...")
            return True
    
    print("🚀 Starting model training...")
    print("   This may take 2-5 minutes depending on your system.")
    print()
    
    try:
        from src.pipeline.train_pipeline import TrainPipeline
        
        print("📊 Training in progress...")
        pipeline = TrainPipeline()
        accuracy, model_name = pipeline.start_training()
        
        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETED!")
        print("=" * 70)
        print(f"🏆 Best Model: {model_name}")
        print(f"📈 Accuracy: {accuracy:.2%}")
        print()
        print("📁 Generated artifacts:")
        print("   ✓ artifacts/model.pkl")
        print("   ✓ artifacts/preprocessor.pkl")
        print("   ✓ artifacts/data.csv")
        print("   ✓ artifacts/train.csv")
        print("   ✓ artifacts/test.csv")
        print()
        
        return True
        
    except Exception as e:
        print()
        print("❌ Training failed!")
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False

def run_app():
    """Run the Flask application"""
    print("=" * 70)
    print("🌐 STARTING WEB APPLICATION")
    print("=" * 70)
    print()
    print("🔗 Access the application at: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    print()
    print("-" * 70)
    print()
    
    try:
        from app import app
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("👋 Application stopped by user")
        print("=" * 70)
    except Exception as e:
        print()
        print("❌ Application error!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution"""
    try:
        # Ensure directories exist
        os.makedirs('artifacts', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        
        # Train or use existing model
        if train_model():
            print("⏳ Starting application in 2 seconds...")
            import time
            time.sleep(2)
            
            # Run the application
            run_app()
        else:
            print()
            print("❌ Cannot start application without trained model.")
            print("   Please fix the training errors and try again.")
            return 1
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ UNEXPECTED ERROR")
        print("=" * 70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())