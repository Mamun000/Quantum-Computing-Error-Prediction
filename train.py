"""
Simplified training script with better error handling and progress tracking
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    try:
        print("=" * 60)
        print("🚀 QUANTUM ERROR PREDICTION - TRAINING PIPELINE")
        print("=" * 60)
        print()
        
        from src.pipeline.train_pipeline import TrainPipeline
        
        print("📊 Starting training pipeline...")
        print()
        
        pipeline = TrainPipeline()
        accuracy, model_name = pipeline.start_training()
        
        print()
        print("=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🏆 Best Model: {model_name}")
        print(f"📈 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print()
        print("📁 Generated Files:")
        print("   ✓ artifacts/data.csv")
        print("   ✓ artifacts/train.csv")
        print("   ✓ artifacts/test.csv")
        print("   ✓ artifacts/model.pkl")
        print("   ✓ artifacts/preprocessor.pkl")
        print()
        print("🎉 Ready to run the application!")
        print("   Run: python app.py")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR OCCURRED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())