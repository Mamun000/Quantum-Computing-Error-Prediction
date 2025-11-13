"""
Test script to verify all dependencies and modules are working
"""
import warnings
warnings.filterwarnings('ignore')

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []
    
    modules = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'flask': 'Flask',
        'dill': 'Dill',
    }
    
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"✗ {name} - {e}")
            errors.append(name)
    
    # Test Qiskit separately (optional)
    try:
        import qiskit
        print(f"✓ Qiskit (optional)")
    except ImportError:
        print(f"⚠ Qiskit (optional) - Not installed, will use fallback")
    
    if errors:
        print(f"\n❌ Missing modules: {', '.join(errors)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required modules installed!")
    return True

def test_directories():
    """Test directory structure"""
    print("\nTesting directory structure...")
    import os
    
    dirs = ['src', 'src/components', 'src/pipeline', 'templates']
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - Missing!")
            os.makedirs(dir_path, exist_ok=True)
            print(f"  Created: {dir_path}")
    
    # Create artifacts and logs
    for dir_path in ['artifacts', 'logs']:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path}")
    
    print("\n✅ Directory structure ready!")
    return True

def test_src_modules():
    """Test if src modules can be imported"""
    print("\nTesting src modules...")
    
    try:
        from src.logger import logging
        print("✓ src.logger")
        
        from src.exception import CustomException
        print("✓ src.exception")
        
        from src.utils import save_object
        print("✓ src.utils")
        
        from src.components.data_ingestion import DataIngestion
        print("✓ src.components.data_ingestion")
        
        from src.components.data_transformation import DataTransformation
        print("✓ src.components.data_transformation")
        
        from src.components.model_trainer import ModelTrainer
        print("✓ src.components.model_trainer")
        
        from src.pipeline.train_pipeline import TrainPipeline
        print("✓ src.pipeline.train_pipeline")
        
        from src.pipeline.predict_pipeline import PredictPipeline
        print("✓ src.pipeline.predict_pipeline")
        
        print("\n✅ All src modules working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error importing src modules: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔍 QUANTUM ERROR PREDICTION - SETUP TEST")
    print("=" * 60)
    print()
    
    success = True
    success = test_imports() and success
    success = test_directories() and success
    success = test_src_modules() and success
    
    print()
    print("=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\n🚀 Ready to train! Run: python train.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 60)
        print("\nPlease fix the issues above before training.")
        return 1

if __name__ == "__main__":
    exit(main())