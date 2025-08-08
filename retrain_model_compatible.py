#!/usr/bin/env python3
"""
Retrain XGBoost model with deployment-compatible package versions
This ensures the model can be loaded in the cloud environment
"""

import subprocess
import sys
import os

def check_and_install_packages():
    """Check current versions and install deployment-compatible versions if needed"""
    
    print("🔍 Checking current package versions...")
    
    # Check current versions
    import numpy
    import pandas
    import sklearn
    import xgboost
    
    print(f"Current numpy: {numpy.__version__}")
    print(f"Current pandas: {pandas.__version__}")
    print(f"Current scikit-learn: {sklearn.__version__}")
    print(f"Current xgboost: {xgboost.__version__}")
    
    # Required versions for deployment
    required_versions = {
        'numpy': '1.24.3',
        'pandas': '2.0.3',
        'scikit-learn': '1.3.0',
        'xgboost': '2.0.3'
    }
    
    print("\n📋 Required versions for deployment:")
    for pkg, version in required_versions.items():
        print(f"{pkg}: {version}")
    
    # Check if we need to create a virtual environment
    if (numpy.__version__ != required_versions['numpy'] or 
        pandas.__version__ != required_versions['pandas'] or
        sklearn.__version__ != required_versions['scikit-learn'] or
        xgboost.__version__ != required_versions['xgboost']):
        
        print("\n⚠️  Version mismatch detected!")
        print("Please create a virtual environment with the correct versions:")
        print("\n# Create and activate virtual environment")
        print("python3 -m venv venv_deploy")
        print("source venv_deploy/bin/activate  # On Windows: venv_deploy\\Scripts\\activate")
        print("\n# Install required versions")
        print("pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 xgboost==2.0.3 joblib==1.3.2")
        print("\n# Then run this script again")
        return False
    
    print("\n✅ All packages are at the correct versions!")
    return True

def retrain_model():
    """Retrain the model with deployment-compatible versions"""
    
    print("\n🚀 Starting model retraining with deployment-compatible versions...")
    
    # Import the model class
    from xgboost_miami_model import XGBoostMiamiModel
    
    # Create and train the model
    print("📊 Loading data and training model...")
    model = XGBoostMiamiModel('uber_ml_data.db')
    
    # Train the model
    model.train()
    
    # Save with a new filename to avoid overwriting
    new_model_path = 'xgboost_miami_model_deploy.pkl'
    model.save_model(new_model_path)
    
    print(f"\n✅ Model trained and saved to: {new_model_path}")
    
    # Test the model
    print("\n🧪 Testing the new model...")
    test_model = XGBoostMiamiModel()
    test_model.load_model(new_model_path)
    
    # Test prediction
    predictions = test_model.predict_all_services(
        pickup_lat=25.7959,
        pickup_lng=-80.2870,
        dropoff_lat=25.7617,
        dropoff_lng=-80.1918
    )
    
    print("Sample predictions (Airport to South Beach):")
    for service, price in predictions.items():
        print(f"  {service}: ${price:.2f}")
    
    print("\n✅ Model testing successful!")
    
    # Backup old model and replace with new one
    print("\n📦 Replacing old model file...")
    if os.path.exists('xgboost_miami_model.pkl'):
        os.rename('xgboost_miami_model.pkl', 'xgboost_miami_model_backup.pkl')
        print("  - Old model backed up to: xgboost_miami_model_backup.pkl")
    
    os.rename(new_model_path, 'xgboost_miami_model.pkl')
    print("  - New model saved as: xgboost_miami_model.pkl")
    
    print("\n🎉 Model retraining complete!")
    print("Next steps:")
    print("1. Test the API locally: python3 app.py")
    print("2. Commit and push: git add xgboost_miami_model.pkl && git commit -m 'Retrain model with deployment-compatible versions' && git push")

def main():
    """Main function"""
    print("="*60)
    print("XGBoost Model Retraining for Deployment Compatibility")
    print("="*60)
    
    # Check if we have the right versions
    if check_and_install_packages():
        # Check if database exists
        if not os.path.exists('uber_ml_data.db'):
            print("\n❌ Error: uber_ml_data.db not found!")
            print("Please ensure the database file is in the current directory.")
            return
        
        # Proceed with retraining
        try:
            retrain_model()
        except Exception as e:
            print(f"\n❌ Error during retraining: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()