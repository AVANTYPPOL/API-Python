#!/usr/bin/env python3
"""
Retrain the model ensuring cloud compatibility
This script should be run in an environment with:
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- xgboost==2.0.3
"""

import sys
import os

print("="*60)
print("Cloud-Compatible Model Training")
print("="*60)

# Check current versions
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost as xgb
    
    print("\n📦 Current package versions:")
    print(f"   NumPy: {np.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   Scikit-learn: {sklearn.__version__}")
    print(f"   XGBoost: {xgb.__version__}")
    print(f"   Has numpy._core: {hasattr(np, '_core')}")
    
    # Check if versions match cloud requirements
    cloud_versions = {
        'numpy': '1.24.3',
        'pandas': '2.0.3',
        'scikit-learn': '1.3.0',
        'xgboost': '2.0.3'
    }
    
    version_mismatch = False
    if np.__version__ != cloud_versions['numpy']:
        print(f"\n⚠️  NumPy version mismatch: {np.__version__} != {cloud_versions['numpy']}")
        version_mismatch = True
    
    if version_mismatch:
        print("\n❌ Version mismatch detected!")
        print("\n🔧 To fix this, create a virtual environment:")
        print("   python3 -m venv venv_cloud")
        print("   source venv_cloud/bin/activate")
        print("   pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 xgboost==2.0.3 joblib==1.3.2")
        print("   python retrain_for_cloud.py")
        
        # Continue anyway for now
        response = input("\nContinue with current versions? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Import the model class
    print("\n🔄 Importing model class...")
    
    # First, let's create a minimal version that doesn't need all the compatibility fixes
    import pickle
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score
    import sqlite3
    
    print("✅ Imports successful")
    
    # Load the existing model to get its configuration
    print("\n📂 Loading existing model configuration...")
    
    try:
        # Try to load just to get the configuration
        with open('xgboost_miami_model.pkl', 'rb') as f:
            existing_data = joblib.load(f)
        
        print("✅ Loaded existing model data")
        feature_columns = existing_data.get('feature_columns', [])
        service_multipliers = existing_data.get('service_multipliers', {})
        
    except Exception as e:
        print(f"⚠️  Could not load existing model: {e}")
        print("   Will train fresh model")
        feature_columns = None
        service_multipliers = {'UBERX': 1.0, 'UBERXL': 1.4, 'PREMIER': 1.8, 'SUV_PREMIER': 2.2}
    
    # Now train a fresh model
    print("\n🚀 Training fresh model with cloud-compatible setup...")
    
    # Import the model class properly
    from xgboost_miami_model import XGBoostMiamiModel
    
    # Create and train model
    model = XGBoostMiamiModel('uber_ml_data.db')
    
    # If we have existing feature columns, use them
    if feature_columns:
        model.feature_columns = feature_columns
    if service_multipliers:
        model.service_multipliers = service_multipliers
    
    # Train the model
    model.train()
    
    # Save with explicit protocol
    output_file = 'xgboost_miami_model_cloud_v2.pkl'
    
    print(f"\n💾 Saving model to {output_file}...")
    
    # Save with joblib protocol 4 for compatibility
    model_data = {
        'model': model.model,
        'label_encoders': model.label_encoders,
        'feature_columns': model.feature_columns,
        'service_multipliers': model.service_multipliers
    }
    
    joblib.dump(model_data, output_file, protocol=4)
    print(f"✅ Model saved successfully!")
    
    # Test loading
    print("\n🧪 Testing model load...")
    test_data = joblib.load(output_file)
    print("✅ Model loads successfully!")
    
    # Make a test prediction
    print("\n🔮 Testing prediction...")
    test_model = XGBoostMiamiModel()
    test_model.model = test_data['model']
    test_model.label_encoders = test_data['label_encoders'] 
    test_model.feature_columns = test_data['feature_columns']
    test_model.service_multipliers = test_data['service_multipliers']
    test_model.is_trained = True
    
    predictions = test_model.predict_all_services(
        pickup_lat=25.7959, pickup_lng=-80.2870,
        dropoff_lat=25.7617, dropoff_lng=-80.1918
    )
    
    print("   Test predictions (Airport → South Beach):")
    for service, price in predictions.items():
        print(f"   {service}: ${price:.2f}")
    
    print("\n✅ Model training and testing complete!")
    print(f"\n📝 Next steps:")
    print(f"1. Replace the model file:")
    print(f"   mv {output_file} xgboost_miami_model.pkl")
    print(f"2. Commit and deploy:")
    print(f"   git add xgboost_miami_model.pkl")
    print(f"   git commit -m 'Retrained model with cloud-compatible versions'")
    print(f"   git push")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\n📦 Please install required packages:")
    print("   pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 xgboost==2.0.3 joblib==1.3.2")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)