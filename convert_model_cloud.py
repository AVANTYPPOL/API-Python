#!/usr/bin/env python3
"""
Convert the existing model to be compatible with cloud environment
This extracts the model parameters and saves a clean version
"""

import os
import sys
import numpy as np
import pickle
import joblib

print("="*60)
print("Model Conversion for Cloud Compatibility")
print("="*60)

# Step 1: Load the model with compatibility fixes
print("\n1️⃣ Loading existing model with compatibility layer...")

class CompatibilityUnpickler(pickle.Unpickler):
    """Custom unpickler to handle various compatibility issues"""
    
    def find_class(self, module, name):
        # Handle numpy version differences
        if 'numpy._core' in module:
            module = module.replace('numpy._core', 'numpy.core')
        
        # Handle XGBoost __main__ references
        if module == '__main__' or module == 'XGBRegressor':
            if 'XGB' in name or name == 'XGBRegressor':
                import xgboost
                return xgboost.XGBRegressor
        
        return super().find_class(module, name)
    
    def load_build(self):
        """Override to handle dictionary building errors"""
        try:
            super().load_build()
        except TypeError as e:
            if "cannot convert dictionary" in str(e):
                # Skip problematic dictionary updates
                pass
            else:
                raise

# Try to load the model
try:
    # First attempt with joblib
    try:
        model_data = joblib.load('xgboost_miami_model.pkl')
        print("✅ Loaded with joblib")
    except:
        # Fallback to custom unpickler
        with open('xgboost_miami_model.pkl', 'rb') as f:
            unpickler = CompatibilityUnpickler(f)
            model_data = unpickler.load()
        print("✅ Loaded with compatibility unpickler")
    
    print(f"   Model keys: {list(model_data.keys())}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\n🔧 Attempting alternative approach...")
    
    # Alternative: Create a fresh model and train it
    print("\n2️⃣ Creating fresh model with cloud-compatible versions...")
    
    # Check if we can import the model class
    try:
        from xgboost_miami_model import XGBoostMiamiModel
        
        # Create and train a new model
        print("   Training new model...")
        model = XGBoostMiamiModel('uber_ml_data.db')
        model.train()
        
        # Save in cloud-compatible format
        output_file = 'xgboost_miami_model_cloud.pkl'
        model.save_model(output_file)
        
        print(f"✅ New model saved to: {output_file}")
        print("\n📝 Next steps:")
        print(f"1. Replace the old model: mv {output_file} xgboost_miami_model.pkl")
        print("2. Commit and push the new model")
        
    except Exception as e2:
        print(f"❌ Alternative approach also failed: {e2}")
        sys.exit(1)

# Step 2: Extract model components
if 'model_data' in locals():
    print("\n2️⃣ Extracting model components...")
    
    try:
        # Extract the XGBoost model
        xgb_model = model_data.get('model')
        if xgb_model is None:
            print("❌ No model found in data")
            sys.exit(1)
        
        # Get other components
        label_encoders = model_data.get('label_encoders', {})
        feature_columns = model_data.get('feature_columns', [])
        service_multipliers = model_data.get('service_multipliers', {})
        
        print(f"✅ Extracted XGBoost model: {type(xgb_model)}")
        print(f"   Feature columns: {len(feature_columns)} features")
        print(f"   Label encoders: {list(label_encoders.keys())}")
        print(f"   Service multipliers: {service_multipliers}")
        
        # Step 3: Save clean version
        print("\n3️⃣ Saving cloud-compatible model...")
        
        # Create clean model data
        clean_model_data = {
            'model': xgb_model,
            'label_encoders': label_encoders,
            'feature_columns': feature_columns,
            'service_multipliers': service_multipliers
        }
        
        # Save with joblib using protocol 4
        output_file = 'xgboost_miami_model_cloud.pkl'
        joblib.dump(clean_model_data, output_file, protocol=4)
        
        print(f"✅ Cloud-compatible model saved to: {output_file}")
        
        # Test loading the new model
        print("\n4️⃣ Testing cloud-compatible model...")
        test_data = joblib.load(output_file)
        print("✅ Model loads successfully!")
        
        print("\n✅ Conversion complete!")
        print(f"\n📝 Next steps:")
        print(f"1. Replace the old model: mv {output_file} xgboost_miami_model.pkl")
        print("2. Commit and push: git add xgboost_miami_model.pkl && git commit -m 'Use cloud-compatible model' && git push")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()