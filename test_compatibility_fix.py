#!/usr/bin/env python3
"""Test the numpy compatibility fix"""

# First, let's simulate the cloud environment by temporarily hiding numpy._core
import numpy as np

# Backup the original _core if it exists
original_core = getattr(np, '_core', None)

# Remove _core to simulate numpy 1.x environment
if hasattr(np, '_core'):
    delattr(np, '_core')
    print(f"✅ Simulating numpy 1.x environment (removed numpy._core)")

try:
    # Now try to load the model
    from xgboost_pricing_api import XGBoostPricingAPI
    
    print("\n🔄 Testing model loading with compatibility fix...")
    api = XGBoostPricingAPI('xgboost_miami_model.pkl')
    
    if api.is_loaded:
        print("\n✅ Model loaded successfully!")
        
        # Test prediction
        print("\n🧪 Testing prediction...")
        predictions = api.predict_all_services(
            pickup_lat=25.7959,
            pickup_lng=-80.2870,
            dropoff_lat=25.7617,
            dropoff_lng=-80.1918
        )
        
        print("\nPredictions (Airport to South Beach):")
        for service, price in predictions.items():
            print(f"  {service}: ${price:.2f}")
        
        print("\n✅ Compatibility fix works!")
    else:
        print("\n❌ Model failed to load")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore numpy._core if it was originally there
    if original_core is not None:
        np._core = original_core
        print("\n✅ Restored numpy._core")