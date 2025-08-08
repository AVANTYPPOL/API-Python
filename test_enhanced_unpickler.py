#!/usr/bin/env python3
"""Test the enhanced unpickler with various scenarios"""

import numpy as np
import sys

# Test 1: Simulate numpy 1.x environment
print("="*60)
print("TEST 1: Simulating numpy 1.x environment")
print("="*60)

# Remove numpy._core if it exists
original_core = getattr(np, '_core', None)
if hasattr(np, '_core'):
    delattr(np, '_core')
    print("✅ Removed numpy._core to simulate numpy 1.x")

try:
    from xgboost_miami_model import XGBoostMiamiModel
    
    print("\n🔄 Testing model loading with enhanced unpickler...")
    model = XGBoostMiamiModel()
    model.load_model('xgboost_miami_model.pkl')
    
    print("\n✅ Model loaded successfully!")
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    predictions = model.predict_all_services(
        pickup_lat=25.7959,
        pickup_lng=-80.2870,
        dropoff_lat=25.7617,
        dropoff_lng=-80.1918
    )
    
    print("\nPredictions (Airport to South Beach):")
    for service, price in predictions.items():
        print(f"  {service}: ${price:.2f}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore numpy._core
    if original_core is not None:
        np._core = original_core

# Test 2: Direct API test
print("\n" + "="*60)
print("TEST 2: Testing through API wrapper")
print("="*60)

try:
    from xgboost_pricing_api import XGBoostPricingAPI
    
    api = XGBoostPricingAPI('xgboost_miami_model.pkl')
    
    if api.is_loaded:
        print("✅ API loaded model successfully!")
        
        predictions = api.predict_all_services(
            pickup_lat=25.7751,
            pickup_lng=-80.1900,
            dropoff_lat=25.6892,
            dropoff_lng=-80.1613
        )
        
        print("\nPredictions (Downtown to Key Biscayne):")
        for service, price in predictions.items():
            print(f"  {service}: ${price:.2f}")
    else:
        print("❌ API failed to load model")
        
except Exception as e:
    print(f"❌ API Error: {e}")
    import traceback
    traceback.print_exc()