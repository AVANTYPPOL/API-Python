#!/usr/bin/env python3
"""Simple test to verify service names are in internal format"""

# Test the exact code that runs in Dockerfile validation
try:
    from xgboost_pricing_api import XGBoostPricingAPI
    
    print("Testing model load and service names...")
    api = XGBoostPricingAPI('xgboost_miami_model.pkl')
    print(f"Model loaded: {api.is_loaded}")
    
    # Test prediction
    result = api.predict_all_services(
        pickup_lat=25.7959,
        pickup_lng=-80.2870,
        dropoff_lat=25.7617,
        dropoff_lng=-80.1918
    )
    
    print(f"\nPrediction result: {result}")
    
    # Check service names
    expected_services = ['PREMIER', 'SUV_PREMIER', 'UBERX', 'UBERXL']
    for service in expected_services:
        assert service in result, f"Missing service: {service}"
    
    # Check order
    result_keys = list(result.keys())
    print(f"\nService order: {result_keys}")
    assert result_keys == expected_services, f"Order mismatch! Expected {expected_services}, got {result_keys}"
    
    print("\n✅ All tests passed! Service names are correct.")
    print("   Order: PREMIER, SUV_PREMIER, UBERX, UBERXL")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)