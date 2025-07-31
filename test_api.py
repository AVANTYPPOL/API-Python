#!/usr/bin/env python
"""
API Testing Script for XGBoost Miami Pricing API
================================================
Tests all endpoints and verifies the API is working correctly
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_URL = "http://localhost:5000"  # Change this if your API runs on a different port

def test_health_endpoint():
    """Test the /health endpoint"""
    print("\n🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health Check Passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            print(f"   API Name: {data.get('api_name')}")
            print(f"   Model Type: {data.get('model_info', {}).get('model_type')}")
            print(f"   Accuracy: {data.get('model_info', {}).get('accuracy')}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def test_model_info_endpoint():
    """Test the /model/info endpoint"""
    print("\n🔍 Testing /model/info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Info Retrieved!")
            info = data.get('model_info', {})
            print(f"   Model Type: {info.get('model_type')}")
            print(f"   R² Score: {info.get('r2_score')}")
            print(f"   RMSE: {info.get('rmse')}")
            print(f"   Training Samples: {info.get('training_samples')}")
            print(f"   Services: {', '.join(info.get('services', []))}")
            return True
        else:
            print(f"❌ Model info failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_endpoint():
    """Test the /predict endpoint with various routes"""
    print("\n🔍 Testing /predict endpoint...")
    
    # Test cases with different Miami routes
    test_routes = [
        {
            "name": "Miami Airport → South Beach",
            "data": {
                "pickup_latitude": 25.7959,
                "pickup_longitude": -80.2870,
                "dropoff_latitude": 25.7907,
                "dropoff_longitude": -80.1300,
                "surge_multiplier": 1.0
            }
        },
        {
            "name": "Downtown Miami → Wynwood (Rush Hour)",
            "data": {
                "pickup_latitude": 25.7617,
                "pickup_longitude": -80.1918,
                "dropoff_latitude": 25.8103,
                "dropoff_longitude": -80.1934,
                "surge_multiplier": 1.5,
                "traffic_level": "heavy"
            }
        },
        {
            "name": "Brickell → Coral Gables",
            "data": {
                "pickup_latitude": 25.7614,
                "pickup_longitude": -80.1917,
                "dropoff_latitude": 25.7215,
                "dropoff_longitude": -80.2684,
                "weather_condition": "rain"
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_routes, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    predictions = data.get('predictions', {})
                    print("   ✅ Prediction successful!")
                    print(f"      Distance: {data.get('request_details', {}).get('distance_miles')} miles")
                    for service, price in predictions.items():
                        print(f"      {service}: ${price}")
                    
                    # Verify all 4 services are returned
                    expected_services = ['UberX', 'UberXL', 'Uber Premier', 'Premier SUV']
                    if all(service in predictions for service in expected_services):
                        print("   ✅ All 4 services returned correctly")
                    else:
                        print("   ❌ Missing some services!")
                        all_passed = False
                else:
                    print(f"   ❌ Prediction failed: {data.get('error')}")
                    all_passed = False
            else:
                print(f"   ❌ Request failed with status: {response.status_code}")
                print(f"      Response: {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
    
    return all_passed

def test_batch_predict_endpoint():
    """Test the /predict/batch endpoint"""
    print("\n🔍 Testing /predict/batch endpoint...")
    
    batch_data = {
        "rides": [
            {
                "pickup_latitude": 25.7959,
                "pickup_longitude": -80.2870,
                "dropoff_latitude": 25.7907,
                "dropoff_longitude": -80.1300
            },
            {
                "pickup_latitude": 25.7617,
                "pickup_longitude": -80.1918,
                "dropoff_latitude": 25.8103,
                "dropoff_longitude": -80.1934,
                "surge_multiplier": 2.0
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=batch_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Batch prediction successful!")
                print(f"   Total rides processed: {data.get('total_rides')}")
                results = data.get('results', [])
                for result in results:
                    print(f"   Ride {result.get('ride_index')}: {result.get('distance_miles')} miles")
                    predictions = result.get('predictions', {})
                    for service, price in predictions.items():
                        print(f"      {service}: ${price}")
                return True
            else:
                print(f"❌ Batch prediction failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_error_handling():
    """Test API error handling"""
    print("\n🔍 Testing error handling...")
    
    # Test with missing fields
    print("\n   Test 1: Missing required fields")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"pickup_latitude": 25.7959},  # Missing other required fields
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            print("   ✅ Correctly returned 400 for missing fields")
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test with invalid coordinates
    print("\n   Test 2: Invalid coordinates")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "pickup_latitude": 91,  # Invalid latitude
                "pickup_longitude": -80.2870,
                "dropoff_latitude": 25.7907,
                "dropoff_longitude": -80.1300
            },
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            print("   ✅ Correctly returned 400 for invalid coordinates")
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 XGBOOST MIAMI PRICING API TEST SUITE")
    print("="*60)
    print(f"Testing API at: {API_URL}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    print("\n⏳ Checking if API is running...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print("✅ API is running!")
    except:
        print("❌ API is not running!")
        print("\nPlease start the API with one of these commands:")
        print("  python app.py")
        print("  flask run")
        print("  gunicorn app:app")
        return
    
    # Run all tests
    tests_passed = 0
    total_tests = 5
    
    if test_health_endpoint():
        tests_passed += 1
    
    if test_model_info_endpoint():
        tests_passed += 1
    
    if test_predict_endpoint():
        tests_passed += 1
    
    if test_batch_predict_endpoint():
        tests_passed += 1
    
    if test_error_handling():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! API is working correctly!")
        print("\n🎉 The XGBoost model is successfully integrated and operational!")
    else:
        print(f"\n⚠️  Some tests failed ({total_tests - tests_passed} failures)")
        print("Please check the error messages above.")
    
    print("\n💡 Tips:")
    print("- Monitor the API logs for any warnings or errors")
    print("- Test with real coordinates from your application")
    print("- Check memory usage if processing many requests")
    print("- Consider adding API authentication for production")

if __name__ == "__main__":
    main()