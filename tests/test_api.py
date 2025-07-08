#!/usr/bin/env python3
"""
Test script for the Ride Share Pricing API
"""

import requests
import json
import time
import sys
from datetime import datetime

class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self):
        """Test the health check endpoint"""
        print("🏥 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_model_info(self):
        """Test the model info endpoint"""
        print("🤖 Testing model info...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info: {data['model_type']}")
                print(f"   Accuracy: {data['accuracy']}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_prediction(self):
        """Test the prediction endpoint"""
        print("🎯 Testing price prediction...")
        
        # Test data: Downtown Miami to Miami Beach
        test_data = {
            "pickup_lat": 25.7617,
            "pickup_lng": -80.1918,
            "dropoff_lat": 25.7907,
            "dropoff_lng": -80.1300,
            "distance_km": 8.5,
            "hour_of_day": 19,
            "day_of_week": 5,
            "surge_multiplier": 1.2,
            "traffic_level": "moderate",
            "weather_condition": "clear"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    prediction = data["prediction"]
                    print(f"✅ Prediction successful!")
                    print(f"   Base price: ${prediction['base_price']:.2f}")
                    print(f"   Price per km: ${prediction['price_per_km']:.2f}")
                    print(f"   Service prices:")
                    for service, price in prediction['service_prices'].items():
                        print(f"     {service}: ${price:.2f}")
                    return True
                else:
                    print(f"❌ Prediction failed: {data}")
                    return False
            else:
                print(f"❌ Prediction request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_invalid_prediction(self):
        """Test prediction with invalid data"""
        print("⚠️  Testing invalid prediction...")
        
        invalid_data = {
            "pickup_lat": 200,  # Invalid latitude
            "pickup_lng": -80.1918,
            "dropoff_lat": 25.7907,
            "dropoff_lng": -80.1300,
            "distance_km": 8.5
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                print("✅ Invalid data properly rejected")
                return True
            else:
                print(f"❌ Invalid data not rejected: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Invalid prediction test error: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        print("📊 Testing batch prediction...")
        
        batch_data = {
            "rides": [
                {
                    "pickup_lat": 25.7617,
                    "pickup_lng": -80.1918,
                    "dropoff_lat": 25.7907,
                    "dropoff_lng": -80.1300,
                    "distance_km": 8.5
                },
                {
                    "pickup_lat": 25.7959,
                    "pickup_lng": -80.2870,
                    "dropoff_lat": 25.7617,
                    "dropoff_lng": -80.1918,
                    "distance_km": 15.0
                }
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["success"]:
                    print(f"✅ Batch prediction successful!")
                    print(f"   Processed {len(data['predictions'])} rides")
                    return True
                else:
                    print(f"❌ Batch prediction failed: {data}")
                    return False
            else:
                print(f"❌ Batch prediction request failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_performance(self):
        """Test API performance with multiple requests"""
        print("⚡ Testing performance...")
        
        test_data = {
            "pickup_lat": 25.7617,
            "pickup_lng": -80.1918,
            "dropoff_lat": 25.7907,
            "dropoff_lng": -80.1300,
            "distance_km": 8.5
        }
        
        num_requests = 10
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=test_data,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code != 200:
                    print(f"❌ Request {i+1} failed: {response.status_code}")
                    return False
            except Exception as e:
                print(f"❌ Request {i+1} error: {e}")
                return False
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        
        print(f"✅ Performance test completed!")
        print(f"   {num_requests} requests in {total_time:.2f}s")
        print(f"   Average response time: {avg_time:.3f}s")
        print(f"   Requests per second: {num_requests/total_time:.2f}")
        
        return True
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting API tests...")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Model Info", self.test_model_info),
            ("Price Prediction", self.test_prediction),
            ("Invalid Data", self.test_invalid_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Performance", self.test_performance)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                failed += 1
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All tests passed! API is ready for production.")
            return True
        else:
            print("⚠️  Some tests failed. Check the API before deployment.")
            return False

def main():
    """Main function"""
    base_url = "http://localhost:5000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"🧪 Testing API at: {base_url}")
    
    tester = APITester(base_url)
    
    # First, wait for API to be ready
    print("⏳ Waiting for API to be ready...")
    for i in range(10):
        if tester.test_health_check():
            break
        time.sleep(2)
    else:
        print("❌ API not ready after 20 seconds")
        sys.exit(1)
    
    # Run all tests
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 