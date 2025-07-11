#!/usr/bin/env python3
"""
🧪 Miami Uber Model Testing Script

Test script to validate the trained Ultimate Miami Uber price prediction model.
This script will run various test cases to ensure the model is working correctly
and producing reasonable predictions.

Usage:
    python test_model.py

Prerequisites:
    - Model must be trained first (run train_model.py)
    - ultimate_miami_model.pkl file must exist
"""

import os
import sys
import time
import traceback

def print_banner():
    """Print the welcome banner"""
    print("\n" + "="*60)
    print("🧪 MIAMI UBER MODEL TESTING")
    print("="*60)
    print("🔍 Validating trained model performance")
    print("🎯 Testing all 4 Uber services")
    print("📊 Checking prediction accuracy")
    print("="*60 + "\n")

def check_model_exists():
    """Check if the trained model file exists"""
    print("🔍 Checking for trained model...")
    
    if not os.path.exists('ultimate_miami_model.pkl'):
        print("❌ Model file 'ultimate_miami_model.pkl' not found")
        print("   Please run 'python train_model.py' first")
        return False
    
    print("✅ Model file found")
    return True

def load_model():
    """Load and initialize the trained model"""
    print("📥 Loading trained model...")
    
    try:
        from ultimate_miami_model import UltimateMiamiModel
        
        model = UltimateMiamiModel()
        model.load_model('ultimate_miami_model.pkl')
        
        print("✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return None

def test_basic_prediction(model):
    """Test basic prediction functionality"""
    print("\n🔧 Testing basic prediction...")
    
    try:
        # Simple test case
        prices = model.predict_all_services(
            distance_km=10,
            pickup_lat=25.7617,
            pickup_lng=-80.1918,
            dropoff_lat=25.7907,
            dropoff_lng=-80.1300,
            hour_of_day=14,
            day_of_week=1,
            surge_multiplier=1.0,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        # Check if we got 4 services
        expected_services = {'UberX', 'UberXL', 'Uber Premier', 'Premier SUV'}
        if set(prices.keys()) != expected_services:
            print(f"❌ Expected services {expected_services}, got {set(prices.keys())}")
            return False
        
        # Check if prices are reasonable
        for service, price in prices.items():
            if not (5 <= price <= 200):
                print(f"❌ Unreasonable price for {service}: ${price:.2f}")
                return False
        
        print("✅ Basic prediction test passed")
        print(f"   Sample prices: UberX ${prices['UberX']:.2f}, UberXL ${prices['UberXL']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Basic prediction test failed: {e}")
        traceback.print_exc()
        return False

def test_price_differentiation(model):
    """Test that different services have different prices"""
    print("\n💰 Testing price differentiation...")
    
    try:
        prices = model.predict_all_services(
            distance_km=15,
            pickup_lat=25.7617,
            pickup_lng=-80.1918,
            dropoff_lat=25.7907,
            dropoff_lng=-80.1300,
            hour_of_day=18,
            day_of_week=1,
            surge_multiplier=1.0,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        # Check price ordering: UberX < UberXL < Premier < Premier SUV
        if not (prices['UberX'] < prices['UberXL'] < prices['Uber Premier'] < prices['Premier SUV']):
            print("❌ Price ordering incorrect")
            print(f"   UberX: ${prices['UberX']:.2f}")
            print(f"   UberXL: ${prices['UberXL']:.2f}")
            print(f"   Premier: ${prices['Uber Premier']:.2f}")
            print(f"   Premier SUV: ${prices['Premier SUV']:.2f}")
            return False
        
        print("✅ Price differentiation test passed")
        print(f"   UberX: ${prices['UberX']:.2f} < UberXL: ${prices['UberXL']:.2f} < Premier: ${prices['Uber Premier']:.2f} < SUV: ${prices['Premier SUV']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Price differentiation test failed: {e}")
        return False

def test_distance_sensitivity(model):
    """Test that prices increase with distance"""
    print("\n📏 Testing distance sensitivity...")
    
    try:
        distances = [3, 10, 20, 30]
        prices_by_distance = {}
        
        for distance in distances:
            prices = model.predict_all_services(
                distance_km=distance,
                pickup_lat=25.7617,
                pickup_lng=-80.1918,
                dropoff_lat=25.7907,
                dropoff_lng=-80.1300,
                hour_of_day=14,
                day_of_week=1,
                surge_multiplier=1.0,
                traffic_level='moderate',
                weather_condition='clear'
            )
            prices_by_distance[distance] = prices['UberX']
        
        # Check if prices generally increase with distance
        for i in range(len(distances) - 1):
            curr_dist = distances[i]
            next_dist = distances[i + 1]
            
            if prices_by_distance[curr_dist] >= prices_by_distance[next_dist]:
                print(f"❌ Price didn't increase from {curr_dist}km to {next_dist}km")
                print(f"   ${prices_by_distance[curr_dist]:.2f} vs ${prices_by_distance[next_dist]:.2f}")
                return False
        
        print("✅ Distance sensitivity test passed")
        for distance in distances:
            print(f"   {distance}km: ${prices_by_distance[distance]:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Distance sensitivity test failed: {e}")
        return False

def test_surge_impact(model):
    """Test that surge multiplier affects prices"""
    print("\n⚡ Testing surge impact...")
    
    try:
        # Test with no surge vs 2x surge
        base_prices = model.predict_all_services(
            distance_km=15,
            pickup_lat=25.7617,
            pickup_lng=-80.1918,
            dropoff_lat=25.7907,
            dropoff_lng=-80.1300,
            hour_of_day=18,
            day_of_week=1,
            surge_multiplier=1.0,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        surge_prices = model.predict_all_services(
            distance_km=15,
            pickup_lat=25.7617,
            pickup_lng=-80.1918,
            dropoff_lat=25.7907,
            dropoff_lng=-80.1300,
            hour_of_day=18,
            day_of_week=1,
            surge_multiplier=2.0,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        # Check if surge prices are higher
        for service in base_prices:
            if surge_prices[service] <= base_prices[service]:
                print(f"❌ Surge didn't increase price for {service}")
                print(f"   Base: ${base_prices[service]:.2f}, Surge: ${surge_prices[service]:.2f}")
                return False
        
        print("✅ Surge impact test passed")
        print(f"   UberX: ${base_prices['UberX']:.2f} → ${surge_prices['UberX']:.2f} (2x surge)")
        return True
        
    except Exception as e:
        print(f"❌ Surge impact test failed: {e}")
        return False

def test_miami_routes(model):
    """Test specific Miami routes"""
    print("\n🏖️  Testing Miami-specific routes...")
    
    test_routes = [
        {
            "name": "Miami Airport → South Beach",
            "distance": 18,
            "pickup_lat": 25.7959, "pickup_lng": -80.2870,
            "dropoff_lat": 25.7907, "dropoff_lng": -80.1300,
            "expected_range": (25, 50)
        },
        {
            "name": "Downtown → Coral Gables",
            "distance": 12,
            "pickup_lat": 25.7617, "pickup_lng": -80.1918,
            "dropoff_lat": 25.7479, "dropoff_lng": -80.2571,
            "expected_range": (18, 35)
        },
        {
            "name": "Brickell → Wynwood",
            "distance": 8,
            "pickup_lat": 25.7617, "pickup_lng": -80.1918,
            "dropoff_lat": 25.8010, "dropoff_lng": -80.1990,
            "expected_range": (15, 28)
        }
    ]
    
    all_passed = True
    
    for route in test_routes:
        try:
            prices = model.predict_all_services(
                distance_km=route["distance"],
                pickup_lat=route["pickup_lat"],
                pickup_lng=route["pickup_lng"],
                dropoff_lat=route["dropoff_lat"],
                dropoff_lng=route["dropoff_lng"],
                hour_of_day=14,
                day_of_week=1,
                surge_multiplier=1.0,
                traffic_level='moderate',
                weather_condition='clear'
            )
            
            uberx_price = prices["UberX"]
            min_exp, max_exp = route["expected_range"]
            
            if min_exp <= uberx_price <= max_exp:
                status = "✅"
            else:
                status = "⚠️"
                all_passed = False
            
            print(f"{status} {route['name']}: ${uberx_price:.2f} (expected ${min_exp}-${max_exp})")
            
        except Exception as e:
            print(f"❌ {route['name']}: Error - {e}")
            all_passed = False
    
    if all_passed:
        print("✅ Miami routes test passed")
    else:
        print("⚠️  Some routes outside expected range (may still be acceptable)")
    
    return all_passed

def test_performance(model):
    """Test prediction performance/speed"""
    print("\n⚡ Testing prediction performance...")
    
    try:
        # Test prediction speed
        start_time = time.time()
        
        for i in range(10):
            prices = model.predict_all_services(
                distance_km=10 + i,
                pickup_lat=25.7617,
                pickup_lng=-80.1918,
                dropoff_lat=25.7907,
                dropoff_lng=-80.1300,
                hour_of_day=14,
                day_of_week=1,
                surge_multiplier=1.0,
                traffic_level='moderate',
                weather_condition='clear'
            )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        if avg_time > 1.0:
            print(f"⚠️  Prediction time slow: {avg_time:.3f}s per prediction")
        else:
            print(f"✅ Prediction speed good: {avg_time:.3f}s per prediction")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main testing pipeline"""
    print_banner()
    
    # Step 1: Check model exists
    if not check_model_exists():
        return False
    
    # Step 2: Load model
    model = load_model()
    if model is None:
        return False
    
    # Step 3: Run all tests
    tests = [
        ("Basic Prediction", test_basic_prediction),
        ("Price Differentiation", test_price_differentiation),
        ("Distance Sensitivity", test_distance_sensitivity),
        ("Surge Impact", test_surge_impact),
        ("Miami Routes", test_miami_routes),
        ("Performance", test_performance)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func(model):
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TESTING SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Model is ready for production.")
        success_rate = 100
    else:
        success_rate = (passed_tests / total_tests) * 100
        print(f"⚠️  {total_tests - passed_tests} test(s) failed. Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("✅ Model performance is acceptable for production use")
    else:
        print("❌ Model needs improvement before production use")
    
    print("\n📚 Next steps:")
    if passed_tests == total_tests:
        print("   1. Integrate model into your application")
        print("   2. Monitor performance in production")
        print("   3. Collect feedback and retrain monthly")
    else:
        print("   1. Review failed tests and investigate issues")
        print("   2. Check training data quality")
        print("   3. Consider retraining with more data")
    
    print("="*60 + "\n")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 