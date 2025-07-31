"""
Example usage of the XGBoost Miami pricing model
"""

import sys
import os

# Add the deployment directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.xgboost_pricing_api import XGBoostPricingAPI, XGBoostModelWrapper

def test_api_interface():
    """Test the API interface"""
    print("="*60)
    print("XGBoost Miami Model - API Interface Test")
    print("="*60)
    
    # Initialize API
    api = XGBoostPricingAPI('xgboost_miami_model.pkl')
    
    # Get model info
    info = api.get_model_info()
    print(f"\nModel Status: {info['status']}")
    print(f"Model Type: {info['model_type']}")
    print(f"Services: {', '.join(info.get('services', []))}")
    
    # Test routes
    test_routes = [
        {
            'name': 'Miami Airport → South Beach',
            'pickup': (25.7959, -80.2870),
            'dropoff': (25.7907, -80.1300)
        },
        {
            'name': 'Downtown Miami → Wynwood',
            'pickup': (25.7617, -80.1918),
            'dropoff': (25.8103, -80.1934)
        },
        {
            'name': 'Brickell → Coral Gables',
            'pickup': (25.7617, -80.1918),
            'dropoff': (25.7211, -80.2685)
        }
    ]
    
    print("\n" + "="*60)
    print("PRICE PREDICTIONS")
    print("="*60)
    
    for route in test_routes:
        print(f"\n📍 {route['name']}")
        
        prices = api.predict_all_services(
            pickup_lat=route['pickup'][0],
            pickup_lng=route['pickup'][1],
            dropoff_lat=route['dropoff'][0],
            dropoff_lng=route['dropoff'][1],
            hour_of_day=14,
            day_of_week=2,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        for service, price in prices.items():
            print(f"   {service}: ${price:.2f}")


def test_wrapper_interface():
    """Test the GUI-compatible wrapper"""
    print("\n" + "="*60)
    print("GUI WRAPPER INTERFACE TEST")
    print("="*60)
    
    # Initialize wrapper
    model = XGBoostModelWrapper()
    
    # Test prediction
    print("\n📍 Test Route: Airport → South Beach (via wrapper)")
    prices = model.predict_all_services(
        pickup_lat=25.7959, 
        pickup_lng=-80.2870,
        dropoff_lat=25.7907, 
        dropoff_lng=-80.1300
    )
    
    for service, price in prices.items():
        print(f"   {service}: ${price:.2f}")


def test_single_prediction():
    """Test single service prediction"""
    print("\n" + "="*60)
    print("SINGLE SERVICE PREDICTION TEST")
    print("="*60)
    
    api = XGBoostPricingAPI('xgboost_miami_model.pkl')
    
    # Single UberX prediction
    price = api.predict_price(
        distance_km=19.8,
        pickup_lat=25.7959,
        pickup_lng=-80.2870,
        dropoff_lat=25.7907,
        dropoff_lng=-80.1300,
        hour_of_day=20,  # 8 PM
        day_of_week=5,   # Saturday
        traffic_level='heavy',
        weather_condition='rain'
    )
    
    print(f"\nUberX price (8PM Saturday, heavy traffic, rain): ${price:.2f}")


if __name__ == "__main__":
    print("🚀 XGBoost Miami Model - Deployment Test\n")
    
    try:
        test_api_interface()
        test_wrapper_interface()
        test_single_prediction()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()