"""
Simple Multi-Service API
=======================

Easy-to-use API for getting prices for all 4 Uber services.
"""

import os
from models.calibrated_miami_model import CalibratedMiamiModel
from datetime import datetime

# Load API keys from environment variables
if 'GOOGLE_MAPS_API_KEY' not in os.environ:
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
if 'WEATHER_API_KEY' not in os.environ:
    raise ValueError("WEATHER_API_KEY environment variable not set")

class SimpleMultiServiceAPI:
    """Simple API for multi-service predictions"""
    
    def __init__(self):
        self.model = CalibratedMiamiModel()
        
        # Try to load calibrated model, if not available train it
        if not self.model.load_model('calibrated_miami_model.pkl'):
            print("🔄 Training calibrated model...")
            self.model.train_model()
            self.model.save_model()
        
        # Removed real-world adjustment for direct Uber comparison
        # self.real_world_adjustment = 0.75  # 25% reduction to match real prices
        
        print("✅ Calibrated Multi-Service API Ready!")
    
    def get_all_prices(self, distance_km, pickup_lat, pickup_lng, 
                      dropoff_lat, dropoff_lng, surge=1.0):
        """
        Get prices for all 4 services
        
        Returns dict with service prices
        """
        now = datetime.now()
        
        # Get base predictions
        prices = self.model.predict_all_services(
            distance_km=distance_km,
            pickup_lat=pickup_lat,
            pickup_lng=pickup_lng,
            dropoff_lat=dropoff_lat,
            dropoff_lng=dropoff_lng,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            surge_multiplier=surge
        )
        
        # Removed real-world calibration for direct comparison
        # for service in prices:
        #     prices[service] = round(prices[service] * self.real_world_adjustment, 2)
        
        # Removed competitive pricing for direct Uber comparison
        competitive_multipliers = {
            'uberx': 1.0,        # No discount - direct comparison
            'uber_xl': 1.0,      # No discount - direct comparison
            'uber_premier': 1.0,  # No discount - direct comparison
            'premier_suv': 1.0    # No discount - direct comparison
        }
        
        # Format response with competitive prices
        return {
            'success': True,
            'distance_km': distance_km,
            'distance_miles': distance_km * 0.621371,
            'surge': surge,
            'pricing_strategy': 'direct_comparison',
            'services': {
                'UberX': {
                    'price': round(prices['uberx'] * competitive_multipliers['uberx'], 2),
                    'model_prediction': prices['uberx'],
                    'comparison': 'Direct model prediction',
                    'description': 'Affordable rides for up to 4'
                },
                'UberXL': {
                    'price': round(prices['uber_xl'] * competitive_multipliers['uber_xl'], 2),
                    'model_prediction': prices['uber_xl'],
                    'comparison': 'Direct model prediction',
                    'description': 'Rides for groups up to 6'
                },
                'Uber Premier': {
                    'price': round(prices['uber_premier'] * competitive_multipliers['uber_premier'], 2),
                    'model_prediction': prices['uber_premier'],
                    'comparison': 'Direct model prediction',
                    'description': 'Premium rides'
                },
                'Premier SUV': {
                    'price': round(prices['premier_suv'] * competitive_multipliers['premier_suv'], 2),
                    'model_prediction': prices['premier_suv'],
                    'comparison': 'Direct model prediction',
                    'description': 'Premium SUVs for up to 6'
                }
            }
        }

def demo():
    """Demo the API"""
    print("\n🚗 Multi-Service Pricing Demo")
    print("="*50)
    
    api = SimpleMultiServiceAPI()
    
    # Test cases
    tests = [
        {
            'name': 'Short Trip (5km)',
            'distance': 5,
            'pickup': (25.7617, -80.1918),
            'dropoff': (25.7750, -80.1850)
        },
        {
            'name': 'Airport Trip (15km)', 
            'distance': 15,
            'pickup': (25.7959, -80.2870),
            'dropoff': (25.7907, -80.1300)
        },
        {
            'name': 'Long Trip (30km)',
            'distance': 30,
            'pickup': (25.7617, -80.1918),
            'dropoff': (26.0000, -80.1500)
        }
    ]
    
    for test in tests:
        print(f"\n📍 {test['name']}")
        print("-"*30)
        
        result = api.get_all_prices(
            distance_km=test['distance'],
            pickup_lat=test['pickup'][0],
            pickup_lng=test['pickup'][1],
            dropoff_lat=test['dropoff'][0],
            dropoff_lng=test['dropoff'][1],
            surge=1.0
        )
        
        for service, data in result['services'].items():
            price_per_km = data['price'] / test['distance']
            print(f"{service:15} ${data['price']:.2f} (${price_per_km:.2f}/km)")

if __name__ == "__main__":
    demo() 