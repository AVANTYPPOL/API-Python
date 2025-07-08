"""
Real-Time Uber Price Prediction API
==================================

This wrapper integrates the production ML model with real-time APIs:
- Google Maps API for traffic data
- OpenWeatherMap API for weather data
- Falls back gracefully when APIs are unavailable

Usage:
- With APIs: Enhanced real-time predictions
- Without APIs: Basic predictions using defaults

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from production_uber_model import ProductionUberPriceModel
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🌐 REAL-TIME UBER PRICE PREDICTION API")
print("=" * 70)
print("🔗 Optional: Google Maps + Weather APIs")
print("🤖 Powered by: Production ML Model")
print("=" * 70)

class RealTimePricingAPI:
    """
    Real-time pricing API that optionally uses external APIs
    for enhanced predictions with current traffic and weather
    """
    
    def __init__(self, 
                 google_api_key=None, 
                 weather_api_key=None,
                 model_path='production_uber_model.pkl'):
        
        # API keys (optional)
        self.google_api_key = google_api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        self.weather_api_key = weather_api_key or os.getenv('OPENWEATHER_API_KEY')
        
        # Initialize ML model
        self.ml_model = ProductionUberPriceModel()
        
        # Try to load trained model
        try:
            self.ml_model.load_model(model_path)
            print("✅ ML model loaded successfully")
        except:
            print("⚠️  No pre-trained model found. Train model first!")
            self.ml_model = None
        
        # API status
        self.google_api_available = bool(self.google_api_key)
        self.weather_api_available = bool(self.weather_api_key)
        
        print(f"🗺️  Google Maps API: {'✅ Available' if self.google_api_available else '❌ Not configured'}")
        print(f"🌤️  Weather API: {'✅ Available' if self.weather_api_available else '❌ Not configured'}")
    
    def get_real_time_traffic(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """
        Get real-time traffic data from Google Maps API
        
        Returns:
            tuple: (traffic_level, distance_km) where traffic_level is str and distance_km is float
                  Returns ('light', None) if API unavailable
        """
        if not self.google_api_available:
            return 'light', None  # Default, no distance
        
        try:
            # Google Maps Directions API
            url = "https://maps.googleapis.com/maps/api/directions/json"
            
            params = {
                'origin': f"{pickup_lat},{pickup_lng}",
                'destination': f"{dropoff_lat},{dropoff_lng}",
                'departure_time': 'now',
                'traffic_model': 'best_guess',
                'key': self.google_api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['status'] == 'OK' and data['routes']:
                    # Get duration in traffic vs normal duration
                    route = data['routes'][0]['legs'][0]
                    
                    normal_duration = route['duration']['value']  # seconds
                    traffic_duration = route.get('duration_in_traffic', {}).get('value', normal_duration)
                    
                    # Get actual driving distance
                    distance_meters = route['distance']['value']
                    distance_km = distance_meters / 1000
                    
                    # Calculate traffic ratio
                    traffic_ratio = traffic_duration / normal_duration
                    
                    # Classify traffic level
                    if traffic_ratio < 1.2:
                        traffic_level = 'light'
                    elif traffic_ratio < 1.5:
                        traffic_level = 'moderate'
                    else:
                        traffic_level = 'heavy'
                    
                    return traffic_level, distance_km
            
            return 'light', None  # Default if API call fails
            
        except Exception as e:
            print(f"⚠️  Google Maps API error: {e}")
            return 'light', None  # Default
    
    def get_real_time_weather(self, lat, lng):
        """
        Get real-time weather data from OpenWeatherMap API
        
        Returns:
            str: Weather condition or 'clear' if API unavailable
        """
        if not self.weather_api_available:
            return 'clear'  # Default
        
        try:
            # OpenWeatherMap Current Weather API
            url = "http://api.openweathermap.org/data/2.5/weather"
            
            params = {
                'lat': lat,
                'lon': lng,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Get main weather condition
                weather_main = data['weather'][0]['main'].lower()
                
                # Map to our categories
                weather_mapping = {
                    'clear': 'clear',
                    'clouds': 'clouds',
                    'rain': 'rain',
                    'drizzle': 'rain',
                    'thunderstorm': 'rain',
                    'snow': 'snow',
                    'mist': 'clouds',
                    'fog': 'clouds'
                }
                
                return weather_mapping.get(weather_main, 'clear')
            
            return 'clear'  # Default if API call fails
            
        except Exception as e:
            print(f"⚠️  Weather API error: {e}")
            return 'clear'  # Default
    
    def predict_price(self, 
                     pickup_lat, pickup_lng, 
                     dropoff_lat, dropoff_lng,
                     distance_km=None,
                     hour_of_day=None,
                     day_of_week=None,
                     surge_multiplier=1.0,
                     use_real_time=True):
        """
        Predict price with optional real-time enhancements
        
        Args:
            pickup_lat (float): Pickup latitude
            pickup_lng (float): Pickup longitude
            dropoff_lat (float): Dropoff latitude
            dropoff_lng (float): Dropoff longitude
            distance_km (float, optional): Distance in km (will calculate if not provided)
            hour_of_day (int, optional): Hour of day (will use current if not provided)
            day_of_week (int, optional): Day of week (will use current if not provided)
            surge_multiplier (float): Surge multiplier (default 1.0)
            use_real_time (bool): Whether to fetch real-time data
            
        Returns:
            dict: Prediction results with metadata
        """
        
        if self.ml_model is None:
            return {
                'error': 'ML model not available. Train model first!',
                'price': None
            }
        
        # Get current time if not provided
        now = datetime.now()
        if hour_of_day is None:
            hour_of_day = now.hour
        if day_of_week is None:
            day_of_week = now.weekday()  # 0=Monday
        
        # Calculate distance if not provided
        if distance_km is None:
            distance_km = self._calculate_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        
        # Default values
        traffic_level = 'light'
        weather_condition = 'clear'
        data_source = 'default'
        
        if use_real_time:
            if self.google_api_available or self.weather_api_available:
                print("🔄 Fetching real-time data...")
                
                # Get traffic data and actual driving distance
                if self.google_api_available:
                    traffic_level, google_distance = self.get_real_time_traffic(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
                    data_source = 'real-time'
                    
                    # Use Google Maps distance if available (more accurate than Haversine)
                    if google_distance is not None:
                        distance_km = google_distance
                        print(f"📏 Using Google Maps distance: {distance_km:.2f} km")
                    else:
                        print(f"📏 Using Haversine distance: {distance_km:.2f} km")
                
                # Get weather data (use pickup location)
                if self.weather_api_available:
                    weather_condition = self.get_real_time_weather(pickup_lat, pickup_lng)
                    data_source = 'real-time'
        
        # Make prediction
        try:
            price = self.ml_model.predict_price(
                distance_km=distance_km,
                pickup_lat=pickup_lat,
                pickup_lng=pickup_lng,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                surge_multiplier=surge_multiplier,
                traffic_level=traffic_level,
                weather_condition=weather_condition
            )
            
            # Calculate competitive price (20% cheaper than prediction)
            competitive_price = price * 0.8
            
            return {
                'price': round(price, 2),
                'competitive_price': round(competitive_price, 2),
                'distance_km': round(distance_km, 2),
                'price_per_km': round(price / distance_km, 2),
                'traffic_level': traffic_level,
                'weather_condition': weather_condition,
                'surge_multiplier': surge_multiplier,
                'data_source': data_source,
                'timestamp': now.isoformat(),
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Prediction error: {str(e)}',
                'price': None,
                'success': False
            }
    
    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two points using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        
        return r * c
    
    def batch_predict(self, trip_requests, use_real_time=True):
        """
        Predict prices for multiple trips
        
        Args:
            trip_requests (list): List of trip dictionaries
            use_real_time (bool): Whether to use real-time data
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i, trip in enumerate(trip_requests):
            print(f"🔄 Processing trip {i+1}/{len(trip_requests)}")
            
            result = self.predict_price(
                pickup_lat=trip['pickup_lat'],
                pickup_lng=trip['pickup_lng'],
                dropoff_lat=trip['dropoff_lat'],
                dropoff_lng=trip['dropoff_lng'],
                distance_km=trip.get('distance_km'),
                hour_of_day=trip.get('hour_of_day'),
                day_of_week=trip.get('day_of_week'),
                surge_multiplier=trip.get('surge_multiplier', 1.0),
                use_real_time=use_real_time
            )
            
            result['trip_id'] = i + 1
            results.append(result)
        
        return results
    
    def get_pricing_summary(self, results):
        """Generate summary statistics from prediction results"""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful predictions'}
        
        prices = [r['price'] for r in successful_results]
        competitive_prices = [r['competitive_price'] for r in successful_results]
        
        return {
            'total_trips': len(results),
            'successful_predictions': len(successful_results),
            'average_price': round(np.mean(prices), 2),
            'average_competitive_price': round(np.mean(competitive_prices), 2),
            'price_range': {
                'min': round(min(prices), 2),
                'max': round(max(prices), 2)
            },
            'total_savings': round(sum(prices) - sum(competitive_prices), 2),
            'average_savings_per_trip': round(np.mean([p - c for p, c in zip(prices, competitive_prices)]), 2)
        }

def main():
    """Demo the real-time pricing API"""
    print("🚀 Starting Real-Time Pricing API Demo")
    
    # Initialize API (will work with or without API keys)
    api = RealTimePricingAPI()
    
    if api.ml_model is None:
        print("❌ Please train the ML model first by running: python production_uber_model.py")
        return
    
    # Test single prediction
    print("\n🎯 Single Prediction Test:")
    print("-" * 40)
    
    result = api.predict_price(
        pickup_lat=25.7617,   # Downtown Miami
        pickup_lng=-80.1918,
        dropoff_lat=25.7907,  # Miami Beach
        dropoff_lng=-80.1300,
        use_real_time=True
    )
    
    if result['success']:
        print(f"🏁 Trip: Downtown Miami → Miami Beach")
        print(f"📏 Distance: {result['distance_km']} km")
        print(f"💰 Predicted Price: ${result['price']}")
        print(f"🎯 Competitive Price: ${result['competitive_price']} (20% discount)")
        print(f"📊 Price per km: ${result['price_per_km']}/km")
        print(f"🚦 Traffic: {result['traffic_level']}")
        print(f"🌤️  Weather: {result['weather_condition']}")
        print(f"📡 Data Source: {result['data_source']}")
    else:
        print(f"❌ Prediction failed: {result['error']}")
    
    # Test batch predictions
    print("\n🎯 Batch Prediction Test:")
    print("-" * 40)
    
    test_trips = [
        {
            'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
            'dropoff_lat': 25.7907, 'dropoff_lng': -80.1300,
            'surge_multiplier': 1.0
        },
        {
            'pickup_lat': 25.7959, 'pickup_lng': -80.2870,  # Airport
            'dropoff_lat': 25.7617, 'dropoff_lng': -80.1918,  # Downtown
            'surge_multiplier': 1.2
        },
        {
            'pickup_lat': 25.7617, 'pickup_lng': -80.1918,
            'dropoff_lat': 25.7700, 'dropoff_lng': -80.1850,
            'surge_multiplier': 1.0
        }
    ]
    
    batch_results = api.batch_predict(test_trips, use_real_time=True)
    summary = api.get_pricing_summary(batch_results)
    
    print(f"\n📊 Batch Results Summary:")
    print(f"   Total trips: {summary['total_trips']}")
    print(f"   Successful: {summary['successful_predictions']}")
    print(f"   Average price: ${summary['average_price']}")
    print(f"   Average competitive price: ${summary['average_competitive_price']}")
    print(f"   Total savings: ${summary['total_savings']}")
    print(f"   Average savings per trip: ${summary['average_savings_per_trip']}")
    
    print("\n✅ Real-time pricing API demo complete!")

if __name__ == "__main__":
    main() 