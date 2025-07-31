"""
XGBoost Pricing API Interface
============================

API wrapper for the XGBoost Miami model to integrate with the GUI.
Provides compatibility with existing GUI interfaces while using
the superior XGBoost model for predictions.
"""

import os
import sys
from dotenv import load_dotenv
import requests
import googlemaps

# Load environment variables
load_dotenv()

# Add the project root and specific paths to avoid __init__.py issues
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_path = os.path.join(project_root, 'src', 'models')
sys.path.append(project_root)
sys.path.append(models_path)

from xgboost_miami_model import XGBoostMiamiModel
import joblib
import numpy as np
from datetime import datetime

class XGBoostPricingAPI:
    """
    API wrapper for XGBoost model with GUI compatibility
    """
    
    def __init__(self, model_path='xgboost_miami_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # Initialize API clients
        self.gmaps = None
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        
        # Initialize Google Maps if API key is available
        google_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if google_api_key:
            try:
                self.gmaps = googlemaps.Client(key=google_api_key)
                print("✅ Google Maps API initialized")
            except Exception as e:
                print(f"⚠️  Google Maps API initialization failed: {e}")
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the XGBoost model"""
        try:
            if os.path.exists(self.model_path):
                self.model = XGBoostMiamiModel()
                self.model.load_model(self.model_path)
                self.is_loaded = True
                print("✅ XGBoost model loaded successfully")
            else:
                print(f"⚠️  Model file not found: {self.model_path}")
                print("   Training new model...")
                self.train_new_model()
        except Exception as e:
            print(f"❌ Failed to load XGBoost model: {e}")
            self.is_loaded = False
    
    def train_new_model(self):
        """Train a new model if none exists"""
        try:
            self.model = XGBoostMiamiModel()
            self.model.train()
            self.model.save_model(self.model_path)
            self.is_loaded = True
            print("✅ New XGBoost model trained and saved")
        except Exception as e:
            print(f"❌ Failed to train new model: {e}")
            self.is_loaded = False
    
    def get_real_distance(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """Get real driving distance from Google Maps API"""
        if self.gmaps:
            try:
                result = self.gmaps.distance_matrix(
                    origins=[(pickup_lat, pickup_lng)],
                    destinations=[(dropoff_lat, dropoff_lng)],
                    mode="driving",
                    units="metric"
                )
                
                if result['status'] == 'OK' and result['rows'][0]['elements'][0]['status'] == 'OK':
                    distance_m = result['rows'][0]['elements'][0]['distance']['value']
                    distance_km = distance_m / 1000
                    duration_s = result['rows'][0]['elements'][0]['duration']['value']
                    
                    # Check for traffic by comparing to duration_in_traffic
                    traffic_level = 'moderate'
                    if 'duration_in_traffic' in result['rows'][0]['elements'][0]:
                        traffic_duration = result['rows'][0]['elements'][0]['duration_in_traffic']['value']
                        traffic_ratio = traffic_duration / duration_s
                        if traffic_ratio > 1.3:
                            traffic_level = 'heavy'
                        elif traffic_ratio < 0.9:
                            traffic_level = 'light'
                    
                    return distance_km, traffic_level
            except Exception as e:
                print(f"⚠️  Google Maps API error: {e}")
        
        return None, None
    
    def get_current_weather(self, lat, lng):
        """Get current weather from OpenWeatherMap API"""
        if self.weather_api_key:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': lat,
                    'lon': lng,
                    'appid': self.weather_api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Map weather conditions
                    weather_main = data['weather'][0]['main'].lower()
                    if weather_main in ['rain', 'drizzle', 'thunderstorm']:
                        return 'rain'
                    elif weather_main in ['snow', 'sleet']:
                        return 'snow'
                    elif weather_main in ['fog', 'mist', 'haze']:
                        return 'fog'
                    else:
                        return 'clear'
            except Exception as e:
                print(f"⚠️  Weather API error: {e}")
        
        return 'clear'
    
    def predict_all_services(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                            hour_of_day=None, day_of_week=None, 
                            traffic_level='moderate', weather_condition='clear'):
        """
        Predict prices for all service types
        Compatible with GUI interface
        """
        if not self.is_loaded:
            # Return fallback prices if model not loaded
            return {
                'UBERX': 25.50,
                'UBERXL': 35.75,
                'PREMIER': 45.20,
                'SUV_PREMIER': 55.90
            }
        
        # Use current time if not provided
        if hour_of_day is None:
            hour_of_day = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        # Get real-time data from APIs if available
        real_distance, real_traffic = self.get_real_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        if real_traffic:
            traffic_level = real_traffic
            print(f"📍 Using Google Maps data - Traffic: {traffic_level}")
        
        real_weather = self.get_current_weather(pickup_lat, pickup_lng)
        if real_weather != 'clear' or self.weather_api_key:
            weather_condition = real_weather
            print(f"🌤️  Using real weather data - Condition: {weather_condition}")
        
        try:
            # Get predictions for all services
            results = self.model.predict_all_services(
                pickup_lat=pickup_lat,
                pickup_lng=pickup_lng,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                traffic_level=traffic_level,
                weather_condition=weather_condition
            )
            
            # Convert to GUI expected format
            gui_results = {
                'UberX': results['UBERX'],
                'UberXL': results['UBERXL'],
                'Uber Premier': results['PREMIER'],
                'Premier SUV': results['SUV_PREMIER']
            }
            
            return gui_results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return fallback prices
            return {
                'UberX': 25.50,
                'UberXL': 35.75,
                'Uber Premier': 45.20,
                'Premier SUV': 55.90
            }
    
    def predict_price(self, distance_km, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                     hour_of_day=None, day_of_week=None, surge_multiplier=1.0,
                     traffic_level='moderate', weather_condition='clear'):
        """
        Single price prediction (for backward compatibility)
        Returns UberX price by default
        """
        if not self.is_loaded:
            return 25.50  # Fallback price
        
        if hour_of_day is None:
            hour_of_day = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        try:
            result = self.model.predict(
                pickup_lat=pickup_lat,
                pickup_lng=pickup_lng,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                service_type='UBERX',
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                traffic_level=traffic_level,
                weather_condition=weather_condition
            )
            
            # Apply surge multiplier if provided
            price = result['predicted_price'] * surge_multiplier
            return price
            
        except Exception as e:
            print(f"❌ Single prediction error: {e}")
            return 25.50  # Fallback price
    
    def predict(self, **kwargs):
        """
        General predict method (for maximum compatibility)
        """
        # Check if this is a multi-service call
        if 'service_type' in kwargs:
            service = kwargs.get('service_type', 'UBERX')
            
            try:
                result = self.model.predict(
                    pickup_lat=kwargs.get('pickup_lat'),
                    pickup_lng=kwargs.get('pickup_lng'),
                    dropoff_lat=kwargs.get('dropoff_lat'),
                    dropoff_lng=kwargs.get('dropoff_lng'),
                    service_type=service,
                    hour_of_day=kwargs.get('hour_of_day', datetime.now().hour),
                    day_of_week=kwargs.get('day_of_week', datetime.now().weekday()),
                    traffic_level=kwargs.get('traffic_level', 'moderate'),
                    weather_condition=kwargs.get('weather_condition', 'clear')
                )
                
                return result['predicted_price']
                
            except Exception as e:
                print(f"❌ Service prediction error: {e}")
                return 25.50
        else:
            # Legacy single prediction
            return self.predict_price(**kwargs)
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {
                'status': 'Not loaded',
                'model_type': 'XGBoost',
                'features': 'Unknown'
            }
        
        return {
            'status': 'Loaded',
            'model_type': 'XGBoost Miami Multi-Service',
            'features': len(self.model.feature_columns) if self.model.feature_columns else 'Unknown',
            'services': ['UBERX', 'UBERXL', 'PREMIER', 'SUV_PREMIER']
        }


class XGBoostModelWrapper:
    """
    Direct wrapper class that mimics the interface expected by the GUI
    """
    
    def __init__(self):
        self.api = XGBoostPricingAPI()
    
    def predict_all_services(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                            hour_of_day=None, day_of_week=None,
                            traffic_level='moderate', weather_condition='clear'):
        """GUI-compatible prediction method"""
        return self.api.predict_all_services(
            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
            hour_of_day, day_of_week, traffic_level, weather_condition
        )
    
    def predict_price(self, **kwargs):
        """GUI-compatible price prediction"""
        return self.api.predict_price(**kwargs)
    
    def predict(self, **kwargs):
        """GUI-compatible general prediction"""
        return self.api.predict(**kwargs)


def test_api():
    """Test the API functionality"""
    print("🧪 Testing XGBoost Pricing API")
    print("="*50)
    
    api = XGBoostPricingAPI()
    
    # Test model info
    info = api.get_model_info()
    print(f"Model Status: {info['status']}")
    print(f"Model Type: {info['model_type']}")
    print(f"Features: {info['features']}")
    
    if api.is_loaded:
        # Test predictions
        print("\n📍 Test Route: Miami Airport → South Beach")
        prices = api.predict_all_services(
            pickup_lat=25.7959, pickup_lng=-80.2870,  # Airport
            dropoff_lat=25.7907, dropoff_lng=-80.1300,  # South Beach
            hour_of_day=14,
            day_of_week=2,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        for service, price in prices.items():
            print(f"   {service}: ${price:.2f}")
        
        # Test single prediction
        single_price = api.predict_price(
            distance_km=19.8,
            pickup_lat=25.7959, pickup_lng=-80.2870,
            dropoff_lat=25.7907, dropoff_lng=-80.1300,
            hour_of_day=14,
            day_of_week=2,
            traffic_level='moderate',
            weather_condition='clear'
        )
        
        print(f"\nSingle UberX prediction: ${single_price:.2f}")
    
    print("\n✅ API test complete!")


if __name__ == "__main__":
    test_api()