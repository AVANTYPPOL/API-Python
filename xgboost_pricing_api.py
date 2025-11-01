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
from datetime import datetime
from zoneinfo import ZoneInfo

# Try to import googlemaps, but make it optional
try:
    import googlemaps
    GOOGLEMAPS_AVAILABLE = True
except ImportError:
    GOOGLEMAPS_AVAILABLE = False
    print("⚠️  googlemaps module not available - will use Haversine distance")

# Load environment variables
load_dotenv()

# Simple import - files are in the same directory
from xgboost_miami_model import XGBoostMiamiModel
import joblib
import numpy as np
import pickle

# Miami timezone (Eastern Time - handles EDT/EST automatically)
MIAMI_TZ = ZoneInfo("America/New_York")

def get_miami_time():
    """Get current time in Miami timezone (Eastern Time)"""
    return datetime.now(MIAMI_TZ)

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
        if google_api_key and GOOGLEMAPS_AVAILABLE:
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
            # Check if model file exists and log detailed info
            print(f"🔍 Looking for model file: {self.model_path}")
            print(f"🔍 Current working directory: {os.getcwd()}")
            print(f"🔍 Files in current directory: {os.listdir('.')[:10]}")
            
            if os.path.exists(self.model_path):
                print(f"✅ Model file found: {self.model_path}")
                file_size = os.path.getsize(self.model_path)
                print(f"📦 Model file size: {file_size} bytes")
                
                self.model = XGBoostMiamiModel()
                self.model.load_model(self.model_path)
                self.is_loaded = True
                print("✅ XGBoost model loaded successfully")
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("📁 Available files:")
                for f in os.listdir('.'):
                    if f.endswith('.pkl'):
                        print(f"   Found .pkl file: {f}")
                print("   Using fallback pricing...")
                self.is_loaded = False
        except Exception as e:
            print(f"❌ Failed to load XGBoost model: {e}")
            import traceback
            traceback.print_exc()
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
        """Get real driving distance and traffic from Google Maps API"""
        if self.gmaps:
            try:
                # Request with traffic model for better accuracy
                result = self.gmaps.distance_matrix(
                    origins=[(pickup_lat, pickup_lng)],
                    destinations=[(dropoff_lat, dropoff_lng)],
                    mode="driving",
                    units="metric",
                    departure_time="now",  # Required for traffic data
                    traffic_model="best_guess"  # Use best guess for traffic
                )

                if result['status'] == 'OK' and result['rows'][0]['elements'][0]['status'] == 'OK':
                    element = result['rows'][0]['elements'][0]

                    # Get distance
                    distance_m = element['distance']['value']
                    distance_km = distance_m / 1000

                    # Get duration and traffic duration
                    duration_s = element['duration']['value']

                    # Determine traffic level from duration_in_traffic
                    traffic_level = 'moderate'  # default

                    if 'duration_in_traffic' in element:
                        traffic_duration = element['duration_in_traffic']['value']
                        traffic_ratio = traffic_duration / duration_s if duration_s > 0 else 1.0

                        # Traffic level thresholds
                        if traffic_ratio >= 1.4:
                            traffic_level = 'heavy'
                        elif traffic_ratio >= 1.15:
                            traffic_level = 'moderate'
                        else:
                            traffic_level = 'light'

                        print(f"   Traffic ratio: {traffic_ratio:.2f}x (normal: {duration_s/60:.1f}min, with traffic: {traffic_duration/60:.1f}min)")
                    else:
                        # Estimate based on time of day if duration_in_traffic not available
                        miami_time = get_miami_time()
                        hour = miami_time.hour
                        if hour in [7, 8, 9, 16, 17, 18, 19]:  # Rush hours
                            traffic_level = 'moderate'
                        else:
                            traffic_level = 'light'

                    return distance_km, traffic_level

            except Exception as e:
                print(f"⚠️  Google Maps API error: {e}")
                import traceback
                traceback.print_exc()

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
                            traffic_level=None, weather_condition=None):
        """
        Predict prices for all service types with intelligent real-time context

        User provides: Just coordinates
        Backend fills: Time, traffic, weather from real-time APIs
        """
        if not self.is_loaded:
            # Return fallback prices if model not loaded
            return {
                'PREMIER': 45.20,
                'SUV_PREMIER': 55.90,
                'UBERX': 25.50,
                'UBERXL': 35.75
            }

        # STEP 1: Get current time if not provided (MIAMI LOCAL TIME)
        miami_time = get_miami_time()

        if hour_of_day is None:
            hour_of_day = miami_time.hour
            print(f"⏰ Using current Miami time: {miami_time.strftime('%I:%M %p')} (hour={hour_of_day})")

        if day_of_week is None:
            day_of_week = miami_time.weekday()
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            print(f"📅 Using current day: {days[day_of_week]}")

        # STEP 2: Get real-time TRAFFIC from Google Maps (if available)
        if traffic_level is None:
            real_distance, real_traffic = self.get_real_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
            if real_traffic:
                traffic_level = real_traffic
                print(f"🚗 Real-time traffic from Google Maps: {traffic_level.upper()}")
            else:
                traffic_level = 'moderate'
                print(f"⚠️  Google Maps unavailable, using default: moderate traffic")

        # STEP 3: Get real-time WEATHER from OpenWeatherMap (if available)
        if weather_condition is None:
            real_weather = self.get_current_weather(pickup_lat, pickup_lng)
            weather_condition = real_weather
            if self.weather_api_key:
                print(f"🌤️  Real-time weather from OpenWeatherMap: {weather_condition}")
            else:
                print(f"⚠️  Weather API unavailable, using default: {weather_condition}")
        
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
            
            # Return results in internal format, ensure JSON serializable
            gui_results = {
                'PREMIER': float(results['PREMIER']),
                'SUV_PREMIER': float(results['SUV_PREMIER']),
                'UBERX': float(results['UBERX']),
                'UBERXL': float(results['UBERXL'])
            }
            
            return gui_results
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return fallback prices
            return {
                'PREMIER': 45.20,
                'SUV_PREMIER': 55.90,
                'UBERX': 25.50,
                'UBERXL': 35.75
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

        miami_time = get_miami_time()
        if hour_of_day is None:
            hour_of_day = miami_time.hour
        if day_of_week is None:
            day_of_week = miami_time.weekday()
        
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
            miami_time = get_miami_time()

            try:
                result = self.model.predict(
                    pickup_lat=kwargs.get('pickup_lat'),
                    pickup_lng=kwargs.get('pickup_lng'),
                    dropoff_lat=kwargs.get('dropoff_lat'),
                    dropoff_lng=kwargs.get('dropoff_lng'),
                    service_type=service,
                    hour_of_day=kwargs.get('hour_of_day', miami_time.hour),
                    day_of_week=kwargs.get('day_of_week', miami_time.weekday()),
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