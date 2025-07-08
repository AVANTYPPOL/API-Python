"""
Hybrid Address-Based Uber Price Prediction Interface
===================================================

This interface uses the hybrid transfer learning model to provide
accurate price predictions based on pickup and dropoff addresses.

Features:
- Address to coordinates conversion (Google Geocoding API)
- Real driving distance calculation (Google Maps API)
- Hybrid ML model predictions (NYC + Miami data)
- Real-time traffic and weather integration
- Competitive pricing (20% below Uber)

Author: AI Assistant
Date: 2024
"""

import requests
import json
from datetime import datetime
from models.hybrid_uber_model import HybridUberPriceModel
import os

class HybridAddressPricingInterface:
    """
    Address-based pricing interface using hybrid ML model
    """
    
    def __init__(self):
        # API Keys from environment variables
        self.google_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
        self.weather_api_key = os.environ.get('WEATHER_API_KEY')
        
        if not self.google_api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
        if not self.weather_api_key:
            raise ValueError("WEATHER_API_KEY environment variable not set")
        
        # Initialize hybrid model
        self.hybrid_model = HybridUberPriceModel()
        
        # Try to load pre-trained model
        if os.path.exists('hybrid_uber_model.pkl'):
            print("🔄 Loading pre-trained hybrid model...")
            self.hybrid_model.load_model('hybrid_uber_model.pkl')
        else:
            print("🏗️  No pre-trained model found. Training new hybrid model...")
            success = self.hybrid_model.train_full_pipeline()
            if success:
                self.hybrid_model.save_model('hybrid_uber_model.pkl')
            else:
                print("❌ Failed to train hybrid model")
        
        print("✅ Hybrid Address Pricing Interface Ready!")
    
    def geocode_address(self, address):
        """
        Convert address to coordinates using Google Geocoding API
        """
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                'address': address,
                'key': self.google_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['status'] == 'OK' and len(data['results']) > 0:
                location = data['results'][0]['geometry']['location']
                formatted_address = data['results'][0]['formatted_address']
                
                return {
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'formatted_address': formatted_address,
                    'success': True
                }
            else:
                return {'success': False, 'error': f"Geocoding failed: {data.get('status', 'Unknown error')}"}
                
        except Exception as e:
            return {'success': False, 'error': f"Geocoding error: {str(e)}"}
    
    def get_driving_distance_and_traffic(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """
        Get real driving distance and traffic info using Google Maps API
        """
        try:
            url = "https://maps.googleapis.com/maps/api/distancematrix/json"
            params = {
                'origins': f"{pickup_lat},{pickup_lng}",
                'destinations': f"{dropoff_lat},{dropoff_lng}",
                'mode': 'driving',
                'departure_time': 'now',
                'traffic_model': 'best_guess',
                'key': self.google_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if (data['status'] == 'OK' and 
                len(data['rows']) > 0 and 
                len(data['rows'][0]['elements']) > 0):
                
                element = data['rows'][0]['elements'][0]
                
                if element['status'] == 'OK':
                    # Extract distance and duration
                    distance_m = element['distance']['value']
                    distance_km = distance_m / 1000
                    
                    duration_s = element['duration']['value']
                    duration_min = duration_s / 60
                    
                    # Check for traffic info
                    traffic_duration_s = element.get('duration_in_traffic', {}).get('value', duration_s)
                    traffic_duration_min = traffic_duration_s / 60
                    
                    # Determine traffic level
                    if traffic_duration_min <= duration_min * 1.1:
                        traffic_level = 'light'
                    elif traffic_duration_min <= duration_min * 1.3:
                        traffic_level = 'moderate'
                    else:
                        traffic_level = 'heavy'
                    
                    return {
                        'distance_km': distance_km,
                        'duration_min': duration_min,
                        'traffic_duration_min': traffic_duration_min,
                        'traffic_level': traffic_level,
                        'success': True
                    }
            
            # Fallback to Haversine distance
            print("⚠️  Google Maps API failed, using Haversine distance")
            return self._fallback_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
            
        except Exception as e:
            print(f"⚠️  Distance API error: {e}, using fallback")
            return self._fallback_distance(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    
    def _fallback_distance(self, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        """
        Fallback to Haversine distance calculation
        """
        import math
        
        # Haversine formula
        lat1, lng1, lat2, lng2 = map(math.radians, [pickup_lat, pickup_lng, dropoff_lat, dropoff_lng])
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = 6371 * c  # Earth radius in km
        
        return {
            'distance_km': distance_km,
            'duration_min': distance_km * 3,  # Rough estimate: 20 km/h average
            'traffic_duration_min': distance_km * 3,
            'traffic_level': 'light',
            'success': True
        }
    
    def get_weather_condition(self, lat, lng):
        """
        Get current weather condition
        """
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lng,
                'appid': self.weather_api_key,
                'units': 'imperial'  # Get temperature in Fahrenheit
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if response.status_code == 200:
                weather_main = data['weather'][0]['main'].lower()
                temperature = data['main']['temp']
                
                if 'rain' in weather_main or 'drizzle' in weather_main:
                    condition = 'rain'
                elif 'cloud' in weather_main:
                    condition = 'clouds'
                else:
                    condition = 'clear'
                
                return {
                    'condition': condition,
                    'temperature': temperature
                }
            else:
                return {
                    'condition': 'clear',
                    'temperature': 75.0  # Default temperature
                }
                
        except Exception as e:
            print(f"⚠️  Weather API error: {e}")
            return {
                'condition': 'clear',
                'temperature': 75.0  # Default temperature
            }
    
    def get_price_prediction(self, pickup_address, dropoff_address, 
                           hour_of_day=None, day_of_week=None, surge_multiplier=1.0):
        """
        Get price prediction from pickup and dropoff addresses
        """
        print(f"\n🚗 Getting Price Prediction")
        print(f"📍 From: {pickup_address}")
        print(f"📍 To: {dropoff_address}")
        print("=" * 50)
        
        # Step 1: Geocode addresses
        print("🔍 Step 1: Converting addresses to coordinates...")
        
        pickup_geo = self.geocode_address(pickup_address)
        if not pickup_geo['success']:
            return {'success': False, 'error': f"Pickup address error: {pickup_geo['error']}"}
        
        dropoff_geo = self.geocode_address(dropoff_address)
        if not dropoff_geo['success']:
            return {'success': False, 'error': f"Dropoff address error: {dropoff_geo['error']}"}
        
        print(f"   ✅ Pickup: {pickup_geo['formatted_address']}")
        print(f"   ✅ Dropoff: {dropoff_geo['formatted_address']}")
        
        # Step 2: Get driving distance and traffic
        print("\n🛣️  Step 2: Calculating driving distance and traffic...")
        
        distance_info = self.get_driving_distance_and_traffic(
            pickup_geo['lat'], pickup_geo['lng'],
            dropoff_geo['lat'], dropoff_geo['lng']
        )
        
        if not distance_info['success']:
            return {'success': False, 'error': "Failed to calculate distance"}
        
        print(f"   📏 Distance: {distance_info['distance_km']:.2f} km")
        print(f"   ⏱️  Duration: {distance_info['duration_min']:.1f} min")
        print(f"   🚦 Traffic: {distance_info['traffic_level']}")
        
        # Step 3: Get weather condition
        print("\n🌤️  Step 3: Getting weather condition...")
        
        weather_data = self.get_weather_condition(pickup_geo['lat'], pickup_geo['lng'])
        weather_condition = weather_data['condition']
        temperature = weather_data['temperature']
        print(f"   🌤️  Weather: {weather_condition} ({temperature:.0f}°F)")
        
        # Step 4: Set time parameters
        if hour_of_day is None:
            hour_of_day = datetime.now().hour
        if day_of_week is None:
            day_of_week = datetime.now().weekday()
        
        print(f"\n⏰ Step 4: Time parameters...")
        print(f"   🕐 Hour: {hour_of_day}")
        print(f"   📅 Day of week: {day_of_week}")
        print(f"   📈 Surge: {surge_multiplier}x")
        
        # Step 5: Get hybrid model prediction
        print(f"\n🤖 Step 5: Hybrid ML prediction...")
        
        predicted_price = self.hybrid_model.predict_price(
            distance_km=distance_info['distance_km'],
            pickup_lat=pickup_geo['lat'],
            pickup_lng=pickup_geo['lng'],
            dropoff_lat=dropoff_geo['lat'],
            dropoff_lng=dropoff_geo['lng'],
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            surge_multiplier=surge_multiplier,
            traffic_level=distance_info['traffic_level'],
            weather_condition=weather_condition
        )
        
        # Step 6: Apply competitive pricing (20% below Uber)
        competitive_price = predicted_price * 0.8
        
        print(f"   🎯 Uber prediction: ${predicted_price:.2f}")
        print(f"   💰 Our price (20% off): ${competitive_price:.2f}")
        
        # Calculate price per km
        price_per_km = competitive_price / distance_info['distance_km']
        
        # Return comprehensive result
        result = {
            'success': True,
            'pickup_address': pickup_geo['formatted_address'],
            'dropoff_address': dropoff_geo['formatted_address'],
            'pickup_coordinates': {'lat': pickup_geo['lat'], 'lng': pickup_geo['lng']},
            'dropoff_coordinates': {'lat': dropoff_geo['lat'], 'lng': dropoff_geo['lng']},
            'distance_km': distance_info['distance_km'],
            'duration_min': distance_info['duration_min'],
            'traffic_level': distance_info['traffic_level'],
            'weather_condition': weather_condition,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'surge_multiplier': surge_multiplier,
            'uber_predicted_price': predicted_price,
            'our_competitive_price': competitive_price,
            'price_per_km': price_per_km,
            'savings': predicted_price - competitive_price
        }
        
        print(f"\n✅ Price Prediction Complete!")
        print(f"💵 Final Price: ${competitive_price:.2f}")
        print(f"💰 You Save: ${result['savings']:.2f}")
        
        return result

def main():
    """
    Interactive demo of the hybrid address pricing interface
    """
    print("🚗 HYBRID ADDRESS-BASED UBER PRICE PREDICTION")
    print("=" * 60)
    
    # Initialize interface
    interface = HybridAddressPricingInterface()
    
    # Test cases
    test_cases = [
        ("Miami International Airport", "Ocean Drive, Miami Beach"),
        ("Downtown Miami", "Wynwood, Miami"),
        ("Brickell City Centre", "University of Miami"),
        ("Port of Miami", "Miami Beach Convention Center")
    ]
    
    print(f"\n🧪 Testing Hybrid Model with Real Miami Addresses:")
    print("=" * 60)
    
    for i, (pickup, dropoff) in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        
        result = interface.get_price_prediction(pickup, dropoff)
        
        if result['success']:
            print(f"\n📊 RESULTS:")
            print(f"   Distance: {result['distance_km']:.2f} km")
            print(f"   Duration: {result['duration_min']:.1f} min")
            print(f"   Traffic: {result['traffic_level']}")
            print(f"   Weather: {result['weather_condition']}")
            print(f"   Uber Price: ${result['uber_predicted_price']:.2f}")
            print(f"   Our Price: ${result['our_competitive_price']:.2f}")
            print(f"   Savings: ${result['savings']:.2f}")
            print(f"   Price/km: ${result['price_per_km']:.2f}")
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 60)
    
    # Interactive mode
    print(f"\n🎮 Interactive Mode:")
    print("Enter addresses to get price predictions (or 'quit' to exit)")
    
    while True:
        try:
            pickup = input("\n📍 Pickup address: ").strip()
            if pickup.lower() == 'quit':
                break
            
            dropoff = input("📍 Dropoff address: ").strip()
            if dropoff.lower() == 'quit':
                break
            
            result = interface.get_price_prediction(pickup, dropoff)
            
            if result['success']:
                print(f"\n🎯 PRICE QUOTE:")
                print(f"   From: {result['pickup_address']}")
                print(f"   To: {result['dropoff_address']}")
                print(f"   Distance: {result['distance_km']:.2f} km")
                print(f"   Our Price: ${result['our_competitive_price']:.2f}")
                print(f"   You Save: ${result['savings']:.2f} vs Uber")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            break
    
    print(f"\n👋 Thanks for using Hybrid Address Pricing Interface!")

if __name__ == "__main__":
    main() 